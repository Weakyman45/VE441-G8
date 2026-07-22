"""Build a compact on-device product catalog (catalog.db) from the
Amazon Reviews 2023 dataset (https://amazon-reviews-2023.github.io).

Route A of the VoiceShop++ data plan: keep real products from one or more
category metadata files, apply best-effort rule-based extraction (regex over
`details` / `features`) to fill the fields the UI can use, and write a small
SQLite file the backend serves (and the app can bundle as an offline fallback).

NOTE (previously laptop-only): this builder no longer filters to laptops. It
keeps products from whatever `meta_*.jsonl(.gz)` files you feed it, so the app
can search any product category present in your inputs. Laptop-specific fields
(display / performance / battery / weight / platform) are filled when the item
looks like a computer and left empty otherwise — the UI hides empty fields.

GETTING THE INPUT DATA
----------------------
Amazon Reviews 2023 ships one metadata file PER CATEGORY. Download the ones you
want (each is line-based JSONL, so a partial download of the first chunk works):

  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz

Examples of <Category>: Electronics, Cell_Phones_and_Accessories,
Clothing_Shoes_and_Jewelry, Home_and_Kitchen, Sports_and_Outdoors,
Toys_and_Games, Video_Games, Office_Products, Beauty_and_Personal_Care ...
(full list: https://amazon-reviews-2023.github.io)

Download only the first chunk of each (line-based JSONL tolerates truncation):

  curl.exe -r 0-262144000 -o meta_Electronics_part.jsonl.gz "<url above>"

Then build a combined catalog from several files at once:

  python build_laptops.py --input meta_Electronics_part.jsonl.gz meta_Cell_Phones_and_Accessories_part.jsonl.gz --limit 2000 --out ../backend/data/catalog.db

ONLINE mode (needs internet + `pip install datasets`) streams whole categories:

  python build_laptops.py --online-category Electronics --online-category Video_Games --limit 2000 --out catalog.db

No LLM / API key is used. Missing fields are left empty and the app hides them
gracefully. `average_rating` / `rating_number` come from the metadata, so the
huge review files are never needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from typing import Any, Iterable


DATASET = "McAuley-Lab/Amazon-Reviews-2023"

# Computer keywords are only used to decide whether the laptop-specific fields
# (display / performance / battery / weight / platform) are worth extracting.
_COMPUTER_KEYWORDS = (
    "laptop", "notebook", "chromebook", "macbook", "ultrabook", "desktop",
    "pc ", "computer",
)

# Obvious non-products / placeholder titles we skip regardless of category.
_JUNK_TITLE_RE = re.compile(r"^\s*(unknown|n/?a|null|none)\s*$", re.IGNORECASE)


def looks_like_product(item: dict[str, Any]) -> bool:
    """Very permissive filter: keep anything with a usable title and id.

    We intentionally do NOT restrict by category here — the whole point is to
    let users search any product present in the input files.
    """
    title = (item.get("title") or "").strip()
    if not title or _JUNK_TITLE_RE.match(title):
        return False
    pid = item.get("parent_asin") or item.get("asin")
    if not pid:
        return False
    return True


def _looks_like_computer(title: str) -> bool:
    low = title.lower()
    return any(kw in low for kw in _COMPUTER_KEYWORDS)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _price_of(item: dict[str, Any]) -> int:
    raw = item.get("price")
    if raw in (None, "", "None"):
        return 0
    match = re.search(r"\d[\d,]*\.?\d*", str(raw))
    if not match:
        return 0
    try:
        return int(round(float(match.group(0).replace(",", ""))))
    except ValueError:
        return 0


def _details(item: dict[str, Any]) -> dict[str, str]:
    details = item.get("details")
    # In the 2023 dataset `details` is stored as a JSON *string*.
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    if not isinstance(details, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in details.items()}


def _first_match(details: dict[str, str], *keys: str) -> str:
    for key in keys:
        for actual, value in details.items():
            if key.lower() in actual.lower() and value:
                return value
    return ""


def build_display(details: dict[str, str]) -> str:
    parts = [
        _first_match(details, "Screen Size", "Standing screen display size", "Display Size"),
        _first_match(details, "Resolution", "Display Resolution", "Screen Resolution"),
        _first_match(details, "Display Type", "Display Technology"),
    ]
    return ", ".join(p for p in parts if p)


def build_performance(details: dict[str, str]) -> str:
    cpu = _first_match(details, "Processor", "CPU Model", "Processor Type")
    ram = _first_match(details, "RAM", "Memory", "Computer Memory Size", "Ram Memory")
    disk = _first_match(details, "Hard Drive", "Hard Disk", "SSD", "Flash Memory")
    gpu = _first_match(details, "Graphics", "Graphics Card", "Chipset Brand")
    parts = []
    if cpu:
        parts.append(cpu)
    if ram:
        parts.append(f"{ram} RAM" if "ram" not in ram.lower() else ram)
    if disk:
        parts.append(disk)
    if gpu:
        parts.append(gpu)
    return ", ".join(parts)


def build_battery(details: dict[str, str], features: list[str]) -> str:
    hours = _first_match(details, "Average Battery Life", "Battery Life")
    if hours:
        return hours if "hour" in hours.lower() else f"{hours} hours"
    for feature in features:
        if "battery" in feature.lower() and "hour" in feature.lower():
            return feature.strip()[:60]
    return ""


def parse_weight_kg(details: dict[str, str]) -> float:
    raw = _first_match(details, "Item Weight", "Weight", "Product Dimensions")
    if not raw:
        return 0.0
    lowered = raw.lower()
    match = re.search(r"(\d+\.?\d*)\s*(kg|kilograms|pounds|lbs|lb|ounces|oz|g\b|grams)", lowered)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit in ("kg", "kilograms"):
        return round(value, 2)
    if unit in ("pounds", "lbs", "lb"):
        return round(value * 0.453592, 2)
    if unit in ("ounces", "oz"):
        return round(value * 0.0283495, 2)
    if unit in ("g", "grams"):
        return round(value / 1000.0, 2)
    return 0.0


def detect_platform(title: str, details: dict[str, str]) -> str:
    """Only meaningful for computers; empty for other product categories."""
    if not _looks_like_computer(title):
        return ""
    text = f"{title} {_first_match(details, 'Operating System', 'OS')}".lower()
    if "macbook" in text or "mac os" in text or "macos" in text:
        return "macOS"
    if "chromebook" in text or "chrome os" in text:
        return "ChromeOS"
    return "Windows"


def build_reasons(features: list[str]) -> list[str]:
    reasons = []
    for feature in features:
        clean = re.sub(r"\s+", " ", feature).strip()
        if 10 <= len(clean) <= 90:
            reasons.append(clean)
        if len(reasons) >= 3:
            break
    return reasons


def review_sentiment(rating: float, rating_number: int) -> str:
    if rating_number <= 0:
        return ""
    if rating >= 4.5:
        return f"Highly rated across {rating_number:,} ratings"
    if rating >= 4.0:
        return f"Generally positive across {rating_number:,} ratings"
    if rating >= 3.0:
        return f"Mixed feedback across {rating_number:,} ratings"
    return f"Mostly critical across {rating_number:,} ratings"


def to_row(item: dict[str, Any]) -> tuple:
    details = _details(item)
    features = _as_list(item.get("features"))
    description = _as_list(item.get("description"))
    rating = float(item.get("average_rating") or 0.0)
    rating_number = int(item.get("rating_number") or 0)
    summary = (features[0] if features else (description[0] if description else "")).strip()[:120]
    title = (item.get("title") or "").strip()
    is_computer = _looks_like_computer(title)
    return (
        item.get("parent_asin") or item.get("asin") or "",
        title[:90],
        _price_of(item),
        round(rating, 1),
        rating_number,
        build_display(details) if is_computer else "",
        build_performance(details) if is_computer else "",
        build_battery(details, features) if is_computer else "",
        parse_weight_kg(details),
        summary,
        review_sentiment(rating, rating_number),
        "",  # weakness: no reliable rule-based source; UI hides when empty
        "||".join(build_reasons(features)),
        "",  # trade_offs: no reliable rule-based source; UI hides when empty
        (item.get("store") or "").strip()[:60],
        _first_image(item),
        detect_platform(title, details),
    )


def _first_image(item: dict[str, Any]) -> str:
    images = item.get("images")
    # 2023 metadata: images is a dict of parallel lists.
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            values = images.get(key)
            if isinstance(values, list):
                for v in values:
                    if v:
                        return str(v)
            elif isinstance(values, str) and values:
                return values
    if isinstance(images, list):
        for entry in images:
            if isinstance(entry, dict):
                for key in ("large", "hi_res", "thumb"):
                    if entry.get(key):
                        return str(entry[key])
    return ""


def create_schema(conn: sqlite3.Connection) -> None:
    # Table kept named `laptops` for backward compatibility with the backend
    # (server.py) and the Android offline reader. It now holds any product.
    conn.execute("DROP TABLE IF EXISTS laptops")
    conn.execute(
        """
        CREATE TABLE laptops (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price INTEGER NOT NULL,
            rating REAL NOT NULL,
            rating_number INTEGER NOT NULL,
            display TEXT,
            performance TEXT,
            battery TEXT,
            weight_kg REAL,
            summary TEXT,
            review_sentiment TEXT,
            weakness TEXT,
            reasons TEXT,
            trade_offs TEXT,
            store TEXT,
            image_url TEXT,
            platform TEXT
        )
        """
    )


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_local(path: str, limit: int) -> Iterable[dict[str, Any]]:
    """Read a local meta_*.jsonl or .jsonl.gz file line by line.

    Tolerates a truncated file (e.g. a partial download): it just stops
    when the stream ends or the last line cannot be parsed.
    """
    kept = 0
    scanned = 0
    fp = _open_text(path)
    try:
        for line in fp:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial/truncated final line
            if not looks_like_product(item):
                continue
            yield item
            kept += 1
            if kept >= limit:
                break
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        print(f"  note: input ended early after scanning {scanned} rows ({exc})")
    finally:
        fp.close()


def iter_online(category: str, limit: int) -> Iterable[dict[str, Any]]:
    """Fallback: stream one category from HuggingFace (needs internet + datasets)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Online mode needs the 'datasets' package. Either run:\n"
            "  pip install datasets\n"
            "or use offline mode with --input <meta_*.jsonl.gz>."
        ) from exc
    config = f"raw_meta_{category}"
    stream = load_dataset(DATASET, config, split="full", streaming=True)
    kept = 0
    for item in stream:
        if not looks_like_product(item):
            continue
        yield item
        kept += 1
        if kept >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a product catalog.db for VoiceShop++")
    parser.add_argument("--limit", type=int, default=2000, help="max total products to keep across all inputs")
    parser.add_argument(
        "--per-input-limit",
        type=int,
        default=None,
        help="cap products taken from EACH input file/category (keeps categories balanced).",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="one or more local meta_*.jsonl(.gz) files (any categories).",
    )
    parser.add_argument(
        "--online-category",
        action="append",
        default=None,
        metavar="CATEGORY",
        help="stream this category online (repeatable). Needs `pip install datasets`.",
    )
    parser.add_argument("--out", default="catalog.db", help="output SQLite path")
    args = parser.parse_args()

    if not args.input and not args.online_category:
        parser.error("provide --input <files...> (offline) or --online-category <name> (online)")

    conn = sqlite3.connect(args.out)
    try:
        create_schema(conn)
        inserted = 0
        seen: set[str] = set()

        def consume(source: Iterable[dict[str, Any]], label: str) -> None:
            nonlocal inserted
            before = inserted
            for item in source:
                if inserted >= args.limit:
                    break
                row = to_row(item)
                pid = row[0]
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                conn.execute(
                    "INSERT OR REPLACE INTO laptops VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
                inserted += 1
                if inserted % 100 == 0:
                    print(f"  ... {inserted} products")
            print(f"  [{label}] added {inserted - before} products")

        def source_cap() -> int:
            remaining = args.limit - inserted
            if args.per_input_limit:
                return min(remaining, args.per_input_limit)
            return remaining

        if args.input:
            for path in args.input:
                if inserted >= args.limit:
                    break
                print(f"Reading products from local file: {path}")
                consume(iter_local(path, source_cap()), path)
        if args.online_category and inserted < args.limit:
            for category in args.online_category:
                if inserted >= args.limit:
                    break
                print(f"Streaming category online: {category}")
                consume(iter_online(category, source_cap()), category)

        conn.commit()
        print(f"Done. Wrote {inserted} products to {args.out}")
        if inserted == 0:
            print(
                "WARNING: 0 products found. If you used a partial download, grab a\n"
                "bigger chunk, or check the input file path / category name."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
