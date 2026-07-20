"""Build a compact on-device laptop catalog (laptops.db) from the
Amazon Reviews 2023 dataset (https://amazon-reviews-2023.github.io).

Route A of the VoiceShop++ data plan: keep only laptop-like products from
the *Electronics* item metadata, apply rule-based extraction (regex over
`details` / `features`) to fill the fields the Android UI needs, and write
a small SQLite file that the app loads from its assets folder.

TWO WAYS TO GET THE INPUT DATA
------------------------------
1) OFFLINE (recommended when the machine running Python has no internet):
   Download the metadata file on any machine that has internet:

     meta_Electronics.jsonl.gz  (~1.31 GB, 2023 version)
     https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz

   You do NOT need the whole file. Since it is line-based JSONL, you can
   download only the first chunk (e.g. ~200 MB) and this script will read
   what it can and stop after finding enough laptops:

     curl.exe -r 0-209715200 -o meta_Electronics_part.jsonl.gz "<url above>"

   Then run:
     python build_laptops.py --input meta_Electronics_part.jsonl.gz --limit 500 --out laptops.db

2) ONLINE (needs internet + `pip install datasets`):
     python build_laptops.py --limit 500 --out laptops.db

No LLM / API key is used. Missing fields are left empty and the app hides
them gracefully. `average_rating` / `rating_number` come from the metadata,
so the huge review files are never needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from typing import Any, Iterable


DATASET = "McAuley-Lab/Amazon-Reviews-2023"
META_CONFIG = "raw_meta_Electronics"  # laptops live inside Electronics

LAPTOP_KEYWORDS = ("laptop", "notebook", "chromebook", "macbook", "ultrabook")
# Accessory phrasing: a product that is FOR a laptop, not a laptop itself,
# e.g. "Sun Shade for 15-16 Laptops", "lid for 15-Inch Inspiron Laptop".
ACCESSORY_PHRASES = ("compatible with",)
_FOR_LAPTOP_RE = re.compile(r"\bfor\b.{0,40}(laptop|notebook|macbook|chromebook)")
# Words that usually mean "accessory / part", not an actual laptop.
EXCLUDE_KEYWORDS = (
    "case", "sleeve", "cover", "bag", "backpack", "charger", "adapter",
    "cable", "cooling pad", "screen protector", "protector", "sticker",
    "skin", "keyboard", "mouse", "replacement", "sodimm", "dimm", "ddr",
    "ssd", "hard drive", "hdd", "docking", "usb hub", "cleaner",
    "power strip", "power bank", "surge", "tablet", "drawing",
    "earbud", "earphone", "headphone", "headset", "speaker", "webcam",
    "microphone", "tripod", "stylus", "memory card", "flash drive",
    "battery", "lcd", "digitizer", "decal",
    " stand", "cooler", "cool pad", "cooling", "projector", "riser",
    "mount", "holder", "lap desk", "pad-", "tray", "pillow",
    "filter", "screwdriver", "portfolio", "briefcase", "tote", "messenger",
    "lid", "hood", "shade", "privacy", "daypack", "rucksack", "grip",
)


def looks_like_laptop(item: dict[str, Any]) -> bool:
    title = (item.get("title") or "").lower()
    if not title:
        return False
    # Reject accessories that merely mention a laptop.
    if any(phrase in title for phrase in ACCESSORY_PHRASES):
        return False
    if _FOR_LAPTOP_RE.search(title):
        return False
    # Require a laptop keyword in the TITLE (not just categories) for precision.
    if not any(kw in title for kw in LAPTOP_KEYWORDS):
        return False
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False
    # NOTE: 2023 metadata often has price == "None". Do NOT require a price,
    # otherwise most real laptops get dropped and you end up with 0 rows.
    return True


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
    return (
        item.get("parent_asin") or item.get("asin") or "",
        (item.get("title") or "").strip()[:90],
        _price_of(item),
        round(rating, 1),
        rating_number,
        build_display(details),
        build_performance(details),
        build_battery(details, features),
        parse_weight_kg(details),
        summary,
        review_sentiment(rating, rating_number),
        "",  # weakness: no reliable rule-based source; UI hides when empty
        "||".join(build_reasons(features)),
        "",  # trade_offs: no reliable rule-based source; UI hides when empty
        (item.get("store") or "").strip()[:60],
        _first_image(item),
        detect_platform(item.get("title") or "", details),
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
            if not looks_like_laptop(item):
                continue
            yield item
            kept += 1
            if kept >= limit:
                break
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        print(f"  note: input ended early after scanning {scanned} rows ({exc})")
    finally:
        fp.close()


def iter_online(limit: int) -> Iterable[dict[str, Any]]:
    """Fallback: stream from HuggingFace (requires internet + datasets)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Online mode needs the 'datasets' package. Either run:\n"
            "  pip install datasets\n"
            "or use offline mode with --input <meta_Electronics.jsonl.gz>."
        ) from exc
    stream = load_dataset(DATASET, META_CONFIG, split="full", streaming=True)
    kept = 0
    for item in stream:
        if not looks_like_laptop(item):
            continue
        yield item
        kept += 1
        if kept >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Build laptops.db for VoiceShop++")
    parser.add_argument("--limit", type=int, default=500, help="max laptops to keep")
    parser.add_argument(
        "--input",
        default=None,
        help="local meta_*.jsonl(.gz) file. If omitted, streams online via datasets.",
    )
    parser.add_argument("--out", default="laptops.db", help="output SQLite path")
    args = parser.parse_args()

    source = iter_local(args.input, args.limit) if args.input else iter_online(args.limit)
    if args.input:
        print(f"Reading laptops from local file: {args.input}")
    else:
        print("Streaming laptops online from HuggingFace ...")

    conn = sqlite3.connect(args.out)
    try:
        create_schema(conn)
        inserted = 0
        seen: set[str] = set()
        for item in source:
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
            if inserted % 50 == 0:
                print(f"  ... {inserted} laptops")
        conn.commit()
        print(f"Done. Wrote {inserted} laptops to {args.out}")
        if inserted == 0:
            print(
                "WARNING: 0 laptops found. If you used a partial download, grab a\n"
                "bigger chunk (laptops may not appear in the very first rows)."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
