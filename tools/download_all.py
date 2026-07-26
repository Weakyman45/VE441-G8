"""Download a small chunk of EVERY Amazon Reviews 2023 category metadata file,
then build a combined VoiceShop++ product catalog (catalog.db).

Why only a chunk: the files are line-based JSON, and we only KEEP a few hundred
products per category (see --per-input-limit), so a small prefix of each file is
plenty. Downloading the whole multi-GB dataset is unnecessary.

Mirrors (pick with --source; default is the China-friendly hf-mirror):
  hfmirror : https://hf-mirror.com/...          (fast in mainland China)
  hf       : https://huggingface.co/...         (official HuggingFace)
  ucsd     : https://mcauleylab.ucsd.edu/...     (original, US server, slow from CN)

Usage (from the project root):
    python tools/download_all.py
    python tools/download_all.py --source hf --mb 80 --jobs 6
    python tools/download_all.py --categories Electronics Clothing_Shoes_and_Jewelry Cell_Phones_and_Accessories
    python tools/download_all.py --no-build        # only download
    python tools/download_all.py --skip-existing   # keep files already present

Only the Python standard library is used (urllib). No API key, no pip install.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Each source: (url_template, file_extension). HuggingFace hosts uncompressed
# .jsonl; UCSD hosts .jsonl.gz. The builder handles both automatically.
SOURCES = {
    "hfmirror": (
        "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw/meta_categories/meta_{category}.jsonl",
        ".jsonl",
    ),
    "hf": (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw/meta_categories/meta_{category}.jsonl",
        ".jsonl",
    ),
    "ucsd": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
        "raw/meta_categories/meta_{category}.jsonl.gz",
        ".jsonl.gz",
    ),
}

ALL_CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Video_Games",
]

HERE = os.path.dirname(os.path.abspath(__file__))
_print_lock = threading.Lock()


def _human(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.1f} MB"


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def download_partial(
    category: str, url_tmpl: str, out_path: str, max_bytes: int, *, skip_existing: bool
) -> str | None:
    """Download up to max_bytes of one category. Returns the file path or None."""
    if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        _log(f"  [skip] {os.path.basename(out_path)} ({_human(os.path.getsize(out_path))})")
        return out_path

    url = url_tmpl.format(category=category)
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "voiceshop-dl/1.0"},
    )
    written = 0
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(out_path, "wb") as fh:
                while written < max_bytes:
                    chunk = resp.read(min(1024 * 1024, max_bytes - written))
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
    except Exception as exc:  # noqa: BLE001 - keep going with other categories
        # A truncated file is still usable by the builder; keep it if non-empty.
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            _log(f"  [warn] {category}: {exc} (kept partial {_human(os.path.getsize(out_path))})")
            return out_path
        _log(f"  [FAIL] {category}: {exc}")
        return None

    secs = max(time.time() - start, 0.001)
    _log(f"  [ok]  {category:<32} {_human(written)}  ({_human(written / secs)}/s)")
    return out_path if written > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a chunk of every Amazon 2023 category, then build catalog.db"
    )
    parser.add_argument("--source", choices=list(SOURCES), default="hfmirror",
                        help="download mirror (default: hfmirror, fast in China)")
    parser.add_argument("--base-url", default=None,
                        help="override URL template; must contain {category}")
    parser.add_argument("--mb", type=int, default=80,
                        help="MB to download per category (default 80; enough for a few hundred kept products)")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")
    parser.add_argument("--dir", default=HERE, help="download directory (default: tools/)")
    parser.add_argument(
        "--out",
        default=os.path.normpath(os.path.join(HERE, "..", "backend", "data", "catalog.db")),
        help="output SQLite path (default: backend/data/catalog.db)",
    )
    parser.add_argument("--categories", nargs="+", default=None,
                        help="subset of categories to fetch (default: all).")
    parser.add_argument("--limit", type=int, default=15000, help="max total products in catalog.db")
    parser.add_argument("--per-input-limit", type=int, default=500,
                        help="max products kept per category (keeps the catalog balanced).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="don't re-download files already present")
    parser.add_argument("--no-build", action="store_true", help="only download; skip building catalog.db")
    args = parser.parse_args()

    url_tmpl, ext = SOURCES[args.source]
    if args.base_url:
        url_tmpl = args.base_url
        ext = ".jsonl.gz" if ".gz" in args.base_url else ".jsonl"

    categories = args.categories or ALL_CATEGORIES
    max_bytes = args.mb * 1024 * 1024
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    _log(f"Source: {args.source}  ({url_tmpl.split('{')[0]}...)")
    _log(f"Downloading up to {args.mb} MB of {len(categories)} categories "
         f"with {args.jobs} parallel jobs into {args.dir}")
    _log("(small categories smaller than the cap download in full — that's expected)\n")

    downloaded: list[str] = []
    done = 0
    total = len(categories)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                download_partial,
                cat,
                url_tmpl,
                os.path.join(args.dir, f"meta_{cat}_part{ext}"),
                max_bytes,
                skip_existing=args.skip_existing,
            ): cat
            for cat in categories
        }
        for fut in as_completed(futures):
            done += 1
            path = fut.result()
            if path:
                downloaded.append(path)
            _log(f"    progress: {done}/{total} done, {len(downloaded)} available")

    _log(f"\nDownloaded/available files: {len(downloaded)}/{total}")
    if not downloaded:
        _log("Nothing downloaded. Try a different --source (hf / ucsd) or check your network.")
        sys.exit(1)

    if args.no_build:
        _log("\n--no-build set: skipping catalog build.")
        _log("Build later with:")
        _log(f"  python {os.path.join('tools', 'build_laptops.py')} --input <files...> "
             f"--per-input-limit {args.per_input_limit} --limit {args.limit} --out {args.out}")
        return

    _log("\nBuilding catalog.db from all downloaded files ...")
    cmd = [
        sys.executable,
        os.path.join(HERE, "build_laptops.py"),
        "--input", *downloaded,
        "--per-input-limit", str(args.per_input_limit),
        "--limit", str(args.limit),
        "--out", args.out,
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
