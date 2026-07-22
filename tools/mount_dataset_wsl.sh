#!/usr/bin/env bash
# Mount the ENTIRE Amazon Reviews 2023 metadata directory (all 33+ categories)
# inside WSL as a read-only, on-demand filesystem using rclone's http backend.
#
# Nothing is fully downloaded: the files just "appear" under the mountpoint, and
# only the bytes actually read are fetched over the network. Because the catalog
# builder keeps ~500 products per category, it reads only the small prefix of
# each file — so building a full multi-category catalog transfers only tens of MB.
#
# No sudo required (WSL already ships fusermount3; rclone installs to ~/.local/bin).
#
# Usage (inside WSL):
#   bash /mnt/c/Users/f0407/Desktop/VE441-G8-main/tools/mount_dataset_wsl.sh          # mount only
#   bash /mnt/c/Users/f0407/Desktop/VE441-G8-main/tools/mount_dataset_wsl.sh --build  # mount + build catalog.db
#   bash .../mount_dataset_wsl.sh --umount                                            # unmount
set -euo pipefail

BASE_HOST="https://mcauleylab.ucsd.edu"
REMOTE_PATH="public_datasets/data/amazon_2023/raw/meta_categories"
MOUNT="${HOME}/amazon_meta"
BIN_DIR="${HOME}/.local/bin"
RCLONE="${BIN_DIR}/rclone"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DB="${PROJECT_DIR}/backend/data/catalog.db"

if [ "${1:-}" = "--umount" ]; then
  fusermount3 -u "$MOUNT" 2>/dev/null || fusermount -u "$MOUNT" 2>/dev/null || true
  echo "Unmounted $MOUNT"
  exit 0
fi

mkdir -p "$BIN_DIR" "$MOUNT" "$(dirname "$OUT_DB")"

# 1. Install rclone to ~/.local/bin if missing (no sudo).
if ! command -v rclone >/dev/null 2>&1 && [ ! -x "$RCLONE" ]; then
  echo "Installing rclone into $BIN_DIR (no sudo) ..."
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/rclone.zip" https://downloads.rclone.org/rclone-current-linux-amd64.zip
  python3 - "$tmp/rclone.zip" "$BIN_DIR" <<'PY'
import sys, zipfile, os
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    member = next(n for n in z.namelist() if n.rstrip("/").endswith("rclone"))
    data = z.read(member)
out = os.path.join(dest, "rclone")
with open(out, "wb") as f:
    f.write(data)
os.chmod(out, 0o755)
print("installed:", out)
PY
  rm -rf "$tmp"
fi
export PATH="$BIN_DIR:$PATH"
RCLONE_BIN="$(command -v rclone || echo "$RCLONE")"
echo "rclone: $($RCLONE_BIN version | head -n1)"

# 2. Define an http remote via env (no config file, no secrets).
export RCLONE_CONFIG_AMZ_TYPE=http
export RCLONE_CONFIG_AMZ_URL="$BASE_HOST"

# 3. Mount the whole category directory (on-demand fetch).
if mountpoint -q "$MOUNT"; then
  echo "Already mounted at $MOUNT"
else
  echo "Mounting whole dataset directory at $MOUNT ..."
  "$RCLONE_BIN" mount "amz:${REMOTE_PATH}" "$MOUNT" \
    --read-only --daemon \
    --dir-cache-time 9999h --attr-timeout 1s \
    --vfs-cache-mode off --buffer-size 0
  for _ in $(seq 1 30); do mountpoint -q "$MOUNT" && break; sleep 0.5; done
fi

if ! mountpoint -q "$MOUNT"; then
  echo "ERROR: mount failed. Check network to $BASE_HOST." >&2
  exit 1
fi

echo
echo "All categories are now visible in WSL (read-only, fetched on demand):"
ls -lh "$MOUNT" | sed -n '1,8p'
echo "  ...(list truncated)"
echo
echo "Mountpoint: $MOUNT"

BUILD_CMD=(python3 "$SCRIPT_DIR/build_laptops.py"
  --input "$MOUNT"/meta_*.jsonl.gz
  --per-input-limit 500 --limit 15000 --out "$OUT_DB")

if [ "${1:-}" = "--build" ]; then
  echo "Building catalog.db (reads only the first ~500 products per category)..."
  "${BUILD_CMD[@]}"
  echo "Done -> $OUT_DB"
else
  echo "To build the catalog now, run:"
  printf '  %q ' "${BUILD_CMD[@]}"; echo
fi
