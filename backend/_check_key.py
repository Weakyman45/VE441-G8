import os
from pathlib import Path

for p in (Path(".env"), Path("../.env")):
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
print("env_file_loaded", True)
print("DASHSCOPE_API_KEY", "SET len="+str(len(key)) if key and not key.startswith("sk-your") else "MISSING_OR_PLACEHOLDER")
print("QWEN_CHAT_MODEL", os.environ.get("QWEN_CHAT_MODEL") or "(default)")
