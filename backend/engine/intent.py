from __future__ import annotations

import re
from typing import Iterable

from .models import PreferenceProfile


def extract_preference(text: str, prior: PreferenceProfile | None = None) -> PreferenceProfile:
    """Rule-based preference extraction (Phase 1 — no LLM required)."""
    profile = PreferenceProfile(
        category=(prior.category if prior else "laptop"),
        budget=prior.budget if prior else None,
        use_case=prior.use_case if prior else "",
        platform=prior.platform if prior else "No preference",
        touch=prior.touch if prior else "Not required",
        hard=list(prior.hard) if prior else [],
        soft=list(prior.soft) if prior else [],
        visual_context=prior.visual_context if prior else "",
        raw_query=text.strip(),
    )
    lower = text.lower()

    budgets = [
        int(m.group(1).replace(",", ""))
        for m in re.finditer(r"(?<!\d)(\d[\d,]{2,5})(?!\d)", text)
    ]
    budgets = [b for b in budgets if 500 <= b <= 50000]
    # Also catch "under 1500" / "$1200"
    for m in re.finditer(r"(?:under|below|预算|以内|左右|about|around)\s*\$?\s*(\d{3,5})", lower):
        try:
            budgets.append(int(m.group(1)))
        except ValueError:
            pass
    if budgets:
        profile.budget = min(budgets)
        _upsert(profile.hard, f"Budget up to {profile.budget}")

    if any(k in lower for k in ("macbook", "macos", " mac ")):
        profile.platform = "macOS"
    elif "windows" in lower:
        profile.platform = "Windows"

    if any(k in lower for k in ("touch screen", "touchscreen", "2-in-1", "触屏")):
        profile.touch = "Prefer touch"

    use_cases = []
    mapping = (
        (("design", "figma", "adobe", "ps", "视频", "剪辑", "video", "camera", "拍照", "拍摄"), "creative/video"),
        (("gaming", "游戏", "rtx", "gpu"), "gaming"),
        (("study", "学生", "campus", "上课", "作业"), "study"),
        (("office", "办公", "excel", "coding", "开发", "programming"), "productivity"),
    )
    for keys, label in mapping:
        if any(k in lower for k in keys):
            use_cases.append(label)
    if use_cases:
        profile.use_case = use_cases[0]
        _upsert(profile.soft, f"Use case: {profile.use_case}")

    if any(k in lower for k in ("portable", "lightweight", "轻薄", "便携", "campus", "通勤")):
        _upsert(profile.soft, "Portable / lightweight")
    if any(k in lower for k in ("oled", "display", "屏幕", "color", "色彩")):
        _upsert(profile.soft, "Strong display")
    if any(k in lower for k in ("battery", "续航")):
        _upsert(profile.soft, "Long battery life")
    mem = re.search(r"(8|16|32)\s*gb", lower)
    if mem:
        _upsert(profile.hard, f"{mem.group(1)}GB RAM preferred")

    if "phone" in lower or "手机" in lower:
        profile.category = "phone"
    else:
        profile.category = "laptop"

    return profile


def preference_search_query(profile: PreferenceProfile) -> str:
    parts: list[str] = []
    if profile.category:
        parts.append(profile.category if profile.category != "phone" else "phone")
    if profile.use_case:
        parts.append(profile.use_case.split("/")[0])
    if profile.platform in ("Windows", "macOS"):
        parts.append("macbook" if profile.platform == "macOS" else "windows")
    soft_bits = " ".join(profile.soft + profile.hard)
    if profile.visual_context:
        parts.append(profile.visual_context)
    for token in ("oled", "gaming", "touch", "portable"):
        if token in soft_bits.lower():
            parts.append(token)
    if profile.raw_query:
        parts.append(profile.raw_query)
    return " ".join(parts) or "laptop"


def _upsert(values: list[str], label: str) -> None:
    key = label.split(":")[0].split(" up to")[0].lower()
    for i, existing in enumerate(values):
        if existing.lower().startswith(key[:12]):
            values[i] = label
            return
    values.append(label)
