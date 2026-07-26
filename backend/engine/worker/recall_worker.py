"""召回 Agent(Recall Agent):文本检索 + 图片检索,由用户"图文输入"驱动。

把两路召回收拢为一个独立 Agent,职责单一、便于消融与测试。评论(Reviewer)只
负责抽关键词供 Verifier 校验 must-have,**不参与召回**,因此这里不做评论向量召回。

两路召回:
  - 文本检索(text_recall):关键词检索 run_search → catalog.db 的
    name/enriched_text LIKE 命中(enriched_text 已含 enrichment 抽出的视觉词)。
  - 图片检索(image_recall):enrichment.visual_recall → 用户图向量与商品图向量
    余弦 top-K(视觉相似)。

融合(fuse):并集去重(文本在前、视觉补充),并给"所有"候选统一补 `_visual_score`
(即使候选只来自文本召回也能拿到视觉分),供 Recommend 阶段做融合排序。

走哪几路由 route_plan(modality_router 产出)决定;由 ExpConfig 做消融开关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .. import enrichment
from ..models import PreferenceProfile
from .search_worker import run_search

SearchFn = Callable[[dict], list[dict]]


@dataclass
class RecallResult:
    """一次召回的产物:融合后的候选 + 各来源命中数(供日志/前端解释)。"""
    candidates: list[dict[str, Any]]
    text_count: int = 0
    visual_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.candidates),
            "text": self.text_count,
            "visual": self.visual_count,
        }


def _merge_candidates(groups: list[list[dict]]) -> list[dict]:
    """按传入顺序并集去重(按 id),保留首次出现的 dict(及其已有的召回分)。"""
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            pid = str(item.get("id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            merged.append(item)
    return merged


class RecallAgent:
    """统一编排文本 / 图片两路召回并融合。无状态,可长期持有。"""

    def __init__(self, search_fn: SearchFn) -> None:
        self.search_fn = search_fn

    # ---- 文本检索:关键词(含 enriched_text) ----------------------------- #
    def text_recall(self, profile: PreferenceProfile) -> list[dict]:
        return run_search(profile, self.search_fn)

    # ---- 图片检索:视觉相似 ---------------------------------------------- #
    def image_recall(self, query_image: tuple[bytes, str] | None,
                     category: str | None, cfg) -> list[dict]:
        if not query_image or not enrichment.has_enrichment():
            return []
        return enrichment.visual_recall(
            query_image[0], mime=query_image[1],
            top_k=cfg.visual_top_k, category=(category or None),
        )

    # ---- 编排 + 融合 ----------------------------------------------------- #
    def recall(self, *, state, route_plan, cfg,
               query_image: tuple[bytes, str] | None) -> RecallResult:
        profile = state.preference
        text_cands: list[dict] = []
        visual_cands: list[dict] = []

        if route_plan.do_text_recall:
            text_cands = self.text_recall(profile)
        if route_plan.do_visual_recall:
            cat = profile.category if route_plan.infer_category_from_image else None
            visual_cands = self.image_recall(query_image, cat, cfg)

        candidates = _merge_candidates([text_cands, visual_cands])

        # 给所有候选统一补视觉相似度分,供 Recommend 融合排序
        if query_image and enrichment.has_enrichment():
            enrichment.attach_visual_scores(candidates, query_image[0], mime=query_image[1])

        return RecallResult(
            candidates=candidates,
            text_count=len(text_cands),
            visual_count=len(visual_cands),
        )
