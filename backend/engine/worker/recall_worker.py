"""召回 Agent(Recall Agent):文本检索 + 图片检索,由用户"图文输入"驱动。

把两路召回收拢为一个独立 Agent,职责单一、便于消融与测试。评论(Reviewer)只
负责抽关键词供 Verifier 校验 must-have,**不参与召回**,因此这里不做评论向量召回。

文本检索(text_recall)两路经 RRF 融合:
  - 关键词:run_search → catalog.db 的 name/enriched_text LIKE;
  - 向量语义:enrichment.text_semantic_recall → 查询句与商品 text_embedding 余弦 top-K
    (由 ExpConfig.text_semantic / VS_TEXT_SEMANTIC 消融)。

图片检索(image_recall):enrichment.visual_recall → 用户图向量与商品图向量余弦 top-K。

融合(fuse):多路排名列表做 Reciprocal Rank Fusion(RRF),写出 `_rrf_score`,
并补 `_visual_score` / `_text_score` 供 Recommend 参考。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .. import enrichment
from ..intent import preference_search_query
from ..models import PreferenceProfile
from ..query_fusion import RRF_K, rrf_fuse
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
    """兼容旧测试:并集去重。新路径请用 ``rrf_merge``。"""
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


def rrf_merge(
    groups: list[list[dict]],
    *,
    weights: list[float] | None = None,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[dict]:
    """多路召回 RRF 融合(空列表自动跳过)。"""
    lists = [g for g in (groups or []) if g]
    if not lists:
        return []
    if len(lists) == 1 and not weights:
        # 单路仍写 _rrf_score,便于下游统一按 RRF 排序
        return rrf_fuse(lists, k=k, limit=limit)
    ws = None
    if weights is not None:
        # 对齐到非空 lists:按原 groups 下标取权重
        ws = []
        for i, g in enumerate(groups or []):
            if not g:
                continue
            ws.append(float(weights[i]) if i < len(weights) else 1.0)
    return rrf_fuse(lists, k=k, weights=ws, limit=limit)


def _text_query(profile: PreferenceProfile, query_hint: str | None = None) -> str:
    keyword_q = " ".join(profile.search_keywords).strip()
    return (
        keyword_q
        or (query_hint or "").strip()
        or preference_search_query(profile)
        or (profile.raw_query or "").strip()
    )


class RecallAgent:
    """统一编排文本 / 图片两路召回并 RRF 融合。无状态,可长期持有。"""

    def __init__(self, search_fn: SearchFn) -> None:
        self.search_fn = search_fn

    # ---- 文本检索:关键词 ⊕ 向量语义 (RRF) ------------------------------ #
    def text_recall(
        self,
        profile: PreferenceProfile,
        cfg=None,
        *,
        query_hint: str | None = None,
    ) -> list[dict]:
        kw = run_search(profile, self.search_fn, query_hint=query_hint)
        for c in kw:
            c.setdefault("_text_score", 1.0)
            c.setdefault("_recall_source", "text_keyword")

        use_sem = True if cfg is None else bool(getattr(cfg, "text_semantic", True))
        if not use_sem or not enrichment.has_text_embeddings():
            return rrf_merge([kw]) if kw else []

        q = _text_query(profile, query_hint)
        if not q:
            return rrf_merge([kw]) if kw else []
        try:
            sem = enrichment.text_semantic_recall(
                q,
                top_k=int(getattr(cfg, "text_top_k", 40) or 40) if cfg is not None else None,
                category=(profile.category or None),
                provider=getattr(cfg, "embedding_provider", None) if cfg is not None else None,
            )
        except Exception:
            sem = []
        return rrf_merge([kw, sem])

    # ---- 图片检索:视觉相似 ---------------------------------------------- #
    def image_recall(self, query_image: tuple[bytes, str] | None,
                     category: str | None, cfg) -> list[dict]:
        if not query_image or not enrichment.has_enrichment():
            return []
        return enrichment.visual_recall(
            query_image[0], mime=query_image[1],
            top_k=cfg.visual_top_k, category=(category or None),
        )

    # ---- 编排 + RRF 融合 ------------------------------------------------- #
    def recall(self, *, state, route_plan, cfg,
               query_image: tuple[bytes, str] | None) -> RecallResult:
        profile = state.preference
        text_cands: list[dict] = []
        visual_cands: list[dict] = []

        if route_plan.do_text_recall:
            text_cands = self.text_recall(profile, cfg)
        if route_plan.do_visual_recall:
            cat = profile.category if route_plan.infer_category_from_image else None
            visual_cands = self.image_recall(query_image, cat, cfg)
            for c in visual_cands:
                c.setdefault("_recall_source", "visual")

        # 文本路与视觉路再做一次 RRF(权重可走 modality 默认;路有结果则权重至少 1)
        from ..query_fusion import default_fuse_weights

        modality = getattr(route_plan, "modality", None) or "text"
        fw = default_fuse_weights(str(modality))
        tw = float(fw.get("text", 1.0) or 0.0)
        vw = float(fw.get("visual", 1.0) or 0.0)
        if text_cands and tw <= 0:
            tw = 1.0
        if visual_cands and vw <= 0:
            vw = 1.0
        candidates = rrf_merge(
            [text_cands, visual_cands],
            weights=[tw, vw],
        )

        # 给所有候选统一补视觉 / 文本相似度分
        if query_image and enrichment.has_enrichment():
            enrichment.attach_visual_scores(candidates, query_image[0], mime=query_image[1])
        q = _text_query(profile)
        if q and enrichment.has_text_embeddings():
            try:
                enrichment.attach_text_scores(
                    candidates,
                    q,
                    provider=getattr(cfg, "embedding_provider", None),
                )
            except Exception:
                pass

        return RecallResult(
            candidates=candidates,
            text_count=len(text_cands),
            visual_count=len(visual_cands),
        )
