"""创新点一:模态自适应的多智能体规划(Modality-Adaptive Planning)。

把"输入模态"提升为一等规划变量:根据本次输入是 纯文本 / 图文 / 纯图片,
产出不同的任务图(走哪几条召回、是否需要先从图推品类、是否校验),供
WorkerRuntime 按需调度。对应《创新点与实验设计.md》§1。

设计要点:
  - 纯文本 T   : 文本召回 → 校验 → 排序
  - 图文  T+I  : (文本召回 ∥ 视觉召回) → 合并 → 校验 → 排序
  - 纯图片 I   : VL 推品类 → 同类内视觉召回 → (反向)校验 → 排序

消融:当 VS_MODALITY_ROUTING=0 时退化为"固定管道"(所有模态都只走文本召回),
用于对照实验;当 VS_VISUAL_RECALL=0 时关闭视觉召回。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .exp_config import ExpConfig, CONFIG

# 模态常量
MODALITY_TEXT = "text"
MODALITY_TEXT_IMAGE = "text_image"
MODALITY_IMAGE = "image"

# 占位文本(server 在图片上传时写入的对话占位),不算作"有效文本需求"
_PLACEHOLDER_TEXTS = {"[image uploaded]", "[图片]", ""}


@dataclass
class RoutePlan:
    """一次检索的模态化执行计划。"""
    modality: str
    do_text_recall: bool
    do_visual_recall: bool
    infer_category_from_image: bool   # 纯图片时先用 VL 从图推品类
    reverse_verify: bool              # 纯图片时对候选做反向(图→属性)验证
    do_verify: bool                   # 是否走约束校验(创新点三)
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_meaningful_text(preference) -> bool:
    """判断本次偏好里是否含"真实的文本需求"(而非仅图片占位)。"""
    raw = (getattr(preference, "raw_query", "") or "").strip().lower()
    if raw and raw not in _PLACEHOLDER_TEXTS:
        return True
    # 关键词 / 硬软约束 / 品类 / 用途 任一非空,也算有文本意图
    if getattr(preference, "search_keywords", None):
        return True
    if getattr(preference, "hard", None) or getattr(preference, "soft", None):
        return True
    if (getattr(preference, "category", "") or "").strip():
        return True
    if (getattr(preference, "use_case", "") or "").strip():
        return True
    return False


def _valid_image_refs(session) -> list[dict]:
    refs = getattr(session, "image_refs", None) or []
    out = []
    for ref in refs:
        if isinstance(ref, dict) and (ref.get("path") or ref.get("data_url")):
            out.append(ref)
    return out


def detect_modality(session) -> str:
    """根据 session 的偏好文本与上传图片判定输入模态。"""
    has_text = _has_meaningful_text(getattr(session, "preference", None))
    has_image = len(_valid_image_refs(session)) > 0
    if has_image and has_text:
        return MODALITY_TEXT_IMAGE
    if has_image and not has_text:
        return MODALITY_IMAGE
    return MODALITY_TEXT


def route(session, config: ExpConfig | None = None) -> RoutePlan:
    """产出本次检索的模态化执行计划。"""
    cfg = config or CONFIG
    modality = detect_modality(session)

    # 消融:关闭模态路由 → 固定管道(所有模态都只做文本召回)
    if not cfg.modality_routing:
        return RoutePlan(
            modality=modality,
            do_text_recall=True,
            do_visual_recall=False,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=cfg.verifier_enabled,
            reason="fixed-pipeline (modality routing disabled)",
        )

    visual_ok = cfg.visual_recall and cfg.enrichment  # 视觉召回依赖离线富集的向量

    if modality == MODALITY_TEXT:
        return RoutePlan(
            modality=modality,
            do_text_recall=True,
            do_visual_recall=False,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=cfg.verifier_enabled,
            reason="text-only: text recall -> verify -> rank",
        )

    if modality == MODALITY_TEXT_IMAGE:
        return RoutePlan(
            modality=modality,
            do_text_recall=True,
            do_visual_recall=visual_ok,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=cfg.verifier_enabled,
            reason="text+image: (text ∥ visual) recall -> merge -> verify -> rank",
        )

    # MODALITY_IMAGE(纯图片): A_I 先抽品类/关键词,再走文本+视觉召回
    return RoutePlan(
        modality=modality,
        do_text_recall=True,
        do_visual_recall=visual_ok,
        infer_category_from_image=True,
        reverse_verify=True,
        do_verify=cfg.verifier_enabled,
        reason="image-only: A_I -> (text ∥ visual) recall -> reverse verify -> rank",
    )
