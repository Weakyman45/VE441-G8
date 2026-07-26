"""实验 / 消融开关集中配置(创新点 1~3 的总开关)。

所有开关都从环境变量读取,默认= 完整系统(Full)。做消融实验时,只需在启动
后端前设置对应环境变量即可,无需改代码、也无需改前端。例如(PowerShell):

    $env:VS_MODALITY_ROUTING="0"      # 消融一:关闭模态自适应规划(退化为固定管道)
    $env:VS_ENRICHMENT="0"            # 消融二:关闭离线富集(仅原始 name/summary)
    $env:VS_SOURCE_LAYERING="0"       # 消融二:关闭属性来源分层(物理属性也让 VL 抽)
    $env:VS_VERIFIER="off"            # 消融三:关闭校验器
    $env:VS_VERIFIER="rule"           # 消融三:规则版校验器
    $env:VS_VERIFIER="llm_strict"     # 消融三:LLM 校验但 unknown 即拒
    $env:VS_VERIFIER="llm"            # 消融三:LLM 校验 + unknown 容错(完整版,默认)

    $env:VS_REVIEWS="0"               # 关闭评论语义召回 / 评论方面信号(Reviewer)
    $env:VS_TEXT_SEMANTIC="0"         # 关闭文本向量语义召回(仅关键词)
    $env:VS_INTENT_SHORTCIRCUIT="0"   # 关闭 refine/followup 短路(总是重召回)
    $env:VS_PLANNER_REPLAN="0"        # 关闭 verify 拒绝后的 replan
    $env:VS_PLANNER_LLM="0"           # Planner 纯规则(无 LLM hints/意图)
    $env:VS_MEMORY="0"                # 关闭记忆预填/引用消解/写回

对应关系(见《创新点与实验设计.md》):
    创新点一  ↔ VS_MODALITY_ROUTING / VS_VISUAL_RECALL / VS_REVIEWS
    创新点二  ↔ VS_ENRICHMENT / VS_SOURCE_LAYERING
    创新点三  ↔ VS_VERIFIER
    Planner   ↔ VS_INTENT_SHORTCIRCUIT / VS_PLANNER_REPLAN / VS_PLANNER_LLM / VS_MEMORY
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in allowed:
        return raw
    return default


@dataclass(frozen=True)
class ExpConfig:
    # --- 创新点一:模态自适应规划 ---
    modality_routing: bool          # 是否按模态动态选择任务 DAG
    visual_recall: bool             # 是否启用视觉召回(图文/纯图片分支)

    # --- 创新点二:离线富集 + 属性来源分层 ---
    enrichment: bool                # 是否使用富集出的 enriched_text / 向量
    source_layering: bool           # 物理属性走 metadata(True)还是也让 VL 抽(False)
    text_semantic: bool             # 文本向量语义召回(与关键词并集)

    # --- 创新点三:约束感知校验 ---
    verifier: str                   # off | rule | llm_strict | llm

    # --- 创新点一(评论侧):Reviewer Agent 评论语义召回 ---
    reviews: bool                   # 是否启用评论语义召回 / 评论方面信号

    # --- Planner / 执行控制 ---
    intent_shortcircuit: bool       # refine/followup/compare 走 rerank_existing
    planner_replan: bool            # verify 拒绝后是否回调 Planner 重规划
    planner_llm: bool               # Planner 是否调用 LLM(意图/hints/relax)
    memory: bool                    # 记忆预填 / 引用消解 / 写回
    max_replans: int                # verify→replan 最大次数

    # --- 通用 ---
    visual_top_k: int               # 视觉召回 top-K
    text_top_k: int                 # 文本语义召回 top-K
    review_top_k: int               # 评论语义召回 top-K
    embedding_provider: str         # dashscope | hash(离线/无网时的确定性兜底)

    @property
    def verifier_enabled(self) -> bool:
        return self.verifier != "off"

    @property
    def verifier_uses_llm(self) -> bool:
        return self.verifier in ("llm", "llm_strict")

    @property
    def verifier_unknown_tolerant(self) -> bool:
        # 只有完整版 "llm" 对 unknown 容错;"llm_strict" 缺失即拒
        return self.verifier == "llm"

    def summary(self) -> dict:
        return {
            "modality_routing": self.modality_routing,
            "visual_recall": self.visual_recall,
            "enrichment": self.enrichment,
            "source_layering": self.source_layering,
            "text_semantic": self.text_semantic,
            "verifier": self.verifier,
            "reviews": self.reviews,
            "intent_shortcircuit": self.intent_shortcircuit,
            "planner_replan": self.planner_replan,
            "planner_llm": self.planner_llm,
            "memory": self.memory,
            "max_replans": self.max_replans,
            "visual_top_k": self.visual_top_k,
            "text_top_k": self.text_top_k,
            "review_top_k": self.review_top_k,
            "embedding_provider": self.embedding_provider,
        }


def load_config() -> ExpConfig:
    """每次调用都重新读环境变量,便于实验脚本在进程内切换配置。"""
    try:
        top_k = int((os.environ.get("VS_VISUAL_TOPK") or "40").strip())
    except ValueError:
        top_k = 40
    try:
        r_top_k = int((os.environ.get("VS_REVIEW_TOPK") or "20").strip())
    except ValueError:
        r_top_k = 20
    try:
        t_top_k = int((os.environ.get("VS_TEXT_TOPK") or "40").strip())
    except ValueError:
        t_top_k = 40
    try:
        max_replans = int((os.environ.get("VS_MAX_REPLANS") or "2").strip())
    except ValueError:
        max_replans = 2
    return ExpConfig(
        modality_routing=_flag("VS_MODALITY_ROUTING", True),
        visual_recall=_flag("VS_VISUAL_RECALL", True),
        enrichment=_flag("VS_ENRICHMENT", True),
        source_layering=_flag("VS_SOURCE_LAYERING", True),
        text_semantic=_flag("VS_TEXT_SEMANTIC", True),
        verifier=_choice("VS_VERIFIER", "llm", ("off", "rule", "llm_strict", "llm")),
        reviews=_flag("VS_REVIEWS", True),
        intent_shortcircuit=_flag("VS_INTENT_SHORTCIRCUIT", True),
        planner_replan=_flag("VS_PLANNER_REPLAN", True),
        planner_llm=_flag("VS_PLANNER_LLM", True),
        memory=_flag("VS_MEMORY", True),
        max_replans=max(0, min(max_replans, 5)),
        visual_top_k=max(5, min(top_k, 200)),
        text_top_k=max(5, min(t_top_k, 200)),
        review_top_k=max(5, min(r_top_k, 200)),
        embedding_provider=_choice(
            "VS_EMBEDDING_PROVIDER", "dashscope", ("dashscope", "hash")
        ),
    )


def _load_dotenv() -> None:
    """启动时自动读取 .env(KEY=VALUE 行)写入 os.environ(不覆盖已存在的变量)。

    这样在 bash / PowerShell / Jupyter 任何环境下都**无需手动 export** 就能连上
    DashScope —— 只要项目里有一个 .env 文件即可。搜索顺序:backend/.env → 项目根 .env。
    .env 应加入 .gitignore,不要提交(内含密钥)。
    """
    here = os.path.dirname(os.path.abspath(__file__))     # backend/engine
    backend = os.path.dirname(here)                       # backend
    for path in (os.path.join(backend, ".env"),
                 os.path.join(os.path.dirname(backend), ".env")):  # 项目根
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:  # utf-8-sig 自动去 BOM
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # 已 export 的优先,不被覆盖
                        os.environ[key] = val
        except OSError:
            pass


# 先加载 .env,再读取配置(顺序很重要:load_config 依赖环境变量)
_load_dotenv()

# 便捷的模块级默认实例(大多数运行时代码直接用它;实验脚本可调用 load_config())
CONFIG = load_config()
