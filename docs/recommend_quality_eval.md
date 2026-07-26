# VoiceShop++ 端到端推荐质量实验报告

本文评估多 Agent Worker（Planner → Recall → Verify → Recommend）在真实商品目录上的**推荐命中率、约束满足率与端到端延迟**，并对比有/无 Planner LLM 与各消融变体。数据来自 `scripts/eval_recommend_quality.py`（$N=30$）。

---

## 1. 实验准备

### 1.1 系统与数据

| 项目 | 说明 |
| --- | --- |
| 系统 | VoiceShop++ Talker–Worker；本实验只评 Worker（不含 ASR/TTS） |
| 目录 | `backend/data/catalog.db`（Amazon Reviews 2023 子集，约 1.5 万条，`laptops` 表） |
| 评测脚本 | `scripts/eval_recommend_quality.py` |
| Case 生成 | `scripts/build_recommend_cases.py` → `backend/data/eval/recommend_cases.jsonl` |
| 检索入口 | `server.search`（关键词 LIKE + 可选语义/富集，按变体开关） |

每条 case 绑定**真实 ASIN** 作为 gold（与 Planner 路由评测中的合成 `prod_a/b/c` 不同）。生成时校验：gold 商品名至少命中一个 `search_keywords`。

### 1.2 测试集构成（$N=30$）

| Bucket | 数量 | 评测重点 |
| --- | --- | --- |
| `text_search` | 12 | 首轮检索 Hit@K |
| `constraint` | 8 | 预算等硬约束 + CSR |
| `refine` | 4 | 多轮改约束；`last_bundle` 为真实商品 |
| `followup` | 3 | 指代/比较（「第二个」等） |
| `safety` | 3 | 应拒绝的危险请求 |

Gold 字段：`relevant_ids`、`primary_id`、`must_satisfy`（如 `budget_le:50`）、`should_refuse`。

质量指标在 **27** 条非 safety case 上计算；Refuse Acc 在 **3** 条 safety case 上计算。

### 1.3 指标定义

| 指标 | 定义 |
| --- | --- |
| Hit@$K$ | top-$K$ 推荐中是否命中任一 `relevant_ids`（$K\in\{1,3,5\}$） |
| MRR | `primary_id` 排名的倒数；未命中为 0 |
| Recall@pool | verify 后候选池是否覆盖 gold |
| CSR | top-$K$（$K=\max$ 截断）中满足 `must_satisfy` 的比例；仅含约束标注的 case |
| Empty rate | 非 refuse 却无推荐的比例 |
| Refuse Acc | safety case 是否正确 `refused` |
| Planner / E2E $P_{50},P_{95}$ | Planner  alone / Planner+Executor 墙钟延迟（ms），不含语音 |

### 1.4 运行环境与命令

- 依赖：本地 `catalog.db`；有 LLM 组需配置 `DASHSCOPE_API_KEY`
- Verifier 统一为 `rule`（降低 LLM 校验方差）
- 结果目录：
  - 无 LLM：`backend/data/eval/rec_no_llm/`
  - 有 LLM：`backend/data/eval/rec_with_llm/`

```text
# 无 Planner LLM（可复现主表）
py -3 scripts/eval_recommend_quality.py --variants full,fixed,no_enrichment,no_verifier,no_modality --no-llm --out-dir backend/data/eval/rec_no_llm

# 有 Planner LLM
py -3 scripts/eval_recommend_quality.py --variants full,fixed,no_enrichment,no_verifier,no_modality --out-dir backend/data/eval/rec_with_llm
```

> 注：评测脚本会 `chdir` 到 `backend/`，若使用相对 `--out-dir`，请写成相对 backend 的路径（如 `data/eval/rec_with_llm`），或使用绝对路径。

---

## 2. 实验设定

### 2.1 两组对照：Planner LLM On / Off

| 组 | `VS_PLANNER_LLM` | 目的 |
| --- | --- | --- |
| **Rules**（`--no-llm`） | 0 | 可复现；隔离召回/校验/短接贡献 |
| **LLM** | 1 | 接近部署；观察意图/hints 对质量与延迟的影响 |

两组均跑同一套 5 个系统变体；case 内已给出结构化 `preference` 与 `search_keywords`，因此 LLM 主要影响意图/短接/hints，而非从零抽槽。

### 2.2 消融变体

| 变体 | 关闭内容 | 对应创新点/模块 |
| --- | --- | --- |
| **Full** | 无（默认全开；verifier=`rule`） | 完整系统 |
| **Fixed** | 模态路由、短接、记忆、replan、视觉召回、Planner LLM | 固定管道基线 |
| **no_enrichment** | `VS_ENRICHMENT`、`VS_TEXT_SEMANTIC`、`VS_REVIEWS` | 离线富集 / 语义召回 |
| **no_verifier** | `VS_VERIFIER=off`，且关 replan | 约束感知校验 |
| **no_modality** | 模态路由 + 视觉召回 | 模态自适应规划 |

其余开关保持与 Full 一致（在 Fixed 中 Planner LLM 恒为关）。

### 2.3 执行协议

对每个 (变体 × case)：

1. 从 JSON 重建 `SessionState`（含可选 `last_bundle`）  
2. 调用 `planner.plan`  
3. 非 refuse 时跑 `Executor`（真实 catalog）；若 `need_replan` 且允许，最多 replan 1 次  
4. 对最终 `bundle.ranked` 计 Hit/MRR/CSR，并对候选池计 Recall@pool  

---

## 3. 实验结果

### 3.1 总表：Rules（无 LLM）

| Variant | Hit@1 | Hit@3 | Hit@5 | MRR | CSR | Empty | Refuse | Planner $P_{50}$ | E2E $P_{50}$ | E2E $P_{95}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.741 | 0.815 | 0.815 | 0.778 | 0.858 | 0.037 | 0.333 | 17 | 3648 | 4405 |
| Fixed | 0.630 | 0.741 | 0.741 | 0.685 | 0.850 | 0.000 | 0.333 | 17 | 3737 | 4536 |
| no_enrichment | 0.815 | 0.815 | 0.815 | 0.815 | 0.958 | 0.037 | 0.333 | 0.3 | 2546 | 7179 |
| no_verifier | 0.556 | **1.000** | **1.000** | 0.765 | **0.583** | 0.000 | 0.333 | 16 | 2497 | 3379 |
| no_modality | 0.741 | 0.815 | 0.815 | 0.778 | 0.858 | 0.037 | 0.333 | 14 | 3458 | 4436 |

### 3.2 总表：LLM（有 Planner LLM）

| Variant | Hit@1 | Hit@3 | Hit@5 | MRR | CSR | Empty | Refuse | Planner $P_{50}$ | E2E $P_{50}$ | E2E $P_{95}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.741 | 0.815 | 0.815 | 0.778 | 0.850 | 0.037 | 0.333 | **4395** | **7856** | **10217** |
| Fixed | 0.630 | 0.741 | 0.741 | 0.685 | 0.850 | 0.000 | 0.333 | 14 | 3617 | 4648 |
| no_enrichment | 0.815 | 0.815 | 0.815 | 0.815 | 0.958 | 0.037 | 0.333 | 3974 | 6975 | 10073 |
| no_verifier | 0.556 | **1.000** | **1.000** | 0.765 | **0.567** | 0.000 | 0.333 | 4245 | 6906 | 8347 |
| no_modality | 0.741 | 0.815 | 0.815 | 0.778 | 0.850 | 0.037 | 0.333 | 4037 | 7774 | 9900 |

### 3.3 Full 分桶 Hit@3（两组几乎一致）

| Bucket | Full Hit@3 | Fixed Hit@3 | 说明 |
| --- | ---: | ---: | --- |
| text_search | 0.75 | 0.75 | 首轮检索接近 |
| constraint | 0.75 | 0.875 | Fixed 略高（见分析） |
| refine | **1.00** | **0.50** | 短接 + 记忆明显有利 |
| followup | **1.00** | **0.67** | 指代/比较依赖记忆短接 |
| safety (Refuse Acc) | 0.33 | 0.33 | 安全拒识仍弱 |

---

## 4. 结果分析

### 4.1 Planner LLM：质量几乎不变，延迟近似翻倍

在当前 case 设计下（偏好与关键词已结构化写入），**Hit@K / MRR / CSR 在 Rules 与 LLM 两组间几乎逐格相同**。差异集中在时延：

- Full Planner $P_{50}$：约 **17 ms → 4.4 s**
- Full E2E $P_{50}$：约 **3.6 s → 7.9 s**
- Fixed 在 LLM 组仍关闭 Planner LLM，故其延迟与 Rules 组接近（E2E $P_{50}$≈3.6 s），说明额外时延主要来自 **Planner 在线 LLM**，而非 Executor。

结论：对「槽位已填好」的离线推荐基准，Planner LLM **不是 Hit@K 的主因**；其价值更可能体现在开放对话抽槽、模糊意图与路由（参见既有 Planner 路由评测）。部署上可对高置信规则路径跳过 LLM，以换取延迟。

### 4.2 Full vs Fixed：多轮场景拉开差距

相对 Fixed，Full 将 Hit@1 从 0.63 提到 0.74，Hit@3 从 0.74 提到 0.82。分桶显示增益几乎全部来自 **refine / followup**：

- Full 在 refine、followup 上 Hit@3 = 1.0  
- Fixed 因关闭 short-circuit 与 memory，被迫重新全量召回，多轮 Hit@3 掉到 0.50 / 0.67  

这与控制面评测中「Fixed 损失 SC/Ref/DAG」一致：**端到端推荐质量上，短接与会话记忆是多轮购物助手的关键模块**。

constraint 桶上 Fixed Hit@3（0.875）略高于 Full（0.75），可能因 Full 的校验/过滤更严导致个别空结果或重排（Full empty rate 非零）。样本量小（每桶 3–12），该差异需谨慎解读。

### 4.3 Verifier：用 Top-1 精度与 CSR 换「宽松召回」

关闭校验器后：

- Hit@3 / Hit@5 → **1.0**，Recall@pool → **1.0**（候选几乎总能盖住 gold）  
- Hit@1 降至 **0.56**，CSR 降至约 **0.57**  

说明无校验时，gold 更容易进 top-3，但排序更噪，且**预算等硬约束大量被破坏**。Verifier（本实验为 rule）是约束满足的主要贡献者，代价是略降宽松 Hit@3、并增加少量空推荐。

### 4.4 Enrichment：本测试集上未体现正面增益

`no_enrichment` 的 Hit@1（0.815）与 CSR（0.958）不低于甚至高于 Full。原因可能包括：

1. Case 偏**强关键词可检索**的商品名，LIKE 已足够；  
2. 当前集**无图搜 case**，富集的视觉侧无从发挥；  
3. 语义召回可能引入近义噪声，拉低 Top-1。

因此：**不能据此否定 enrichment**；应在后续加入视觉/弱关键词 case 再评。Rules 组下关闭 enrichment 后 E2E $P_{50}$ 更低（约 2.5 s），但 $P_{95}$ 升高，尾部不稳定。

### 4.5 Modality routing：本集无差异（符合预期）

`no_modality` 与 Full 质量指标相同——评测集均为纯文本、`has_image=false`，关闭视觉路由不会改变 DAG 实质。该消融需配套图文 case 才有区分度。

### 4.6 安全拒识仍然薄弱

Refuse Acc 在所有变体上均为 **1/3**。日志中部分 safety 话语被判为 `new_search` 而非 `refuse`（尤其中文刀械相关表述）。推荐质量实验暴露出：**安全词表/意图规则与端到端拒识仍需加强**；不能仅依赖路由评测中的模板 safety 句。

### 4.7 综合权衡（论文可用结论）

1. **多轮能力**：Full 相对 Fixed 的主要端到端收益在 refine/followup，验证短接与记忆的产品价值。  
2. **约束能力**：Verifier 显著提升 CSR，是「可信任推荐」必要模块；关闭后 Hit@3 虚高但 Top-1 与约束崩塌。  
3. **Planner LLM**：在结构化离线集上几乎不改变 Hit@K，但使 E2E 中位延迟约 **+4 s**；适合作为不确定轮次的可选增强，而非每轮必开。  
4. **局限**：$N=30$、类目偏杂货目录、缺少图像与开放式抽槽；Refuse Acc 与 enrichment/modality 结论受测试集覆盖限制。

---

## 5. 附录：结果文件

| 组 | 路径 |
| --- | --- |
| Rules | `backend/data/eval/rec_no_llm/recommend_quality_summary.{json,csv}` |
| LLM | `backend/data/eval/rec_with_llm/recommend_quality_summary.{json,csv}` |
| Cases | `backend/data/eval/recommend_cases.jsonl` |
| 逐条明细 | 对应目录下 `recommend_quality_results.jsonl` |
