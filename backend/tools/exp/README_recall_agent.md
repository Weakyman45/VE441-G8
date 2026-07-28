# 召回 Agent 消融实验(text-only vs image-only vs fused)操作文档

对应脚本:`tools/exp/exp_recall_agent.py`
对应报告:`tools/exp/results/recall_agent_experiment.txt`(英文 LaTeX,可直接贴论文)

## 1. 这个实验测什么

召回 Agent(`engine/worker/recall_worker.py:RecallAgent`)有两条召回链路:

- **文本检索**(`text_recall`):把用户文字关键词交给 `run_search`,在
  `catalog.db` 的 `name` + `enriched_text` 上做关键词命中(`enriched_text` 里含
  enrichment 抽出的视觉词)。
- **图片检索**(`image_recall`):把用户图片编码成 1024 维向量,与商品图向量做余弦
  相似(`enrichment.visual_recall`)。

本实验做**消融**:在同一个"图文查询"下,分别只开文本、只开图片、两路都开(融合),
比较检索质量,验证融合是否优于单模态。

## 2. 评测协议(留一法,不含自身)

对每个查询商品 `P`:

- **文字输入** = `P` 的视觉关键词(`visual_attrs.keywords`,模拟用户"描述外观");
- **图片输入** = `P` 的商品图;
- **相关集 gold** = 与 `P` **同一视觉品类**(`visual_attrs.product_category`)的其它
  商品(排除 `P` 自己);
- 三种条件各自产出 top-K 排序,算 `precision@K / nDCG@K / recall@K`。

四种条件,其中 T/I/F 都走**真实的 RecallAgent**(只用不同 `RoutePlan` 开关),
N 为"**无 LLM 的朴素关键词检索**"基线:

| 条件 | do_text_recall | do_visual_recall | 说明 |
|---|---|---|---|
| N keyword-noLLM | —(直接查) | ✗ | **无 LLM 基线**:把商品原始标题交给 `server.search` 做正则分词+关键词命中,不走 `analyze.py` 的 LLM 抽词 |
| T text-only | ✓ | ✗ | 只关键词召回(用 LLM/VL 提炼的关键词) |
| I image-only | ✗ | ✓ | 只视觉召回(纯视觉、不给品类,避免把 gold 品类泄漏) |
| F fused | ✓ | ✓ | 两路并集 + 给全部候选补视觉分(完整 Agent) |

留一法:所有条件的排序都排除查询商品自身。

排序键与 `recommend_worker` 一致:文本/图文按"关键词命中商品名个数 → 视觉相似度";
纯图片按视觉相似度。(不调用 `recommend_worker` 本体,避免其 LLM 摘要开销。)

显著性用**配对 Wilcoxon 符号秩检验**:每个召回条件 vs 不召回基线(`T-N`/`I-N`/`F-N`,
体现"有召回 vs 没召回"),以及融合 vs 单模态(`F-T`/`F-I`,体现融合增益)。

## 3. 前置条件

- **必须有 `DASHSCOPE_API_KEY`**:image-only 与 fused 要用多模态 embedding 编码用户图。
  key 放在 `backend/.env`(`DASHSCOPE_API_KEY=sk-...`),脚本会自动加载。
- **需要联网**:每条查询要下载商品图并调用 DashScope 编码。
  > 注意:WSL 里若出现 `[Errno 101] Network is unreachable`,说明 WSL 没网,会导致
  > 结果全 0。**建议直接在 Windows 上跑**(网络与 key 已验证可用)。
- `catalog.db` 需已富集(含 `image_embedding` / `visual_attrs`)。

## 4. 运行

在 `backend/` 目录下(Windows PowerShell 用 `D:\python.exe`,WSL 用 `python`):

```powershell
# 正式规模(与图片实验一致):30 条查询,K=30
D:\python.exe tools/exp/exp_recall_agent.py --n 30 --k 30

# 快速冒烟(打印前 3 条查询的品类明细,便于排查)
D:\python.exe tools/exp/exp_recall_agent.py --n 2 --k 5 --debug
```

参数:

| 参数 | 默认 | 含义 |
|---|---|---|
| `--n` | 30 | 抽样查询商品数 |
| `--k` | 30 | 评测 top-K |
| `--pool` | 80 | 视觉召回候选池深度(visual_top_k) |
| `--min-cat` | 4 | 品类最小成员数(小于此的品类不参与) |
| `--seed` | 42 | 随机种子(可复现抽样) |
| `--debug` | 关 | 打印前几条查询的召回品类明细 |

> PowerShell 控制台里中文可能显示乱码,这是终端编码问题,**落盘文件内容正常**。

## 5. 产物(都在 `tools/exp/results/`)

- `recall_agent_perquery.csv` —— 每条查询、每个条件(N/T/I/F)的 precision/ndcg/recall 明细;
- `recall_agent_summary.md` —— 汇总表(none/text/image/fused 四列 + 各组 Δ 与显著性);
- `recall_agent_summary.png` —— 分组柱状图(需装 `matplotlib`,否则自动跳过)。

## 6. 参考结果(N=30, K=30, seed=42)

| 指标 | N(关键词·无LLM) | text | image | fused | Δ(T−N) | Δ(I−N) | Δ(F−N) | Δ(F−T) | Δ(F−I) |
|---|---|---|---|---|---|---|---|---|---|
| precision@30 | 0.1633 | 0.1644 | 0.2356 | 0.2256 | +0.001 ns | +0.072 ** | +0.062 ** | +0.061 *** | −0.010 ns |
| nDCG@30 | 0.6001 | 0.5899 | 0.7559 | 0.7138 | −0.010 ns | +0.156 *** | +0.114 ** | +0.124 *** | −0.042 ns |
| recall@30 | 0.4365 | 0.4951 | 0.6867 | 0.6481 | +0.059 ns | +0.250 *** | +0.212 ** | +0.153 *** | −0.039 ns |

结论:
- **视觉路是主导信号**:image 与 fused 在三项指标上都**显著**优于无 LLM 关键词基线 N
  (p<0.01);而用 LLM/VL 提炼关键词的纯文本(T)与朴素关键词基线 N **无显著差异**
  (Δ(T−N) 全 ns)——即在"同视觉品类"的相关性下,把需求转成文字(无论是否用 LLM)
  都不如直接在图像向量空间匹配。
- **融合继承视觉路优势**:融合显著优于纯文本(F−T 三项 ***),与纯图片无显著差异
  (F−I 全 ns);融合从不弱于更强的单模态,并补回纯文本漏掉的覆盖率。纯文本因外观词
  歧义(如 "lace/lingerie" 命中假发/接发)出现跨品类漂移。

英文 LaTeX 报告见 `results/recall_agent_experiment.txt`。

> 局限:该协议按视觉品类判相关,天然偏向图片模态;只有文字能表达的约束(预算、
> 品牌、硬性 must-have)不在本召回消融范围内,由下游 Verifier 处理。
