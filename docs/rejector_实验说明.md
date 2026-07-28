# Rejector / Verifier 实验:运行说明与论文呈现

本实验对应**创新点三**:在召回与排序之间插入"约束感知校验闸门"(Reject Agent),
证明它能显著提升**最终输出商品与用户硬约束(品类 + must-have)的匹配度**。

脚本:`backend/tools/exp/exp_rejector.py`;结果落盘到 `backend/tools/exp/results/`。
指标/表格是纯标准库(CSV + Markdown 一定能跑),图用 matplotlib(未装则跳过画图)。

> 消融开关已接进运行时:`VS_VERIFIER = off | rule | llm_strict | llm`
> (见 `engine/exp_config.py`)。本实验直接在进程内切换该配置,无需改代码。

---

## 0. 核心思路

- **有 rejector**:候选经过校验,品类不符 / 缺 must-have 的商品被剔除 → 输出高度匹配用户要求。
- **无 rejector**(`off`):候选原样进入结果 → 匹配度 = 候选里正样本的基础占比(低)。

对比二者,即可量化 rejector 的价值。

---

## 1. 实验流程(它内部怎么跑)

对每条测试查询 = **品类 c + must-have 属性 a**(如 "skincare" + "gentle"):

1. **构造候选池**(跳过召回,直接从全库取,以隔离 Verifier):
   - 正样本 `positives`:品类为 c **且**结构化标注具备属性 a 的商品(真正满足要求);
   - 负样本(错品类):品类不是 c 的商品(违反品类约束);
   - 负样本(缺属性):品类为 c 但没有属性 a 的商品(违反 must-have)。
   - 三者均衡采样后混合成候选池。
2. **金标准 gold = positives**。品类用 `visual_attrs.product_category`(结构化标签),
   属性用 `review_aspects` / `visual_attrs` 的结构化列表判定 —— 尽量独立于校验器的"文本命中"机制。
3. **同一候选池**,分别用 `VS_VERIFIER = off / rule /(可选 llm)` 跑校验,得到各自的"输出(kept)"。
4. 对每个输出计算 precision / recall / F1 / 输出规模,对所有查询取平均。

查询是自动构造的:遍历成员数够多的品类,取该品类里出现最频繁的若干属性组成 (c, a)。

---

## 2. 输入 / 输出

- **输入(input)**:用户要求 = `{category: c, must-have: [a]}` + 一个混合候选池。
- **输出(output)**:每种模式下,**输出商品列表**的匹配度指标。

### 指标含义

| 指标 | 定义 | 意义 |
|---|---|---|
| **match_rate (precision@kept)** | 输出里真正满足要求的比例 = \|kept ∩ gold\| / \|kept\| | **核心"匹配度"**:用户看到的商品有多少确实符合要求 |
| **coverage (recall)** | 正样本被保留的比例 = \|kept ∩ gold\| / \|gold\| | 校验器有没有**误杀**正样本 |
| **F1** | precision 与 recall 的调和均值 | 综合匹配度与不误杀 |
| **avg_output_size** | 平均输出商品数 | off 会很大(全放行),on 会收敛 |

---

## 3. 怎么跑(WSL)

```bash
cd backend

# 主实验:off vs rule(确定、免 API、可复现)—— 论文主结果用这个
python tools/exp/exp_rejector.py

# 额外加 LLM 判官一组(需 DASHSCOPE_API_KEY),验证结论在 LLM 校验下也成立
export DASHSCOPE_API_KEY=sk-xxxx
python tools/exp/exp_rejector.py --llm

# 可调参数
python tools/exp/exp_rejector.py --max-queries 40 --min-pos 5 --attrs-per-cat 3 --seed 42

# (可选)画图
pip install matplotlib
```

产物:
- `results/rejector_summary.md` / `.csv`:每种模式一行的汇总(可直接粘论文)。
- `results/rejector_perquery.csv`:每条查询的明细(off/rule/llm 的 match/recall/size)。
- `results/rejector_summary.png`:match_rate / coverage / F1 的分组柱状图(off vs rule[/llm])。

---

## 4. 结果怎么读(示例)

一次典型运行(N=40 条查询,off vs rule):

| mode | match_rate(precision) | coverage(recall) | F1 | avg_output_size |
| --- | --- | --- | --- | --- |
| off  | 0.485 | 1.000 | 0.652 | 28.6 |
| rule | 0.974 | 0.964 | 0.961 | 14.2 |

配对 Wilcoxon(rule vs off,match_rate):p ≈ 3.6e-8(\*\*\*)。

**结论**:
- 关闭 rejector,输出匹配度只有 ~48%(候选里近一半违反品类/must-have);
- 开启 rejector,匹配度升到 ~97%,同时保留 96% 的正样本(几乎不误杀),输出规模收敛一半;
- 即 rejector **在几乎不牺牲召回的前提下,把输出与用户要求的匹配度翻倍**,差异极显著。

> 三档对比建议:主表用 `off` vs `rule`;若跑了 `--llm`,再补一行 `llm`,说明"换成
> LLM 语义判官结论依然成立"(LLM 版对语义/隐性约束更强,但有 API 成本与随机性)。

---

## 5. 论文里怎么呈现

### 表格(直接用 `rejector_summary.md`)
一张"模式 × {match_rate, coverage, F1, avg_output_size}"表,配 Wilcoxon p 值一行。
LaTeX 用 booktabs;给每行/每列标注显著性(\* p<0.05, \*\* p<0.01, \*\*\* p<0.001)。

### 图(用 `rejector_summary.png`)
分组柱状图:三组(match_rate / coverage / F1),每组 off vs rule(可加 llm)两三根柱。
一眼可见 match_rate 大幅抬升、coverage 基本不掉。

### 定性佐证(可选)
从 `rejector_perquery.csv` 挑 1~2 条查询举例:同一候选池,off 输出里混入错品类/缺属性
的商品,rule/llm 把它们剔除、只留真正匹配的 —— 用具体商品名做对照最有说服力。

---

## 6. 备注

- `off` vs `rule` **完全离线**、确定、可复现,适合作论文主结果;`llm` 需 API、判定带随机性。
- 金标准用结构化字段(product_category / review_aspects / visual_attrs)判定,
  与校验器的"文本命中"机制并不完全相同,规则版仍存在部分相关性;LLM 版独立性更强。
  论文中如实说明即可(见脚本 docstring)。
- 随机采样有 `--seed`,结果可复现。
- 其它硬约束(预算 budget、平台 platform)也可加入:在 `PreferenceProfile` 里设 `budget`
  并往候选池混入超预算商品即可,规则粗筛会剔除它们(本脚本默认聚焦品类 + must-have 两条)。
```

