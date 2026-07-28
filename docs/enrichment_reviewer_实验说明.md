# Enrichment & Reviewer 实验:运行说明与论文呈现

本套件用于证明两个创新点的有效性,分开做:

- **Reviewer**(评论方面用于校验):实验 R
- **Enrichment**(离线多模态富集):实验 E-image / E-text / E-hallucination

所有脚本在 `backend/tools/exp/` 下,结果统一落盘到 `backend/tools/exp/results/`。
指标与表格是**纯标准库**产出(CSV + Markdown,一定能跑);图是 matplotlib 画的,
未安装则自动跳过画图、不影响出数据。

```bash
# (可选)只为画图;不装也能出全部 CSV/Markdown 数据
pip install matplotlib
```

> 代码里已把两个消融开关接进运行时:
> - `VS_ENRICHMENT=0/1` 控制文本检索是否使用 `enriched_text`(`server.py`);
> - `VS_REVIEWS=0/1` 控制校验器是否使用 `review_aspects`(`verifier.py`,规则路径与 LLM 判官都接了)。
>
> 因此**无需重新预处理数据库**:富集/评论已烤进 `catalog.db`,实验只在"查询时"翻开关模拟"有/无该组件"。

---

## 0. 准备

```bash
cd backend
export DASHSCOPE_API_KEY=sk-xxxx        # E-image / E-hallucination 需要;R / E-text 不需要
python tools/exp/build_gold.py          # 从库里自动挖金标准,先看一眼规模(可选)
```

`build_gold.py` 会打印并保存三份金标准到 `results/gold_*.json`:
评论软属性(durable/comfortable/soft…)、视觉属性(leather/ceramic/velvet…)、
视觉品类分组(供图片实验的"同类=相关")。**金标准全部从已落库的
`review_aspects`/`visual_attrs` 自动挖掘,无需人工标注。**

---

## 实验 R:Reviewer —— 评论方面对"软需求校验"的贡献

**证明**:很多体验类软需求(耐穿 durable、舒适 comfortable、柔软 soft、透气
breathable…)只写在**用户评论**里,商品标题和富集文本都没有。开启 Reviewer 后,
校验器能据评论满足这些 must-have;关闭后就漏掉。

- **输入(input)**:每个属性一条查询 = `{must-have=[attr], category=""}`;
  候选池 = gold(评论提到 attr、标题/富集没有的商品)+ distractor(都没提到的同池商品)。
- **被测系统**:`verifier.verify_candidates`,只差一个开关 `reviews=ON/OFF`。
- **输出(output)**:每属性的 `must-have 满足率(recall of gold)` ON vs OFF;
  distractor 保留率(证明不是无脑放行);跨 gold 商品的配对 Wilcoxon 显著性;一个定性例子。

```bash
python tools/exp/exp_reviewer.py                       # 默认 rule 校验(确定、免 API)
python tools/exp/exp_reviewer.py --verifier llm        # 用 LLM 判官(需 API,评论喂给判官)
python tools/exp/exp_reviewer.py --attrs durable comfortable soft breathable
```

产物:`results/reviewer_recall.{csv,md,png}`、`reviewer_example.json`。

**预期**:`recall_off ≈ 0`,`recall_on ≈ 1.0`,Δ≈0.9,p<0.001;distractor 保留率两边都很低。

---

## 实验 E-text:Enrichment —— 纯文本输入,富集视觉属性提升检索

**证明**(对应你的需求 (2)):用户用纯文字描述一个视觉属性(材质/款式/图案,如
leather / ceramic / velvet / striped),很多相关商品**标题里没有这个词**,只有离线
富集(Qwen-VL 抽的颜色/款式/材质写进 `enriched_text`)才带 → 开启富集检索召回大增。

- **输入(input)**:文本查询 = 一个视觉属性词。
- **被测系统**:真实 `server.search()`,切换 `VS_ENRICHMENT=1/0`(ON=name+enriched_text,OFF=仅 name)。
- **输出(output)**:每属性 `recall@K` / `nDCG@K`(ON vs OFF)、`enrich_only_hits`
  (只有富集才召回的相关商品数)、配对 Wilcoxon。

```bash
python tools/exp/exp_enrich_text.py
python tools/exp/exp_enrich_text.py --k 20 --attrs leather ceramic velvet wooden striped metallic fabric
```

产物:`results/enrich_text_recall.{csv,md,png}`。

**预期**:`recall_off ≈ 0`,`recall_on` 明显更高,`enrich_only_hits > 0`,p<0.001。
> 注:属性 gold 很大时(如 plastic/metal 有几百个)`recall@20` 会被 K 限制而偏低,
> 这时看 **nDCG@20** 与 **enrich_only_hits** 更能说明问题(小/中 gold 属性如
> ceramic/velvet/leather 的 recall 提升最直观)。

---

## 实验 E-image:Enrichment —— 有图片输入,图→图 vs 图→文

**证明**(对应你的需求 (1)):有图片输入时,两种用法哪个更准——
- **A 图→图**:输入图编码成向量,与商品图向量做余弦(视觉召回);
- **B 图→文**:先用 VL 把输入图抽成关键词,再用关键词检索**商品名字**。

- **输入(input)**:把某商品 P 的图片当作"用户上传图"(留一法)。
- **相关集 gold**:与 P **同一视觉品类**的其它商品(排除 P 自己)。
- **输出(output)**:A、B 各自的 `precision@K / nDCG@K / recall@K` 均值,以及
  A vs B 的差值与配对 Wilcoxon;每条查询明细 CSV。

```bash
export DASHSCOPE_API_KEY=sk-xxxx
python tools/exp/exp_enrich_image.py --n 30 --k 10
```

产物:`results/enrich_image_summary.{md,png}`、`enrich_image_perquery.csv`。

**预期**:A(图→图)在 precision/nDCG 上显著高于 B(图→文)——因为图向量直接对齐
视觉相似,而"图→关键词→商品名"会丢失大量视觉信息、且受标题措辞影响。这正好论证
"离线图像 embedding + 视觉召回"相对"图转文字再检索"的优势。

---

## 实验 E-hallucination:Enrichment —— 属性来源分层抑制物理属性幻觉

**证明**(对应你的需求 (3)):让 VL 去报**非视觉的物理数值**(重量/尺寸/容量/成分/
件数/续航)会大量脑补;来源分层(物理走 metadata、VL 只抽可见属性)从根上消除这类幻觉。

- **输入(input)**:抽样 N 个商品的图片。
- **两种模式**:L 分层(当前系统,VL 只抽可见属性)vs N 不分层(额外让 VL 报物理数值)。
- **金标准**:把 VL 的每条物理断言与商品**自身文字资料**比对(数字是否出现 / 关键词是否出现),
  不可佐证 = 幻觉。**可复现、无需人工。**
- **输出(output)**:两种模式的"每商品物理断言数"与 `hallucination_rate`;不可佐证断言的例子。

```bash
export DASHSCOPE_API_KEY=sk-xxxx
python tools/exp/exp_hallucination.py --n 40
```

产物:`results/hallucination_summary.{md,png}`、`hallucination_examples.json`。

**预期**:分层版 VL 物理断言≈0(无从幻觉);不分层版产生大量物理断言,其中高比例
不可被 metadata 佐证 → 幻觉率高。用 `hallucination_examples.json` 里的例子(如 VL 说
"250 ml"但资料是"8 fl oz")做定性佐证最有力。

---

## 如何在论文里画表格 / 图

每个实验都同时给了 **Markdown 表格(可直接粘)**、**CSV(可再加工)**、**PNG 图**。

### 表格(直接用 `results/*.md`)
- 实验 R:一张"属性 × {recall_off, recall_on, Δ}"表 + Wilcoxon p 值一行。
- 实验 E-text:一张"视觉属性 × {recall@K off/on, nDCG@K, enrich_only_hits}"表。
- 实验 E-image:一张"指标 × {A 图→图, B 图→文, Δ, p}"小表(3 行:precision/nDCG/recall)。
- 实验 E-hallucination:一张"模式 × {物理断言数, 不可佐证数, 幻觉率}"两行表。

LaTeX 里用 `booktabs` 的 `\toprule/\midrule/\bottomrule`;把 Markdown 表照抄成
`tabular` 即可。给每张表配一句"结论句 + 显著性星号(\* p<0.05, \*\* p<0.01, \*\*\* p<0.001)"。

### 图(用 `results/*.png`,或用 CSV 自己重画)
- R / E-text:**分组柱状图**(每个属性两根柱:OFF vs ON)。
- E-image:**分组柱状图**(precision/nDCG/recall 三组,每组 A vs B 两根柱)。
- E-hallucination:**分组柱状图**(断言数、幻觉率两组,Layered vs Non-layered)。

> 图里的文字标签用的是英文(避免服务器/WSL 缺中文字体导致方块)。如果你要中文标签,
> 在 WSL 里装一个中文字体并在 `common.py` 里设 `matplotlib` 的 `font.sans-serif`
> (如 `Noto Sans CJK SC`),或直接用导出的 CSV 在 Excel/Origin 里重画。

### 定性例子(审稿人最买账)
- R:`reviewer_example.json` —— 某商品评论关键词含 "durable",reviews ON 保留 / OFF 拒绝。
- E-hallucination:`hallucination_examples.json` —— VL 越界报的物理数值 vs 商品真实资料。
- E-text:表里 `enrich_only_hits` 就是"仅靠富集才召回"的相关商品计数,可在正文举 1~2 个具体商品。

---

## 备注

- **R 和 E-text 完全离线**(只读 DB + 跑校验/SQL),不需要 API,可反复快速跑。
- **E-image 和 E-hallucination 需要 `DASHSCOPE_API_KEY`**(调用 embedding / Qwen-VL),
  会有网络耗时与少量费用;用 `--n` 控制样本量(建议先 `--n 10` 跑通再放大)。
- 所有随机抽样都有 `--seed`,结果可复现。
- 消融开关一览见 `backend/engine/exp_config.py` 顶部注释。
