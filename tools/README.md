# VoiceShop++ 数据管线(通用商品目录)

从 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) 抽取商品,构建成一个
SQLite 文件 `catalog.db`,供后端 `backend/server.py` 提供搜索/推荐接口
(App 也可把它作为离线兜底)。

> **不再限定笔记本。** `build_laptops.py` 现在会保留你喂给它的 `meta_*.jsonl.gz`
> 里的各类商品,所以「用户输入什么品类就能搜到什么」,取决于你下载了哪些品类文件。
> 笔记本专属字段(display/performance/battery/platform)只有当商品像电脑时才填,
> 其他品类留空,UI 会自动隐藏。

---

## 0.(最省事)一键下载所有品类并构建

在**项目根目录**执行,脚本会遍历全部品类、每类下一小段、下完自动构建
`backend/data/catalog.db`(纯标准库,不需要联网装包):

```powershell
python tools\download_all.py
```

默认走**国内镜像 hf-mirror.com**(比原始 UCSD 美国服务器快很多),4 个并行下载,
每类只取前 80MB——因为每个品类最终只保留 ~500 条商品,几十 MB 足够挑出来了,
**不需要下几百 MB**。

常用参数:

```powershell
python tools\download_all.py --source hf          # 换官方 HuggingFace
python tools\download_all.py --source ucsd        # 换原始 UCSD(海外慢)
python tools\download_all.py --mb 60 --jobs 6     # 每类更小、并行更多
python tools\download_all.py --categories Electronics Clothing_Shoes_and_Jewelry Cell_Phones_and_Accessories
python tools\download_all.py --skip-existing      # 已下过的不重下
python tools\download_all.py --no-build           # 只下载不构建
```

- `--source`:下载镜像,`hfmirror`(默认,国内快)/`hf`/`ucsd`。
- `--mb`:每个品类最多下载多少 MB(默认 80)。**比这个小的品类文件会被完整下满**,属正常。
- `--jobs`:并行下载数(默认 4)。
- `--per-input-limit`:每个品类最多进库多少条(默认 500,保证各品类都有代表)。
- `--limit`:整个 `catalog.db` 的商品总上限(默认 15000)。

某个品类失败会跳过并继续其余品类;可加 `--skip-existing` 重跑补齐。

> 注:HuggingFace 上是未压缩的 `.jsonl`,UCSD 上是 `.jsonl.gz`,脚本和构建器都已自动兼容。

想手动控制,见下面的分步说明。

---

## 1. 下载数据(想覆盖哪些品类就下哪些)

Amazon Reviews 2023 **按品类分文件**,命名规则:

```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz
```

`<Category>` 取值见官网品类列表,常见的有:`Electronics`、
`Cell_Phones_and_Accessories`、`Clothing_Shoes_and_Jewelry`、`Home_and_Kitchen`、
`Sports_and_Outdoors`、`Toys_and_Games`、`Video_Games`、`Office_Products`、
`Beauty_and_Personal_Care`、`Grocery_and_Gourmet_Food`、`Pet_Supplies` 等
(完整列表见 https://amazon-reviews-2023.github.io )。

每个文件都是逐行 JSONL,**只下前一小段(如 ~250MB)就够**。在项目根目录用
Windows 自带 `curl.exe` 分别下载你想要的品类到 `tools\` 下:

```powershell
# 示例:下载电子 + 手机 + 服饰 三个品类的前 250MB
curl.exe -r 0-262144000 -o tools\meta_Electronics_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"
curl.exe -r 0-262144000 -o tools\meta_Cell_Phones_and_Accessories_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz"
curl.exe -r 0-262144000 -o tools\meta_Clothing_Shoes_and_Jewelry_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz"
```

> 想要某品类更全就把 `262144000`(250MB)调大,或去掉 `-r` 参数下整包。

## 2. 构建 catalog.db(离线,不需要联网,也不需要 pip)

在**项目根目录**执行,可一次喂入多个品类文件,`--limit` 是所有文件合计的上限:

```powershell
python tools\build_laptops.py --input tools\meta_Electronics_part.jsonl.gz tools\meta_Cell_Phones_and_Accessories_part.jsonl.gz tools\meta_Clothing_Shoes_and_Jewelry_part.jsonl.gz --limit 3000 --out backend\data\catalog.db
```

脚本能容忍"只下了一半"的截断文件:读到断点自动停止。跑完看到
`Done. Wrote N products` 即成功。若显示 `0 products`,把第 1 步的字节数调大再下一次。

验证:

```powershell
python tools\verify_db.py backend\data\catalog.db
```

## 3.(可选)在线模式:直接按品类流式拉取

需要联网并 `pip install -r tools\requirements.txt`,然后用 `--online-category`
指定品类(可重复):

```powershell
python tools\build_laptops.py --online-category Electronics --online-category Video_Games --limit 3000 --out backend\data\catalog.db
```

## 4.(可选)作为 App 离线兜底

若也想让 App 在连不上后端时用本地库,把生成的库拷成:

```
app\src\main\assets\laptops.db
```

(表名仍为 `laptops`,只是里面装的是各类商品,App 无需改动。)

---

## 字段映射(数据集 → 商品)

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| name | `title` | 截断到 90 字符 |
| price | `price` | 字符串/数字都能解析,取整;缺价存 0 |
| rating / reviewCount | `average_rating` / `rating_number` | 直接取用,无需评论文件 |
| display / performance / battery / platform | `details` / `title` | **仅当商品像电脑时**填,其他品类留空 |
| weightKg | `details`: Item Weight | 磅/盎司/克 → 千克换算 |
| summary / reasons | `features` / `description` | 取前几条短要点 |
| store / image_url | `store` / `images` | 直接取用 |
| weakness / tradeOffs | 无可靠规则来源 | 留空,UI 隐藏或用占位文案兜底 |

> 数据集里 `details` 是 JSON 字符串、`images` 是并列数组的字典、`price` 可能是字符串,
> 脚本都已兼容。想要更高质量的 优缺点/评论情感,可后续改成离线 LLM 生成。
