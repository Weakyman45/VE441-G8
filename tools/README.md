# VoiceShop++ 数据管线(Route A:端上 SQLite)

把 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) 里的笔记本子集
抽取成一个小的 `laptops.db`,供 Android App 直接从 `assets` 读取。

App 端已写好:`assets/laptops.db` 存在就用它,不存在就回退内置示例。所以这里
只负责**产出这一个文件**。

---

## 情况一:运行 Python 的机器没有网络(你的虚拟机就是这种)

分三步:**在有网的 Windows 上下载 → 传进虚拟机 → 在虚拟机里解析。**

### 1. 在 Windows 上下载数据(只下前一小段就够)

2023 版 Electronics 元数据整包约 1.96 GB,但它是逐行 JSONL,我们只需要前面
一小段就能凑够 500 台笔记本。在 Windows 的 PowerShell 里用自带的 `curl.exe`
只下前 ~250 MB:

```powershell
curl.exe -r 0-262144000 -o meta_Electronics_part.jsonl.gz "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"
```

> 如果 `datarepo.eng.ucsd.edu` 打不开,换镜像:
> `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz`
>
> 想要更全就把 `262144000`(250 MB)调大,或去掉 `-r` 参数下整包。

### 2. 把这个文件传进虚拟机

放到虚拟机里项目的 tools 目录,例如:
`~/VE441-G8-main/tools/meta_Electronics_part.jsonl.gz`
(用你之前把项目弄进虚拟机的同样方式:共享文件夹 / 拖拽 / scp 都行。)

### 3. 在虚拟机里解析(不需要联网,也不需要装 datasets)

```bash
cd ~/VE441-G8-main/tools
python build_laptops.py --input meta_Electronics_part.jsonl.gz --limit 500 --out laptops.db
```

脚本能容忍"只下了一半"的截断文件:读到断点就自动停止。跑完看到
`Done. Wrote N laptops` 即可。若显示 `0 laptops`,说明下的那段里笔记本太少,
把第 1 步的字节数调大再下一次。

### 4. 把 laptops.db 放进 App

把生成的 `laptops.db` 拷回 Windows:

```
C:\Users\f0407\Desktop\VE441-G8-main\app\src\main\assets\laptops.db
```

然后在 Android Studio 里 Sync + Run。

---

## 情况二:运行 Python 的机器有网络

```bash
pip install -r requirements.txt
python build_laptops.py --limit 500 --out laptops.db
```

(在线模式用 HuggingFace 流式读取,注意新版 `datasets` 已不支持 `trust_remote_code`,
本脚本已去掉该参数。)

---

## 字段映射(数据集 → App)

| App 字段 | 来源 | 说明 |
| --- | --- | --- |
| name | `title` | 截断到 90 字符 |
| price | `price` | 字符串/数字都能解析,取整 |
| rating / reviewCount | `average_rating` / `rating_number` | 直接取用,无需评论文件 |
| display | `details`: Screen Size / Resolution / Display Type | 规则拼接 |
| performance | `details`: Processor / RAM / Hard Drive / Graphics | 规则拼接 |
| battery | `details`: Battery Life 或 features 含 battery+hour 的句子 | 规则抽取 |
| weightKg | `details`: Item Weight | 磅/盎司/克 → 千克换算 |
| summary / reasons | `features` | 取前几条短要点 |
| platform | `title` / OS | mac→macOS,chrome→ChromeOS,否则 Windows |
| weakness / tradeOffs / reviewSentiment | 无可靠规则来源 | 留空,App 用占位文案兜底 |

> 数据集里 `details` 是 JSON 字符串、`images` 是并列数组的字典、`price` 可能是字符串,
> 脚本都已兼容。想要更高质量的 优缺点/评论情感,可后续改成离线 LLM 生成。
