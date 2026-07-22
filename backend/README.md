# VoiceShop++ 后端(做法 B:后端持有数据 + App 联网查询)

一个**零第三方依赖**的只读商品 API,数据来自 Amazon Reviews 2023(Electronics 类目里筛出的笔记本)。
只用 Python 标准库(`http.server` + `sqlite3` + `json`),所以**不需要 `pip install`、不需要联网**即可运行。

> 说明:后端直接在**本地(Windows)**上跑,数据库用项目内的相对路径加载
> (`backend/data/catalog.db`),不依赖任何机器专属路径。任何人克隆本仓库后都能直接跑通。

## 目录结构

```
backend/
  server.py            # API 服务(标准库)
  data/catalog.db      # 商品数据库(由 tools/build_laptops.py 生成)
tools/
  build_laptops.py     # 从 Amazon 数据集抽取笔记本 -> SQLite
  verify_db.py         # 快速查看 db 行数与样例
```

## 0. 准备数据库(第一次运行前)

仓库不包含 `backend/data/catalog.db`,需要先生成一次。数据库现在是**通用商品目录**
(不再只有笔记本),用户输入什么品类就能搜到什么,取决于你下载了哪些品类文件。

最省事:一键下载所有品类前 250MB 并自动构建(在项目根目录):

```powershell
python tools\download_all.py
```

或自己挑品类、手动构建(详见 `tools/README.md`):

```powershell
# 下载若干品类的前一小段(示例:电子 + 手机)
curl.exe -r 0-262144000 -o tools\meta_Electronics_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"
curl.exe -r 0-262144000 -o tools\meta_Cell_Phones_and_Accessories_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz"

# 一次喂入多个品类文件构建(--limit 是合计上限)
python tools\build_laptops.py --input tools\meta_Electronics_part.jsonl.gz tools\meta_Cell_Phones_and_Accessories_part.jsonl.gz --limit 3000 --out backend\data\catalog.db
```

(若已有别人生成好的 `catalog.db`,直接拷到 `backend\data\catalog.db` 即可。)

## 1. 启动后端(本地 Windows)

在**项目根目录**执行任一方式,保持这个终端开着(它会一直服务):

PowerShell:

```powershell
.\run_backend.ps1
```

CMD / 双击:

```bat
run_backend.bat
```

或直接用 Python(路径都是相对项目根,换台机器同样可用):

```powershell
python backend\server.py --host 0.0.0.0 --port 8000
```

看到 `Listening on http://0.0.0.0:8000` 即成功。自检:

```powershell
curl.exe -s http://127.0.0.1:8000/health
# {"status": "ok", "count": 805}
```

## 2. 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查,返回商品总数 |
| GET | `/api/v1/search?q=<关键词>&max_price=<整数>&min_rating=<小数>&sort=<popular\|price\|rating>&limit=<整数>` | 搜索,返回 `{"results":[...]}` |
| GET | `/api/v1/products/{id}` | 按 ASIN 取单个商品 |

示例:

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/search?q=gaming&max_price=1200&limit=5"
```

## 3. 模拟器怎么连到后端

后端跑在 Windows 上,Android 模拟器也跑在同一台 Windows 上。App 里配置的地址是
`http://10.0.2.2:8000`(`10.0.2.2` 是模拟器访问「宿主机 localhost」的固定地址)。

- 后端用 `--host 0.0.0.0` 监听,模拟器通过 `10.0.2.2:8000` 访问宿主机,通常**开箱即用**。
- 如果连不上,用 `adb reverse` 兜底(在 Windows 上执行,需先启动模拟器):

```powershell
adb reverse tcp:8000 tcp:8000
```

  这样模拟器访问自身 `localhost:8000` 会被转发到宿主机后端。
  若用这种方式,把 `MainActivity.kt` 里的 `BACKEND_BASE_URL` 改成 `http://localhost:8000` 亦可
  (`10.0.2.2` 一般已经够用,不必改)。

验证后端是否可访问(在 Windows PowerShell 里):

```powershell
curl.exe -s http://localhost:8000/health
```

## 4. 扩充 / 重建数据(更多品类或更多商品)

想覆盖更多品类,就多下载几个 `meta_<Category>.jsonl.gz`(命名规则和完整品类列表见
`tools/README.md`),然后一次性喂给构建脚本;想要每类更多商品就把分段下大、`--limit` 调大:

```powershell
# 例:再加入服饰、家居、运动三个品类
curl.exe -r 0-262144000 -o tools\meta_Clothing_Shoes_and_Jewelry_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz"
curl.exe -r 0-262144000 -o tools\meta_Home_and_Kitchen_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz"
curl.exe -r 0-262144000 -o tools\meta_Sports_and_Outdoors_part.jsonl.gz "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz"

# 重建(把所有想要的品类文件都列上,--limit 是合计上限)
python tools\build_laptops.py --input tools\meta_*.jsonl.gz --limit 8000 --out backend\data\catalog.db

# 查看结果
python tools\verify_db.py backend\data\catalog.db
```

> PowerShell 不会自动展开 `tools\meta_*.jsonl.gz` 通配符,手动把各文件列全,或用
> `--input (Get-ChildItem tools\meta_*.jsonl.gz).FullName`。

后端每次请求都重新读库,重建后**无需重启**即可生效。

## 说明 / 已知限制

- 价格是数据集里的**美元**数值,App 里目前仍显示 `¥` 符号(仅符号,不换算),属展示细节。
- 部分商品的 `price` 在 2023 数据里是 `None`,库里存为 0;不因缺价而丢弃真实笔记本。
- 极少数配件可能仍会漏进目录;`tools/build_laptops.py` 里的 `EXCLUDE_KEYWORDS` / 正则可继续收紧。
