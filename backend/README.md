# VoiceShop++ 后端(做法 B:后端持有数据 + App 联网查询)

一个**零第三方依赖**的只读商品 API,数据来自 Amazon Reviews 2023(Electronics 类目里筛出的笔记本)。
只用 Python 标准库(`http.server` + `sqlite3` + `json`),所以**不需要 `pip install`、不需要联网**即可运行,非常适合 WSL / 离线环境。

## 目录结构

```
backend/
  server.py            # API 服务(标准库)
  data/catalog.db      # 商品数据库(由 tools/build_laptops.py 生成)
tools/
  build_laptops.py     # 从 Amazon 数据集抽取笔记本 -> SQLite
  verify_db.py         # 快速查看 db 行数与样例
```

## 1. 启动后端(在 WSL 里)

保持这个终端开着(它会一直服务):

```bash
python3 /mnt/c/Users/f0407/Desktop/VE441-G8-main/backend/server.py --port 8000
```

看到 `Listening on http://0.0.0.0:8000` 即成功。自检:

```bash
curl -s http://127.0.0.1:8000/health
# {"status": "ok", "count": 805}
```

## 2. 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查,返回商品总数 |
| GET | `/api/v1/search?q=<关键词>&max_price=<整数>&min_rating=<小数>&sort=<popular\|price\|rating>&limit=<整数>` | 搜索,返回 `{"results":[...]}` |
| GET | `/api/v1/products/{id}` | 按 ASIN 取单个商品 |

示例:

```bash
curl -s 'http://127.0.0.1:8000/api/v1/search?q=gaming&max_price=1200&limit=5'
```

## 3. 模拟器怎么连到后端

Android 模拟器跑在 Windows 上,后端跑在 WSL 里。App 里配置的地址是
`http://10.0.2.2:8000`(`10.0.2.2` 是模拟器访问「宿主机 localhost」的固定地址)。

- WSL2 默认会把监听端口转发到 Windows 的 `localhost`,所以 `10.0.2.2:8000` 通常**开箱即用**。
- 如果连不上,用 `adb reverse` 兜底(在 **Windows** 上执行,需先启动模拟器):

```powershell
adb reverse tcp:8000 tcp:8000
```

  这样模拟器访问自身 `localhost:8000` 会被转发到宿主机再到 WSL。
  若用这种方式,把 `MainActivity.kt` 里的 `BACKEND_BASE_URL` 改成 `http://localhost:8000` 亦可
  （`10.0.2.2` 一般已经够用,不必改）。

验证转发是否正常(在 Windows PowerShell 里):

```powershell
curl.exe -s http://localhost:8000/health
```

## 4. 扩充 / 重建数据

当前 `catalog.db` 用 Electronics 元数据的前 ~120MB 构建,含约 800 台真实笔记本。
想要更多,下载更大的分段再重建即可(在 WSL 里):

```bash
# 下载更大一块(例如 400MB);数据集是逐行 JSONL,截断也能用
curl -sS -r 0-419430400 -o /tmp/meta.gz \
  'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz'

# 重建(--limit 调大)
python3 /mnt/c/Users/f0407/Desktop/VE441-G8-main/tools/build_laptops.py \
  --input /tmp/meta.gz --limit 5000 \
  --out /mnt/c/Users/f0407/Desktop/VE441-G8-main/backend/data/catalog.db

# 查看结果
python3 /mnt/c/Users/f0407/Desktop/VE441-G8-main/tools/verify_db.py \
  /mnt/c/Users/f0407/Desktop/VE441-G8-main/backend/data/catalog.db
```

后端每次请求都重新读库,重建后**无需重启**即可生效。

## 说明 / 已知限制

- 价格是数据集里的**美元**数值,App 里目前仍显示 `¥` 符号(仅符号,不换算),属展示细节。
- 部分商品的 `price` 在 2023 数据里是 `None`,库里存为 0;不因缺价而丢弃真实笔记本。
- 极少数配件可能仍会漏进目录;`tools/build_laptops.py` 里的 `EXCLUDE_KEYWORDS` / 正则可继续收紧。
