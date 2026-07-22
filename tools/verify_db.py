import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "catalog.db"
conn = sqlite3.connect(path)
count = conn.execute("select count(*) from laptops").fetchone()[0]
print(f"rows = {count}")
withprice = conn.execute("select count(*) from laptops where price > 0").fetchone()[0]
print(f"rows with price > 0 = {withprice}")
print("--- sample ---")
for row in conn.execute(
    "select name, price, rating, rating_number, platform from laptops "
    "where price > 0 order by rating_number desc limit 8"
):
    print(f"  ${row[1]:>5} | {row[2]}* ({row[3]}) | {row[4]:8} | {row[0][:60]}")
conn.close()
