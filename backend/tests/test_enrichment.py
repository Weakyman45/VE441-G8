"""enrichment.py 单元测试:向量编解码 / 余弦 / hash 兜底 / 属性分层 /
离线富集 + 在线视觉召回(全程 hash 兜底,无网络、无 numpy)。"""
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import enrichment as e  # noqa: E402


class TestPureFunctions(unittest.TestCase):
    def test_vec_roundtrip(self):
        v = [0.5, -1.25, 3.0, 0.0]
        dec = e.decode_vec(e.encode_vec(v))
        self.assertEqual(len(dec), 4)
        for a, b in zip(v, dec):
            self.assertAlmostEqual(a, b, places=4)

    def test_vec_edge_cases(self):
        self.assertEqual(e.encode_vec([]), "")
        self.assertEqual(e.decode_vec(""), [])
        self.assertEqual(e.decode_vec(None), [])
        self.assertEqual(e.decode_vec("not-base64!!"), [])

    def test_cosine(self):
        self.assertAlmostEqual(e.cosine([1, 0, 0], [1, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(e.cosine([1, 0], [0, 1]), 0.0, places=5)
        self.assertEqual(e.cosine([], [1]), 0.0)
        self.assertEqual(e.cosine([1, 2], [1, 2, 3]), 0.0)   # 维度不一致
        self.assertEqual(e.cosine([0, 0], [0, 0]), 0.0)       # 零向量

    def test_hash_embed_deterministic(self):
        a = e._hash_embed("seed")
        self.assertEqual(a, e._hash_embed("seed"))
        self.assertEqual(len(a), e._HASH_DIM)
        self.assertNotEqual(a, e._hash_embed("other"))

    def test_embed_bytes_hash_fallback(self):
        v1 = e.embed_image_bytes(b"abc", provider="hash")
        v2 = e.embed_image_bytes(b"abc", provider="hash")
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, e.embed_image_bytes(b"xyz", provider="hash"))

    def test_physical_attributes(self):
        p = e.extract_physical_attributes(
            {"weight_kg": 1.0, "battery": "10h", "price": 500, "display": "15in"})
        self.assertTrue(p["lightweight"])
        self.assertEqual(p["weight_kg"], 1.0)
        self.assertEqual(p["price"], 500)
        self.assertFalse(e.extract_physical_attributes({"weight_kg": 3.0})["lightweight"])
        self.assertEqual(e.extract_physical_attributes({}), {})

    def test_build_enriched_text(self):
        row = {"name": "White Sneaker", "summary": "comfy"}
        visual = {"colors": ["white"], "style": ["casual"],
                  "keywords": ["sneaker", "canvas"], "product_category": "shoes"}
        txt = e.build_enriched_text(row, visual, {"lightweight": True}).lower()
        for kw in ("white sneaker", "shoes", "casual", "canvas", "lightweight"):
            self.assertIn(kw, txt)

    def test_parse_json_object(self):
        self.assertEqual(e._parse_json_object('{"a":1}'), {"a": 1})
        self.assertEqual(e._parse_json_object('```json\n{"a":2}\n```'), {"a": 2})
        self.assertEqual(e._parse_json_object('noise {"a":3} tail'), {"a": 3})
        self.assertEqual(e._parse_json_object('not json'), {})


class TestEnrichAndRecall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        self._key = os.environ.pop("DASHSCOPE_API_KEY", None)  # 无网/无 key
        # 线上/线下统一走 hash provider(dashscope 失败不再自动兜底,故须显式指定)
        from types import SimpleNamespace
        self._cfg_patch = patch("engine.enrichment.CONFIG",
                                SimpleNamespace(embedding_provider="hash",
                                                visual_top_k=40))
        self._cfg_patch.start()
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TABLE laptops (id TEXT, name TEXT, image_url TEXT, price INTEGER, "
            "rating REAL, weight_kg REAL, summary TEXT)")
        conn.executemany(
            "INSERT INTO laptops VALUES (?,?,?,?,?,?,?)",
            [("a", "White Running Shoes", "http://img/a.jpg", 300, 4.5, 0.3, "light"),
             ("b", "Leather Boots", "http://img/b.jpg", 600, 4.0, 1.2, "formal"),
             ("c", "No Image Item", None, 100, 3.0, 0.5, "x")])
        conn.commit()
        conn.close()
        e._INDEX_CACHE.clear()

    def tearDown(self):
        e._INDEX_CACHE.clear()
        self._cfg_patch.stop()
        if self._key is not None:
            os.environ["DASHSCOPE_API_KEY"] = self._key
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_ensure_columns_idempotent(self):
        conn = sqlite3.connect(self.db)
        e.ensure_columns(conn)
        e.ensure_columns(conn)  # 第二次不报错
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        conn.close()
        for c in ("image_embedding", "visual_attrs", "enriched_text"):
            self.assertIn(c, cols)

    def test_enrich_then_visual_recall(self):
        stats = e.enrich_catalog(self.db, provider="hash", with_vl=False, verbose=False)
        self.assertEqual(stats["processed"], 2)     # 'c' 无 image_url,被排除
        self.assertEqual(stats["embeddings"], 2)
        self.assertTrue(e.has_enrichment(self.db))

        idx = e.get_index(self.db)
        self.assertEqual(idx.size, 2)
        self.assertLessEqual(len(idx.search(e._hash_embed("q"), top_k=5)), 2)

        res = e.visual_recall(b"userphoto", db_path=self.db, top_k=5)
        self.assertTrue(res)
        self.assertTrue(all("_visual_score" in r for r in res))
        self.assertTrue({r["id"] for r in res} <= {"a", "b"})

    def test_fetch_products_by_ids_preserves_order(self):
        e.enrich_catalog(self.db, provider="hash", with_vl=False, verbose=False)
        got = e.fetch_products_by_ids(self.db, ["b", "a"])
        self.assertEqual([g["id"] for g in got], ["b", "a"])
        self.assertEqual(e.fetch_products_by_ids(self.db, []), [])

    def test_attach_visual_scores(self):
        e.enrich_catalog(self.db, provider="hash", with_vl=False, verbose=False)
        cands = [{"id": "a", "name": "x"}, {"id": "zzz", "name": "y"}]
        e.attach_visual_scores(cands, b"userphoto", db_path=self.db)
        self.assertIn("_visual_score", cands[0])      # 命中索引 → 打分
        self.assertNotIn("_visual_score", cands[1])    # 未知 id 不动

    def test_has_enrichment_false_before(self):
        self.assertFalse(e.has_enrichment(self.db))   # 富集前
        self.assertFalse(e.has_enrichment(os.path.join(self.dir, "nope.db")))


if __name__ == "__main__":
    unittest.main()
