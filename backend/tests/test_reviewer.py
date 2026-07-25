"""reviewer.py 单元测试:评论读取/质量筛选/聚合/关键词抽取/离线方面富集。
评论只产出 review_aspects(供 must-have 校验与展示),不做向量召回。LLM 全部 mock。"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import reviewer as rv  # noqa: E402


class TestPureFunctions(unittest.TestCase):
    def test_review_key(self):
        self.assertEqual(rv._review_key({"parent_asin": "P1", "asin": "A1"}), "P1")
        self.assertEqual(rv._review_key({"asin": "A1"}), "A1")
        self.assertEqual(rv._review_key({}), "")

    def test_review_quality_ordering(self):
        good = {"verified_purchase": True, "helpful_vote": 10, "text": "x"}
        weak = {"verified_purchase": False, "helpful_vote": 0, "text": "x"}
        self.assertGreater(rv._review_quality(good), rv._review_quality(weak))

    def test_aggregate_review_text(self):
        reviews = [
            {"title": "Great", "text": "very comfortable", "rating": 5,
             "verified_purchase": True, "helpful_vote": 9},
            {"title": "Meh", "text": "runs small", "rating": 2},
        ]
        agg = rv.aggregate_review_text(reviews).lower()
        self.assertIn("comfortable", agg)
        self.assertIn("runs small", agg)
        # 高质量评论排前面
        self.assertLess(agg.index("comfortable"), agg.index("runs small"))

    def test_iter_reviews_skips_bad_lines(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "r.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"parent_asin": "a", "text": "ok"}) + "\n")
                f.write("not json\n")
                f.write("\n")
                f.write(json.dumps({"parent_asin": "b", "text": "ok2"}) + "\n")
            rows = list(rv.iter_reviews(p))
            self.assertEqual(len(rows), 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_extract_aspects_no_llm(self):
        self.assertEqual(rv.extract_review_aspects("x", [{"text": "a"}], with_llm=False), {})
        self.assertEqual(rv.extract_review_aspects("x", [], with_llm=True), {})

    def test_extract_aspects_with_mocked_llm(self):
        fake = {"pros": ["comfy"], "cons": ["pricey"], "issues": [],
                "aspects": {"comfort": "positive"}, "summary": "ok"}
        with patch("engine.llm.qwen_client.qwen_configured", return_value=True), \
             patch("engine.llm.qwen_client.chat_json", return_value=fake):
            asp = rv.extract_review_aspects("Shoe", [{"text": "comfy", "title": "t"}],
                                            with_llm=True)
        self.assertEqual(asp["pros"], ["comfy"])


class TestCollectReviews(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "reviews.jsonl")
        rows = [
            {"parent_asin": "a", "text": "x", "verified_purchase": True, "helpful_vote": 5},
            {"parent_asin": "a", "text": "y", "verified_purchase": False, "helpful_vote": 0},
            {"parent_asin": "a", "text": "z", "verified_purchase": True, "helpful_vote": 10},
            {"parent_asin": "b", "text": "b1"},
            {"parent_asin": "a", "text": ""},          # 空正文,跳过
            {"parent_asin": "zzz", "text": "ignored"},  # 不在目标集
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_collect_caps_and_keeps_best(self):
        buckets = rv.collect_reviews_for_ids(self.path, {"a", "b"}, per_product=2)
        self.assertEqual(set(buckets), {"a", "b"})
        # a 有 3 条有效评论,cap=2,应保留质量最高的 x(helpful5)与 z(helpful10)
        texts = {r["text"] for r in buckets["a"]}
        self.assertEqual(texts, {"x", "z"})
        self.assertEqual(len(buckets["b"]), 1)


class TestEnrichReviews(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        self.reviews = os.path.join(self.dir, "reviews.jsonl")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE laptops (id TEXT, name TEXT)")
        conn.executemany("INSERT INTO laptops VALUES (?,?)",
                         [("a", "Running Shoes"), ("b", "Office Boots"), ("c", "No Reviews")])
        conn.commit()
        conn.close()
        with open(self.reviews, "w", encoding="utf-8") as f:
            for r in [
                {"parent_asin": "a", "title": "Nice", "text": "very comfortable and quiet",
                 "rating": 5, "verified_purchase": True, "helpful_vote": 3},
                {"parent_asin": "a", "title": "Ok", "text": "breathable in summer", "rating": 4},
                {"parent_asin": "b", "title": "Meh", "text": "runs small and heavy", "rating": 2},
            ]:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_ensure_columns_idempotent(self):
        conn = sqlite3.connect(self.db)
        rv.ensure_review_columns(conn)
        rv.ensure_review_columns(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        conn.close()
        for c in ("review_aspects", "review_count_used"):
            self.assertIn(c, cols)
        self.assertNotIn("review_embedding", cols)  # 评论不再存向量

    def test_has_review_aspects_false_before(self):
        self.assertFalse(rv.has_review_aspects(self.db))

    def test_enrich_aspects_only_no_embedding(self):
        """评论只产出 review_aspects,不生成任何向量列(召回交给图文)。"""
        fake = {"pros": ["comfortable", "quiet"], "cons": [], "issues": [],
                "aspects": {"comfort": "positive"}, "summary": "good"}
        with patch("engine.llm.qwen_client.qwen_configured", return_value=True), \
             patch("engine.llm.qwen_client.chat_json", return_value=fake):
            stats = rv.enrich_reviews(self.db, reviews_path=self.reviews,
                                      with_llm=True, verbose=False)
        self.assertEqual(stats["processed"], 2)      # c 无评论
        self.assertEqual(stats["aspects"], 2)
        self.assertNotIn("embeddings", stats)         # 不再产出 embedding

        conn = sqlite3.connect(self.db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        conn.close()
        self.assertNotIn("review_embedding", cols)
        self.assertTrue(rv.has_review_aspects(self.db))

    def test_enrich_with_llm_aspects(self):
        fake = {"pros": ["comfortable", "quiet"], "cons": [], "issues": [],
                "aspects": {"comfort": "positive"}, "summary": "good"}
        with patch("engine.llm.qwen_client.qwen_configured", return_value=True), \
             patch("engine.llm.qwen_client.chat_json", return_value=fake):
            stats = rv.enrich_reviews(self.db, reviews_path=self.reviews,
                                      with_llm=True, verbose=False)
        self.assertEqual(stats["aspects"], 2)
        asp = rv.get_review_aspects("a", self.db)
        self.assertEqual(asp["pros"], ["comfortable", "quiet"])


if __name__ == "__main__":
    unittest.main()
