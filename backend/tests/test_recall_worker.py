"""recall_worker.py ????:?? Agent ?????(?? / ??)+ ???? +
????? + ???? + ?????enrichment ??????? mock,????/???"""
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import enrichment as e  # noqa: E402
from engine.models import SessionState, PreferenceProfile  # noqa: E402
from engine.worker import recall_worker as rw  # noqa: E402
from engine.worker.recall_worker import RecallAgent, RecallResult  # noqa: E402


def make_cfg(**kw):
    """Minimal cfg for recall tests (not a full ExpConfig)."""
    from types import SimpleNamespace
    base = dict(
        visual_top_k=40,
        text_top_k=40,
        text_semantic=True,
        embedding_provider="hash",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_plan(*, text=True, visual=False, infer_cat=False):
    from engine.modality_router import RoutePlan
    if text and visual:
        modality = "text_image"
    elif visual:
        modality = "image"
    else:
        modality = "text"
    return RoutePlan(
        modality=modality,
        do_text_recall=text,
        do_visual_recall=visual,
        infer_category_from_image=infer_cat,
        reverse_verify=False,
        do_verify=True,
        reason="unit-test",
    )


def make_state(keywords=("shoes",), category="footwear"):
    pref = PreferenceProfile(search_keywords=list(keywords), category=category)
    return SessionState(session_id="s", preference=pref)


IMG = (b"fake-bytes", "image/jpeg")


class TestMergeCandidates(unittest.TestCase):
    def test_union_dedup_keeps_first(self):
        a = [{"id": "1", "src": "a"}, {"id": "2", "src": "a"}]
        b = [{"id": "2", "src": "b"}, {"id": "3", "src": "b"}]
        merged = rw._merge_candidates([a, b])
        self.assertEqual([c["id"] for c in merged], ["1", "2", "3"])
        # ????????????(?? a)
        self.assertEqual(next(c for c in merged if c["id"] == "2")["src"], "a")

    def test_skips_empty_and_idless(self):
        merged = rw._merge_candidates([[], [{"name": "no id"}], [{"id": ""}], [{"id": "9"}]])
        self.assertEqual([c["id"] for c in merged], ["9"])


class TestRecallAgent(unittest.TestCase):
    def setUp(self):
        self.text_hits = [{"id": "t1", "name": "A"}, {"id": "t2", "name": "B"}]
        self.search_fn = lambda params: [dict(x) for x in self.text_hits]
        self.agent = RecallAgent(self.search_fn)
        # Avoid touching real catalog.db text index in unit tests.
        self._text_emb = patch("engine.enrichment.has_text_embeddings", return_value=False)
        self._text_emb.start()

    def tearDown(self):
        self._text_emb.stop()

    def test_text_only_no_visual_calls(self):
        with patch("engine.enrichment.has_enrichment", return_value=True) as he, \
             patch("engine.enrichment.visual_recall") as vr, \
             patch("engine.enrichment.attach_visual_scores") as avs:
            res = self.agent.recall(state=make_state(), route_plan=make_plan(text=True),
                                    cfg=make_cfg(), query_image=None)
        self.assertIsInstance(res, RecallResult)
        self.assertEqual([c["id"] for c in res.candidates], ["t1", "t2"])
        self.assertEqual(res.text_count, 2)
        self.assertEqual(res.visual_count, 0)
        vr.assert_not_called()
        avs.assert_not_called()
        he.assert_not_called()

    def test_text_image_merge_dedup_and_scores(self):
        visual_hits = [{"id": "v1", "name": "V"}, {"id": "t2", "name": "B'"}]

        def fake_attach(cands, data, mime=None):
            for c in cands:
                c["_visual_score"] = 0.5
            return cands

        with patch("engine.enrichment.has_enrichment", return_value=True), \
             patch("engine.enrichment.visual_recall", return_value=visual_hits) as vr, \
             patch("engine.enrichment.attach_visual_scores", side_effect=fake_attach) as avs:
            res = self.agent.recall(state=make_state(),
                                    route_plan=make_plan(text=True, visual=True),
                                    cfg=make_cfg(), query_image=IMG)

        self.assertEqual([c["id"] for c in res.candidates], ["t2", "t1", "v1"])
        self.assertEqual(res.text_count, 2)
        self.assertEqual(res.visual_count, 2)
        self.assertTrue(all("_visual_score" in c for c in res.candidates))
        self.assertTrue(all("_rrf_score" in c for c in res.candidates))
        # t2 appears in both lists → highest RRF
        self.assertGreater(res.candidates[0]["_rrf_score"], res.candidates[1]["_rrf_score"])
        vr.assert_called_once()
        avs.assert_called_once()

    def test_image_only_skips_text_recall(self):
        search_fn = unittest_fail_if_called(self)
        agent = RecallAgent(search_fn)
        with patch("engine.enrichment.has_enrichment", return_value=True), \
             patch("engine.enrichment.visual_recall",
                   return_value=[{"id": "v1"}]) as vr, \
             patch("engine.enrichment.attach_visual_scores", side_effect=lambda c, *a, **k: c):
            res = agent.recall(state=make_state(),
                               route_plan=make_plan(text=False, visual=True, infer_cat=True),
                               cfg=make_cfg(), query_image=IMG)
        self.assertEqual([c["id"] for c in res.candidates], ["v1"])
        self.assertEqual(res.text_count, 0)
        self.assertEqual(res.visual_count, 1)
        _, kwargs = vr.call_args
        self.assertEqual(kwargs.get("category"), "footwear")

    def test_visual_gracefully_off_without_enrichment(self):
        with patch("engine.enrichment.has_enrichment", return_value=False), \
             patch("engine.enrichment.visual_recall") as vr, \
             patch("engine.enrichment.attach_visual_scores") as avs:
            res = self.agent.recall(state=make_state(),
                                    route_plan=make_plan(text=True, visual=True),
                                    cfg=make_cfg(), query_image=IMG)
        self.assertEqual([c["id"] for c in res.candidates], ["t1", "t2"])
        self.assertEqual(res.visual_count, 0)
        vr.assert_not_called()
        avs.assert_not_called()

    def test_visual_gracefully_off_without_image(self):
        with patch("engine.enrichment.has_enrichment", return_value=True), \
             patch("engine.enrichment.visual_recall") as vr, \
             patch("engine.enrichment.attach_visual_scores") as avs:
            res = self.agent.recall(state=make_state(),
                                    route_plan=make_plan(text=True, visual=True),
                                    cfg=make_cfg(), query_image=None)
        self.assertEqual([c["id"] for c in res.candidates], ["t1", "t2"])
        self.assertEqual(res.visual_count, 0)
        vr.assert_not_called()
        avs.assert_not_called()

    def test_result_to_dict_shape(self):
        with patch("engine.enrichment.has_enrichment", return_value=False):
            res = self.agent.recall(state=make_state(), route_plan=make_plan(text=True),
                                    cfg=make_cfg(), query_image=None)
        d = res.to_dict()
        self.assertEqual(set(d), {"count", "text", "visual"})
        self.assertEqual(d["count"], 2)
        self.assertEqual(d["text"], 2)

    def test_text_semantic_union_with_keywords(self):
        self._text_emb.stop()
        sem_hits = [{"id": "s1", "name": "Semantic", "_text_score": 0.9}]
        with patch("engine.enrichment.has_text_embeddings", return_value=True), \
             patch("engine.enrichment.text_semantic_recall", return_value=sem_hits) as sem, \
             patch("engine.enrichment.has_enrichment", return_value=False), \
             patch("engine.enrichment.attach_text_scores", side_effect=lambda c, *a, **k: c):
            res = self.agent.recall(
                state=make_state(),
                route_plan=make_plan(text=True),
                cfg=make_cfg(text_semantic=True),
                query_image=None,
            )
        self.assertEqual(set(c["id"] for c in res.candidates), {"t1", "t2", "s1"})
        self.assertTrue(all("_rrf_score" in c for c in res.candidates))
        sem.assert_called_once()

        with patch("engine.enrichment.has_text_embeddings", return_value=True), \
             patch("engine.enrichment.text_semantic_recall") as sem2, \
             patch("engine.enrichment.has_enrichment", return_value=False):
            res2 = self.agent.recall(
                state=make_state(),
                route_plan=make_plan(text=True),
                cfg=make_cfg(text_semantic=False),
                query_image=None,
            )
        self.assertEqual([c["id"] for c in res2.candidates], ["t1", "t2"])
        sem2.assert_not_called()
        self._text_emb.start()


class TestVisualRecallRanking(unittest.TestCase):
    """端到端:输入一张图片 → 图片召回 → 按视觉相似度给出**排名**(可控可断言)。

    用 hash provider 让"用户图向量"确定可复现,再把三件商品的图向量分别设成与它
    完全相同 / 部分相同 / 完全相反,于是排名必然是 match > mid > opposite。"""

    QUERY = b"the-user-uploaded-photo"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "catalog.db")
        self._key = os.environ.pop("DASHSCOPE_API_KEY", None)  # 无网/无 key
        from types import SimpleNamespace
        self._cfg = patch("engine.enrichment.CONFIG",
                          SimpleNamespace(embedding_provider="hash", visual_top_k=40))
        self._cfg.start()
        e._INDEX_CACHE.clear()

        # 用户图在 hash provider 下的确定向量(visual_recall 内部会算出同一个)
        q = e.embed_image_bytes(self.QUERY, provider="hash")
        match = list(q)                                            # cos = 1.0
        mid = [v if i % 2 == 0 else 0.0 for i, v in enumerate(q)]  # 0 < cos < 1
        opposite = [-v for v in q]                                 # cos = -1.0

        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE laptops (id TEXT, name TEXT, image_embedding TEXT)")
        conn.executemany(
            "INSERT INTO laptops VALUES (?,?,?)",
            [("match", "Very similar", e.encode_vec(match)),
             ("mid", "Somewhat similar", e.encode_vec(mid)),
             ("opposite", "Opposite look", e.encode_vec(opposite))],
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        e._INDEX_CACHE.clear()
        self._cfg.stop()
        if self._key is not None:
            os.environ["DASHSCOPE_API_KEY"] = self._key
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_visual_recall_orders_by_similarity(self):
        """直接调图片召回:返回结果按 _visual_score 降序排名。"""
        ranked = e.visual_recall(self.QUERY, db_path=self.db, top_k=10)
        self.assertEqual([r["id"] for r in ranked], ["match", "mid", "opposite"])
        scores = [r["_visual_score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))   # 严格降序
        self.assertAlmostEqual(scores[0], 1.0, places=3)         # 最佳命中≈1.0
        self.assertLess(scores[-1], scores[0])

    def test_top_k_truncates_ranking(self):
        """top_k 只取排名前 K 名。"""
        ranked = e.visual_recall(self.QUERY, db_path=self.db, top_k=2)
        self.assertEqual([r["id"] for r in ranked], ["match", "mid"])

    def test_recall_agent_image_only_returns_ranking(self):
        """经召回 Agent(纯图片路由):候选即图片召回排名,且每条带 _visual_score。"""
        real_vr = e.visual_recall
        real_avs = e.attach_visual_scores

        def vr_on_tmp(image_bytes, **kw):
            kw.pop("db_path", None)
            return real_vr(image_bytes, db_path=self.db, **kw)

        def avs_on_tmp(candidates, image_bytes, **kw):
            kw.pop("db_path", None)
            return real_avs(candidates, image_bytes, db_path=self.db, **kw)

        def _no_text(_params):
            self.fail("纯图片召回不应触发文本检索")

        agent = RecallAgent(_no_text)
        with patch("engine.enrichment.has_enrichment", return_value=True), \
             patch("engine.enrichment.visual_recall", side_effect=vr_on_tmp), \
             patch("engine.enrichment.attach_visual_scores", side_effect=avs_on_tmp):
            res = agent.recall(
                state=make_state(category=""),
                route_plan=make_plan(text=False, visual=True, infer_cat=False),
                cfg=make_cfg(),
                query_image=(self.QUERY, "image/jpeg"),
            )
        self.assertEqual([c["id"] for c in res.candidates], ["match", "mid", "opposite"])
        self.assertEqual(res.visual_count, 3)
        self.assertEqual(res.text_count, 0)
        self.assertTrue(all("_visual_score" in c for c in res.candidates))
        scores = [c["_visual_score"] for c in res.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))   # 排名严格降序


def unittest_fail_if_called(testcase):
    def _fn(_params):
        testcase.fail("search_fn ????????????")
    return _fn


if __name__ == "__main__":
    unittest.main()
