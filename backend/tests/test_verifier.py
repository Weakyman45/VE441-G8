"""verifier.py 单元测试:off / 规则粗筛 / 规则品类+must-have / LLM 判定
(unknown 容错 vs 严格)/ LLM 失败回退规则。LLM 调用全部 mock,无网络。"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import verifier  # noqa: E402
from engine.exp_config import ExpConfig  # noqa: E402
from engine.models import PreferenceProfile  # noqa: E402


def cfg(mode="llm", **kw):
    base = dict(
        modality_routing=True,
        visual_recall=True,
        enrichment=True,
        source_layering=True,
        verifier=mode,
        reviews=True,
        intent_shortcircuit=True,
        planner_replan=True,
        planner_llm=True,
        memory=True,
        max_replans=2,
        visual_top_k=40,
        review_top_k=20,
        embedding_provider="hash",
    )
    base.update(kw)
    return ExpConfig(**base)


def item(pid, name, price=100, platform="", summary="", enriched=""):
    return {"id": pid, "name": name, "price": price, "platform": platform,
            "summary": summary, "enriched_text": enriched}


class TestVerifier(unittest.TestCase):
    def test_off_keeps_all(self):
        r = verifier.verify_candidates(PreferenceProfile(category="shoes"),
                                       [item("1", "x"), item("2", "y")], cfg("off"))
        self.assertEqual(len(r.kept), 2)
        self.assertEqual(r.method, "off")

    def test_rule_budget_reject(self):
        prof = PreferenceProfile(category="shoes", budget=500)
        cands = [item("1", "white shoes", price=300), item("2", "white shoes", price=9000)]
        r = verifier.verify_candidates(prof, cands, cfg("rule"))
        kept = {k["id"] for k in r.kept}
        self.assertEqual(kept, {"1"})
        self.assertTrue(any("budget" in x["reason"] for x in r.rejected))

    def test_rule_platform_reject(self):
        prof = PreferenceProfile(platform="Windows")
        cands = [item("1", "laptop", platform="macOS"), item("2", "laptop", platform="Windows")]
        r = verifier.verify_candidates(prof, cands, cfg("rule"))
        self.assertEqual({k["id"] for k in r.kept}, {"2"})

    def test_rule_category_mismatch(self):
        prof = PreferenceProfile(category="shoes")
        cands = [item("1", "white running shoes"), item("2", "gaming laptop computer")]
        r = verifier.verify_candidates(prof, cands, cfg("rule"))
        self.assertEqual({k["id"] for k in r.kept}, {"1"})

    def test_rule_category_unknown_tolerant(self):
        # 无明确品类 → 无从判 → 保留(unknown 容错)
        r = verifier.verify_candidates(PreferenceProfile(category=""),
                                       [item("1", "anything")], cfg("rule"))
        self.assertEqual(len(r.kept), 1)

    def test_rule_musthave(self):
        prof = PreferenceProfile(category="shoes", hard=["waterproof", "板鞋"])
        cands = [item("1", "white shoes waterproof"), item("2", "white shoes")]
        r = verifier.verify_candidates(prof, cands, cfg("rule"))
        kept = {k["id"] for k in r.kept}
        self.assertIn("1", kept)      # waterproof 命中;板鞋(非 ascii)容错
        self.assertNotIn("2", kept)   # 缺 waterproof
        self.assertTrue(any("waterproof" in x["reason"] for x in r.rejected))

    def test_llm_unknown_tolerant_vs_strict(self):
        prof = PreferenceProfile(category="shoes")
        cands = [item("1", "a"), item("2", "b"), item("3", "c")]
        fake = {"results": [
            {"id": "1", "category_match": "yes", "must_haves_met": "yes", "keep": True},
            {"id": "2", "category_match": "unknown", "must_haves_met": "yes", "keep": True},
            {"id": "3", "category_match": "no", "must_haves_met": "yes", "keep": False},
        ]}
        with patch.object(verifier, "qwen_configured", return_value=True), \
             patch.object(verifier, "chat_json", return_value=fake):
            r = verifier.verify_candidates(prof, cands, cfg("llm"))
            self.assertEqual({k["id"] for k in r.kept}, {"1", "2"})   # 容错保留 unknown
            r2 = verifier.verify_candidates(prof, cands, cfg("llm_strict"))
            self.assertEqual({k["id"] for k in r2.kept}, {"1"})        # 严格:unknown 也拒

    def test_llm_fallback_to_rule_on_error(self):
        prof = PreferenceProfile(category="shoes")
        cands = [item("1", "white running shoes"), item("2", "gaming laptop computer")]

        def boom(*a, **k):
            raise RuntimeError("network down")

        with patch.object(verifier, "qwen_configured", return_value=True), \
             patch.object(verifier, "chat_json", side_effect=boom):
            r = verifier.verify_candidates(prof, cands, cfg("llm"))
        self.assertEqual(r.method, "rule(llm_fallback)")
        self.assertEqual({k["id"] for k in r.kept}, {"1"})

    def test_llm_not_configured_falls_back(self):
        prof = PreferenceProfile(category="shoes")
        cands = [item("1", "white running shoes"), item("2", "gaming laptop computer")]
        with patch.object(verifier, "qwen_configured", return_value=False):
            r = verifier.verify_candidates(prof, cands, cfg("llm"))
        self.assertEqual({k["id"] for k in r.kept}, {"1"})

    def test_verify_result_to_dict(self):
        r = verifier.verify_candidates(PreferenceProfile(category="shoes"),
                                       [item("1", "white running shoes")], cfg("rule"))
        d = r.to_dict()
        self.assertEqual(set(d), {"kept", "rejected", "method", "conflicts", "unresolved"})


if __name__ == "__main__":
    unittest.main()
