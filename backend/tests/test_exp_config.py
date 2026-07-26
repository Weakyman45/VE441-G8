"""exp_config.py 单元测试:开关解析 / 默认值 / 派生属性 / 边界裁剪。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import exp_config  # noqa: E402


class TestExpConfig(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("VS_")}
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ.update(self._saved)

    def test_defaults_full_system(self):
        c = exp_config.load_config()
        self.assertTrue(c.modality_routing)
        self.assertTrue(c.visual_recall)
        self.assertTrue(c.enrichment)
        self.assertTrue(c.source_layering)
        self.assertEqual(c.verifier, "llm")
        self.assertTrue(c.reviews)
        self.assertEqual(c.visual_top_k, 40)
        self.assertEqual(c.review_top_k, 20)
        self.assertEqual(c.embedding_provider, "dashscope")
        self.assertTrue(c.verifier_enabled)
        self.assertTrue(c.verifier_uses_llm)
        self.assertTrue(c.verifier_unknown_tolerant)

    def test_flag_parsing(self):
        for val in ("0", "false", "no", "off", ""):
            os.environ["VS_MODALITY_ROUTING"] = val
            # 空字符串取默认(True);其余 falsey 取 False
            expect = (val == "")
            self.assertEqual(exp_config.load_config().modality_routing, expect, val)
        for val in ("1", "true", "yes", "on", "TRUE", "On"):
            os.environ["VS_MODALITY_ROUTING"] = val
            self.assertTrue(exp_config.load_config().modality_routing, val)
        os.environ["VS_MODALITY_ROUTING"] = "garbage"
        self.assertFalse(exp_config.load_config().modality_routing)  # 非法非空 → falsey

    def test_verifier_choices(self):
        for v in ("off", "rule", "llm_strict", "llm"):
            os.environ["VS_VERIFIER"] = v
            self.assertEqual(exp_config.load_config().verifier, v)
        os.environ["VS_VERIFIER"] = "bogus"
        self.assertEqual(exp_config.load_config().verifier, "llm")  # 非法 → 默认

    def test_verifier_derived_properties(self):
        os.environ["VS_VERIFIER"] = "off"
        c = exp_config.load_config()
        self.assertFalse(c.verifier_enabled)
        self.assertFalse(c.verifier_uses_llm)
        os.environ["VS_VERIFIER"] = "rule"
        c = exp_config.load_config()
        self.assertTrue(c.verifier_enabled)
        self.assertFalse(c.verifier_uses_llm)
        os.environ["VS_VERIFIER"] = "llm_strict"
        c = exp_config.load_config()
        self.assertTrue(c.verifier_uses_llm)
        self.assertFalse(c.verifier_unknown_tolerant)  # strict 不容错
        os.environ["VS_VERIFIER"] = "llm"
        self.assertTrue(exp_config.load_config().verifier_unknown_tolerant)

    def test_topk_clamped(self):
        os.environ["VS_VISUAL_TOPK"] = "1"
        self.assertEqual(exp_config.load_config().visual_top_k, 5)     # 下限 5
        os.environ["VS_VISUAL_TOPK"] = "9999"
        self.assertEqual(exp_config.load_config().visual_top_k, 200)   # 上限 200
        os.environ["VS_VISUAL_TOPK"] = "notint"
        self.assertEqual(exp_config.load_config().visual_top_k, 40)    # 非法 → 默认

    def test_review_topk_and_flag(self):
        os.environ["VS_REVIEW_TOPK"] = "1"
        self.assertEqual(exp_config.load_config().review_top_k, 5)
        os.environ["VS_REVIEW_TOPK"] = "9999"
        self.assertEqual(exp_config.load_config().review_top_k, 200)
        os.environ["VS_REVIEW_TOPK"] = "notint"
        self.assertEqual(exp_config.load_config().review_top_k, 20)
        os.environ["VS_REVIEWS"] = "0"
        self.assertFalse(exp_config.load_config().reviews)

    def test_embedding_provider(self):
        os.environ["VS_EMBEDDING_PROVIDER"] = "hash"
        self.assertEqual(exp_config.load_config().embedding_provider, "hash")
        os.environ["VS_EMBEDDING_PROVIDER"] = "bogus"
        self.assertEqual(exp_config.load_config().embedding_provider, "dashscope")

    def test_summary_keys(self):
        s = exp_config.load_config().summary()
        self.assertEqual(set(s), {"modality_routing", "visual_recall", "enrichment",
                                  "source_layering", "verifier", "reviews",
                                  "intent_shortcircuit", "planner_replan", "planner_llm",
                                  "memory", "max_replans",
                                  "visual_top_k", "review_top_k", "embedding_provider"})

    def test_planner_flags_default_on(self):
        c = exp_config.load_config()
        self.assertTrue(c.intent_shortcircuit)
        self.assertTrue(c.planner_replan)
        self.assertTrue(c.planner_llm)
        self.assertTrue(c.memory)
        self.assertEqual(c.max_replans, 2)
        os.environ["VS_MEMORY"] = "0"
        os.environ["VS_PLANNER_LLM"] = "0"
        os.environ["VS_MAX_REPLANS"] = "9"
        c2 = exp_config.load_config()
        self.assertFalse(c2.memory)
        self.assertFalse(c2.planner_llm)
        self.assertEqual(c2.max_replans, 5)  # clamped


if __name__ == "__main__":
    unittest.main()
