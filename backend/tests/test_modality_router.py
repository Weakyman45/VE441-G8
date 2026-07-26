"""modality_router.py 单元测试:模态判定 + 三种模态的执行计划 + 消融开关。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import modality_router as mr  # noqa: E402
from engine.exp_config import ExpConfig  # noqa: E402
from engine.models import SessionState, PreferenceProfile  # noqa: E402


def cfg(**kw):
    base = dict(
        modality_routing=True,
        visual_recall=True,
        enrichment=True,
        source_layering=True,
        text_semantic=True,
        verifier="llm",
        reviews=True,
        intent_shortcircuit=True,
        planner_replan=True,
        planner_llm=True,
        memory=True,
        max_replans=2,
        visual_top_k=40,
        text_top_k=40,
        review_top_k=20,
        embedding_provider="hash",
    )
    base.update(kw)
    return ExpConfig(**base)


def session(text="", image=False, category="", hard=None):
    pref = PreferenceProfile(raw_query=text, category=category, hard=hard or [])
    s = SessionState(session_id="s", preference=pref)
    if image:
        s.image_refs = [{"path": "x.jpg", "mime_type": "image/jpeg"}]
    return s


class TestDetectModality(unittest.TestCase):
    def test_text_only(self):
        self.assertEqual(mr.detect_modality(session(text="white shoes")), mr.MODALITY_TEXT)

    def test_image_only(self):
        # 占位文本 + 有图 → 纯图片
        self.assertEqual(
            mr.detect_modality(session(text="[image uploaded]", image=True)),
            mr.MODALITY_IMAGE,
        )

    def test_text_image(self):
        self.assertEqual(
            mr.detect_modality(session(text="white shoes", image=True)),
            mr.MODALITY_TEXT_IMAGE,
        )

    def test_hard_constraint_counts_as_text(self):
        self.assertEqual(mr.detect_modality(session(text="", hard=["waterproof"])),
                         mr.MODALITY_TEXT)

    def test_empty_is_text(self):
        self.assertEqual(mr.detect_modality(session()), mr.MODALITY_TEXT)


class TestRoute(unittest.TestCase):
    def test_text_route(self):
        p = mr.route(session(text="white shoes"), cfg())
        self.assertEqual(p.modality, mr.MODALITY_TEXT)
        self.assertTrue(p.do_text_recall)
        self.assertFalse(p.do_visual_recall)
        self.assertTrue(p.do_verify)

    def test_text_image_route(self):
        p = mr.route(session(text="white shoes", image=True), cfg())
        self.assertEqual(p.modality, mr.MODALITY_TEXT_IMAGE)
        self.assertTrue(p.do_text_recall)
        self.assertTrue(p.do_visual_recall)
        self.assertFalse(p.infer_category_from_image)

    def test_image_route(self):
        p = mr.route(session(text="[image uploaded]", image=True), cfg())
        self.assertEqual(p.modality, mr.MODALITY_IMAGE)
        # A_I fills keywords, then text ∥ visual recall.
        self.assertTrue(p.do_text_recall)
        self.assertTrue(p.do_visual_recall)
        self.assertTrue(p.infer_category_from_image)
        self.assertTrue(p.reverse_verify)

    def test_ablation_fixed_pipeline(self):
        # 关闭模态路由 → 无论啥模态都只走文本召回
        p = mr.route(session(text="white shoes", image=True), cfg(modality_routing=False))
        self.assertTrue(p.do_text_recall)
        self.assertFalse(p.do_visual_recall)
        self.assertIn("fixed", p.reason)

    def test_ablation_visual_recall_off(self):
        p = mr.route(session(text="x", image=True), cfg(visual_recall=False))
        self.assertFalse(p.do_visual_recall)

    def test_visual_needs_enrichment(self):
        # 视觉召回依赖离线富集,关富集则不走视觉
        p = mr.route(session(text="x", image=True), cfg(enrichment=False))
        self.assertFalse(p.do_visual_recall)

    def test_verifier_off(self):
        p = mr.route(session(text="x"), cfg(verifier="off"))
        self.assertFalse(p.do_verify)

    def test_to_dict(self):
        d = mr.route(session(text="x"), cfg()).to_dict()
        self.assertIn("modality", d)
        self.assertIn("do_visual_recall", d)


if __name__ == "__main__":
    unittest.main()
