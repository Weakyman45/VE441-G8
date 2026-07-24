from __future__ import annotations

import base64
import os
import struct
import tempfile
import unittest

from engine.models import PreferenceProfile
from engine.retrieval.merge import merge_candidates
from engine.retrieval.text_retrieval import retrieve_text
from engine.retrieval.visual_retrieval import retrieve_visual


def _encoded_vector(values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}f", *values)
    return base64.b64encode(raw).decode("ascii")


class TextRetrievalTests(unittest.TestCase):
    def test_weighted_text_retrieval_prefers_name_and_hides_embeddings(self):
        profile = PreferenceProfile(search_keywords=["charcoal cleanser"])
        catalog = [
            {
                "id": "name-match",
                "name": "Charcoal Cleanser",
                "enriched_text": "skin care bottle",
                "image_embedding": "secret-vector-payload",
                "rating": 4.0,
                "rating_number": 10,
            },
            {
                "id": "enriched-match",
                "name": "Daily Face Wash",
                "enriched_text": "charcoal cleanser for skin",
                "rating": 5.0,
                "rating_number": 500,
            },
            {
                "id": "unrelated",
                "name": "Running Shoes",
                "rating": 5.0,
                "rating_number": 1000,
            },
            {
                "id": "one-token-distractor",
                "name": "Charcoal Coffee Filter",
                "rating": 5.0,
                "rating_number": 2000,
            },
        ]

        results = retrieve_text(profile, catalog)

        self.assertEqual([item["id"] for item in results], ["name-match", "enriched-match"])
        self.assertGreater(results[0]["text_similarity"], results[1]["text_similarity"])
        self.assertNotIn("image_embedding", results[0])
        self.assertEqual(results[0]["retrieval"]["sources"], ["text"])


class VisualRetrievalTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        handle.write(b"small-test-image")
        handle.close()
        self.image_path = handle.name
        self.refs = [{
            "path": self.image_path,
            "mime_type": "image/jpeg",
            "filename": "reference.jpg",
        }]

    def tearDown(self):
        os.unlink(self.image_path)

    def test_vector_similarity_uses_only_equal_dimensions(self):
        catalog = [
            {"id": "same", "name": "A", "image_embedding": _encoded_vector([1.0, 0.0])},
            {"id": "opposite", "name": "B", "image_embedding": _encoded_vector([0.0, 1.0])},
            {"id": "mismatch", "name": "C", "image_embedding": _encoded_vector([1.0, 0.0, 0.0])},
        ]

        results = retrieve_visual(
            PreferenceProfile(),
            catalog,
            self.refs,
            embedder=lambda _raw, **_kwargs: [1.0, 0.0],
            analyzer=lambda _raw, **_kwargs: {"provider": "fallback"},
        )

        self.assertEqual(results[0]["id"], "same")
        self.assertEqual(results[0]["visual_similarity"], 1.0)
        self.assertNotIn("mismatch", [item["id"] for item in results])
        self.assertNotIn("image_embedding", results[0])

    def test_visual_attributes_cover_incompatible_or_unavailable_vectors(self):
        catalog = [
            {
                "id": "red-shoe",
                "name": "Athletic Footwear",
                "visual_attrs": {"visual": {"product_category": "shoe", "colors": ["red"]}},
                "image_embedding": _encoded_vector([1.0, 2.0, 3.0]),
            },
            {
                "id": "blue-hat",
                "name": "Blue Hat",
                "visual_attrs": {"visual": {"product_category": "hat", "colors": ["blue"]}},
                "image_embedding": _encoded_vector([1.0, 2.0, 3.0]),
            },
        ]

        results = retrieve_visual(
            PreferenceProfile(),
            catalog,
            self.refs,
            embedder=lambda _raw, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
            analyzer=lambda _raw, **_kwargs: {
                "provider": "qwen-vl",
                "product_category": "shoe",
                "visual_preferences": ["red"],
                "search_keywords": ["red shoe"],
            },
        )

        self.assertEqual([item["id"] for item in results], ["red-shoe"])
        self.assertEqual(results[0]["retrieval"]["method"], "attributes")
        self.assertIn("embedding_warning", results[0]["retrieval"])


class MergeCandidatesTests(unittest.TestCase):
    def test_merge_unions_sources_and_deduplicates_product(self):
        text = [{
            "id": "p1",
            "name": "Product One",
            "text_similarity": 0.8,
            "image_embedding": "must-not-leak",
            "retrieval": {"sources": ["text"], "text_similarity": 0.8},
        }]
        visual = [
            {
                "id": "p1",
                "name": "Product One",
                "visual_similarity": 0.9,
                "retrieval": {"sources": ["visual"], "method": "vector"},
            },
            {
                "id": "p2",
                "name": "Product Two",
                "visual_similarity": 0.7,
                "retrieval": {"sources": ["visual"], "method": "attributes"},
            },
        ]

        merged = merge_candidates(text, visual)

        self.assertEqual([item["id"] for item in merged], ["p1", "p2"])
        self.assertEqual(merged[0]["retrieval"]["sources"], ["text", "visual"])
        self.assertEqual(merged[0]["text_similarity"], 0.8)
        self.assertEqual(merged[0]["visual_similarity"], 0.9)
        self.assertNotIn("image_embedding", merged[0])
        self.assertNotIn("text_similarity", merged[1]["retrieval"])


if __name__ == "__main__":
    unittest.main()
