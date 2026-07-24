"""Image-to-product retrieval using vectors plus enriched visual attributes."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

from ..llm.vision import describe_shopping_image
from ..models import PreferenceProfile


EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
MAX_EMBEDDING_IMAGE_BYTES = 3 * 1024 * 1024
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VISUAL_STOPWORDS = {
    "a", "an", "and", "for", "from", "image", "in", "is", "of", "on",
    "or", "product", "reference", "the", "this", "to", "use", "with",
}


def embed_image(
    image_bytes: bytes,
    *,
    mime_type: str = "image/jpeg",
    timeout: float = 45.0,
) -> list[float]:
    """Generate a DashScope multimodal-embedding-v1 vector for one image."""
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    if not image_bytes:
        raise ValueError("image is empty")
    if len(image_bytes) > MAX_EMBEDDING_IMAGE_BYTES:
        raise ValueError("image exceeds the 3 MB multimodal-embedding-v1 limit")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime_type or 'image/jpeg'};base64,{encoded}"
    payload = {
        "model": (
            os.environ.get("DASHSCOPE_VISUAL_EMBEDDING_MODEL")
            or "multimodal-embedding-v1"
        ).strip(),
        "input": {"contents": [{"image": data_uri}]},
    }
    request = urllib.request.Request(
        EMBEDDING_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"DashScope visual embedding failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DashScope visual embedding failed: {exc}") from exc

    embeddings = ((data.get("output") or {}).get("embeddings") or [])
    if not embeddings or not isinstance(embeddings[0].get("embedding"), list):
        raise RuntimeError("DashScope visual embedding response has no vector")
    vector = [float(value) for value in embeddings[0]["embedding"]]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("DashScope returned an invalid visual embedding")
    return vector


def retrieve_visual(
    profile: PreferenceProfile,
    catalog: Iterable[dict[str, Any]],
    image_refs: Iterable[dict[str, str]],
    *,
    limit: int = 80,
    embedder: Callable[..., list[float]] = embed_image,
    analyzer: Callable[..., dict] = describe_shopping_image,
) -> list[dict[str, Any]]:
    """Return image-similar candidates without accepting/rejecting products.

    Stored vectors are compared only when their dimensions match the query
    vector. Enriched visual attributes provide coverage for products whose
    vectors came from a different model or fallback encoder.
    """
    products = list(catalog)
    ref = _latest_readable_ref(image_refs)
    if not products or ref is None or limit <= 0:
        return []
    image_bytes, mime_type, filename = ref

    query_vector: list[float] = []
    analysis: dict[str, Any] = {}
    embedding_error = ""
    analysis_error = ""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="visual-retrieval") as pool:
        vector_future = pool.submit(embedder, image_bytes, mime_type=mime_type)
        analysis_future = pool.submit(
            analyzer,
            image_bytes,
            mime_type=mime_type,
            filename=filename,
            user_text=profile.raw_query,
        )
        try:
            query_vector = vector_future.result()
        except Exception as exc:  # noqa: BLE001 - attribute retrieval remains available
            embedding_error = str(exc)[:180]
        try:
            result = analysis_future.result()
            if isinstance(result, dict):
                analysis = result
        except Exception as exc:  # noqa: BLE001 - vector retrieval remains available
            analysis_error = str(exc)[:180]

    visual_query = _visual_query(analysis, profile)
    query_tokens = list(dict.fromkeys(_tokens(visual_query)))
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for item in products:
        stored_vector = decode_embedding(item.get("image_embedding"))
        vector_score: float | None = None
        if query_vector and len(stored_vector) == len(query_vector):
            vector_score = max(0.0, min(1.0, cosine_similarity(query_vector, stored_vector)))

        attribute_score = _attribute_similarity(query_tokens, item)
        if vector_score is not None and attribute_score > 0:
            score = 0.75 * vector_score + 0.25 * attribute_score
            method = "vector+attributes"
        elif vector_score is not None:
            score = vector_score
            method = "vector"
        elif attribute_score > 0:
            score = attribute_score
            method = "attributes"
        else:
            continue

        metadata: dict[str, Any] = {
            "method": method,
            "query_vector_dimensions": len(query_vector),
            "catalog_vector_dimensions": len(stored_vector),
            "attribute_similarity": round(attribute_score, 6),
        }
        if vector_score is not None:
            metadata["vector_similarity"] = round(vector_score, 6)
        if embedding_error:
            metadata["embedding_warning"] = embedding_error
        if analysis_error:
            metadata["analysis_warning"] = analysis_error
        scored.append((max(0.0, min(1.0, score)), item, metadata))

    scored.sort(
        key=lambda value: (value[0], _number(value[1].get("rating")), _number(value[1].get("rating_number"))),
        reverse=True,
    )
    return [
        _with_visual_score(item, score, metadata, query_tokens)
        for score, item, metadata in scored[:limit]
    ]


def decode_embedding(value: Any) -> list[float]:
    """Decode the catalog's Base64 little-endian float32 representation."""
    if isinstance(value, (list, tuple)):
        try:
            vector = [float(item) for item in value]
        except (TypeError, ValueError):
            return []
        return vector if all(math.isfinite(item) for item in vector) else []
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        raw = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError, TypeError):
        return []
    if not raw or len(raw) % 4:
        return []
    try:
        vector = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    except struct.error:
        return []
    return vector if all(math.isfinite(item) for item in vector) else []


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _latest_readable_ref(
    refs: Iterable[dict[str, str]],
) -> tuple[bytes, str, str] | None:
    for ref in reversed(list(refs)):
        path = str(ref.get("path") or "")
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as handle:
                image_bytes = handle.read(5 * 1024 * 1024 + 1)
        except OSError:
            continue
        if not image_bytes or len(image_bytes) > 5 * 1024 * 1024:
            continue
        return (
            image_bytes,
            str(ref.get("mime_type") or "image/jpeg"),
            str(ref.get("filename") or os.path.basename(path) or "reference.jpg"),
        )
    return None


def _visual_query(analysis: dict[str, Any], profile: PreferenceProfile) -> str:
    values: list[Any] = []
    if analysis.get("provider") != "fallback":
        values.extend(
            analysis.get(key)
            for key in (
                "product_category", "visual_preferences", "soft_preferences",
                "hard_constraints", "search_keywords", "summary",
            )
        )
    values.extend([profile.visual_context, profile.category, profile.search_keywords])
    return _flatten(values)


def _attribute_similarity(query_tokens: list[str], item: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    weighted_fields = [
        (1.00, set(_tokens(_flatten(item.get("visual_attrs"))))),
        (0.85, set(_tokens(_flatten(item.get("enriched_text"))))),
        (0.60, set(_tokens(_flatten(item.get("name"))))),
    ]
    matched = 0.0
    for token in query_tokens:
        matched += max(
            (weight for weight, field_tokens in weighted_fields if token in field_tokens),
            default=0.0,
        )
    return max(0.0, min(1.0, matched / len(query_tokens)))


def _with_visual_score(
    item: dict[str, Any],
    score: float,
    metadata: dict[str, Any],
    query_tokens: list[str],
) -> dict[str, Any]:
    candidate = dict(item)
    candidate.pop("image_embedding", None)
    candidate.pop("review_embedding", None)
    candidate["visual_similarity"] = round(score, 6)
    candidate["retrieval"] = {
        "sources": ["visual"],
        "visual_similarity": candidate["visual_similarity"],
        "visual_query_tokens": query_tokens[:16],
        **metadata,
    }
    return candidate


def _tokens(value: str) -> list[str]:
    return [
        token for token in _TOKEN_RE.findall((value or "").lower())
        if token not in _VISUAL_STOPWORDS
    ]


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return _flatten(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                pass
        return text
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
