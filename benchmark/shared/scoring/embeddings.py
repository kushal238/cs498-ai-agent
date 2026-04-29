"""
Shared embedding utilities for semantic similarity scoring.

Used by concept_f1.py and ndcg.py to match clinical condition names that are
phrased differently but mean the same thing (e.g. "type 2 DM" vs
"Type 2 diabetes mellitus").

Model selection
---------------
Default: pritamdeka/S-PubMedBert-MS-MARCO
    A PubMed-fine-tuned sentence encoder that understands biomedical
    abbreviations and terminology better than general-purpose models.

Override via env var:
    SCORING_EMBED_MODEL=all-MiniLM-L6-v2  (faster, less domain-aware)

The model is loaded lazily on first use and cached for the process lifetime,
so it is only downloaded once per benchmark run.
"""

from __future__ import annotations

import functools
import os

import numpy as np

_DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# Cosine similarity threshold above which two condition strings are considered
# a semantic match.  0.82 was chosen empirically for pritamdeka/S-PubMedBert-MS-MARCO:
# unrelated medical terms cluster at 0.75–0.80, true paraphrases at 0.95–0.98,
# so 0.82 sits cleanly in the gap.  Adjust via SCORING_EMBED_THRESHOLD if switching
# to a different model (e.g. all-MiniLM-L6-v2 may need a different value).
CONDITION_SIMILARITY_THRESHOLD = float(
    os.environ.get("SCORING_EMBED_THRESHOLD", "0.82")
)


@functools.lru_cache(maxsize=1)
def _get_model():
    """Load and cache the SentenceTransformer model (lazy, process-lifetime)."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding-based scoring. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model_name = os.environ.get("SCORING_EMBED_MODEL", _DEFAULT_MODEL)
    return SentenceTransformer(model_name)


@functools.lru_cache(maxsize=512)
def _embed(text: str) -> np.ndarray:
    """Return a cached unit-norm embedding for a single text string.

    lru_cache requires hashable *arguments* (text is a str, so fine) but
    imposes no constraint on the return value — numpy arrays work.  The
    returned array is marked read-only to prevent accidental mutation of the
    cached object across calls.
    """
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    vec.flags.writeable = False
    return vec


def cosine_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between the embeddings of two strings.

    Both embeddings are L2-normalised so cosine similarity reduces to a
    simple dot product.

    Returns:
        float in [-1.0, 1.0]; in practice always in [0.0, 1.0] for these models.
    """
    return float(np.dot(_embed(a), _embed(b)))


def best_match_similarity(query: str, candidates: list[str]) -> float:
    """Return the highest cosine similarity between query and any candidate string."""
    if not candidates:
        return 0.0
    return max(cosine_similarity(query, c) for c in candidates)
