"""
BERTScore (clinical) scoring for free-text benchmark stages.

Used as a supplementary semantic-similarity metric alongside ROUGE-L for:
    Stage 1 -- transcription_cleaned
    Stage 2 -- clinical_summary
    Stage 6 -- final_report SOAP sections (subjective/objective/assessment/plan)

Encoder: emilyalsentzer/Bio_ClinicalBERT (clinical-domain BERT trained on MIMIC).
This addresses the limitation that lexical ROUGE-L penalizes valid clinical
paraphrases; BERTScore with a clinical encoder credits semantic equivalence.

Override via env var:
    BERTSCORE_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    BERTSCORE_LAYER=9  (default chosen empirically per the BERTScore docs)
"""
from __future__ import annotations

import functools
import os
from typing import Optional

_DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
_DEFAULT_LAYER = 9
_DEFAULT_BATCH = 32

# Bio_ClinicalBERT has a 512-token context. Leave headroom for [CLS]/[SEP]
# and any tokenizer expansion (BERT WordPiece can split clinical terms into
# many subwords). 480 tokens after truncation has been safe in practice.
_MAX_TOKENS = 480

# Resolved at first use, then cached.
_MODEL_NAME = os.environ.get("BERTSCORE_MODEL", _DEFAULT_MODEL)
_LAYER = int(os.environ.get("BERTSCORE_LAYER", str(_DEFAULT_LAYER)))


@functools.lru_cache(maxsize=1)
def _get_scorer():
    """Lazily import bert_score and instantiate a reusable BERTScorer."""
    from bert_score import BERTScorer
    return BERTScorer(
        model_type=_MODEL_NAME,
        num_layers=_LAYER,
        lang="en",
        rescale_with_baseline=False,
        idf=False,
        batch_size=_DEFAULT_BATCH,
    )


@functools.lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_MODEL_NAME)


def _truncate(text: str) -> str:
    """Truncate text to fit Bio_ClinicalBERT's 512-token context.

    Same encoder is used for hypothesis and reference, so per-string
    truncation does not bias one system over another. Long-form stages
    (full transcript, full SOAP report) score the first ~480 tokens of
    both hypothesis and reference, which captures the clinically dense
    opening of each section in practice.
    """
    if not text:
        return text
    tokenizer = _get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= _MAX_TOKENS:
        return text
    truncated = tokenizer.decode(ids[:_MAX_TOKENS], skip_special_tokens=True)
    return truncated


def score_bertscore(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute clinical BERTScore P/R/F1 between hypothesis and reference.

    Empty hypothesis or reference yields {"bertscore_f1": 0.0, ...}, matching
    the convention used by score_rouge for a clean "missing output" signal.

    Returns:
        {"bertscore_p": float, "bertscore_r": float, "bertscore_f1": float}
    """
    if not hypothesis or not reference:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    hypothesis = _truncate(hypothesis)
    reference = _truncate(reference)
    scorer = _get_scorer()
    P, R, F = scorer.score([hypothesis], [reference])
    return {
        "bertscore_p": float(P[0].item()),
        "bertscore_r": float(R[0].item()),
        "bertscore_f1": float(F[0].item()),
    }


def score_bertscore_batch(
    hypotheses: list[str],
    references: list[str],
) -> list[dict[str, float]]:
    """Batched variant. Avoids repeated model-load overhead per call.

    Pairs with empty hypothesis or reference are scored as zeros without
    being sent through the encoder.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    keep_idx: list[int] = []
    keep_hyps: list[str] = []
    keep_refs: list[str] = []
    for i, (h, r) in enumerate(zip(hypotheses, references)):
        if h and r:
            keep_idx.append(i)
            keep_hyps.append(_truncate(h))
            keep_refs.append(_truncate(r))

    out = [{"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
           for _ in hypotheses]
    if not keep_idx:
        return out

    scorer = _get_scorer()
    P, R, F = scorer.score(keep_hyps, keep_refs)
    for j, i in enumerate(keep_idx):
        out[i] = {
            "bertscore_p": float(P[j].item()),
            "bertscore_r": float(R[j].item()),
            "bertscore_f1": float(F[j].item()),
        }
    return out
