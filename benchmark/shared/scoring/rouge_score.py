"""
ROUGE scoring for text-generation benchmark stages.

Used for:
    Stage 1 — transcription_cleaned   (vs ground truth cleaned transcript)
    Stage 2 — clinical_summary        (vs ground truth summary)
    Stage 6 — final_report fields     (vs ground truth SOAP sections)

Depends on the 'rouge-score' package (Google Research implementation).
Install: pip install rouge-score
"""

from __future__ import annotations

from rouge_score import rouge_scorer  # noqa: F401  (imported to validate install)


def score_rouge(
    hypothesis: str,
    reference: str,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute ROUGE scores between a generated text and a reference string.

    Args:
        hypothesis: The model-generated text to evaluate.
        reference:  The ground-truth reference string.
        metrics:    ROUGE variants to compute. Defaults to ["rouge1", "rouge2", "rougeL"].

    Returns:
        Dict mapping each metric name to its F1 score, e.g.
        {"rouge1": 0.72, "rouge2": 0.61, "rougeL": 0.68}.

    Raises:
        NotImplementedError: Always — scoring logic not yet implemented.

    TODO: Instantiate rouge_scorer.RougeScorer(metrics, use_stemmer=True),
          call .score(reference, hypothesis), and return {m: scores[m].fmeasure ...}.
    """
    if metrics is None:
        metrics = ["rouge1", "rouge2", "rougeL"]
    raise NotImplementedError("TODO: implement ROUGE scoring with rouge_scorer.RougeScorer")


def score_stage_text(stage_name: str, hypothesis: str, reference: str) -> dict:
    """Score a single text-generation stage and return a labeled result dict.

    Args:
        stage_name: Human-readable stage label, e.g. "transcription_cleaned".
        hypothesis: Model output for this stage.
        reference:  Ground-truth string for this stage.

    Returns:
        Dict with keys "stage", "rouge1", "rouge2", "rougeL".

    Raises:
        NotImplementedError: Always — delegates to score_rouge which is not yet implemented.

    TODO: Call score_rouge() and merge the result with {"stage": stage_name}.
    """
    raise NotImplementedError("TODO: implement stage-level ROUGE wrapper")
