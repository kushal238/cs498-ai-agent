"""Quick check: BERTScore handles inputs longer than 512 tokens."""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from shared.scoring.bertscore import score_bertscore  # noqa: E402

# ~600 tokens of clinical-flavored text
LONG = ("The patient is a 67-year-old male with a long-standing history of "
        "type 2 diabetes mellitus, hypertension, and stage 3 chronic kidney "
        "disease who presents to the emergency department with worsening "
        "shortness of breath, bilateral lower-extremity edema, and a 4-kilogram "
        "weight gain over the past two weeks. ") * 10

result = score_bertscore(LONG, LONG + " The exam is otherwise unremarkable.")
print("Long-input BERTScore:", result)

assert result["bertscore_f1"] > 0.0, "bertscore should be non-zero on identical-ish long inputs"
print("OK")
