# Final benchmark comparison

Aggregated across all cases (mean of per-case means).

| Stage | Metric | Zero-Shot | No-Tools | Agent (Ours) |
|---|---|---|---|---|
| transcription_cleanup | rougeL | 0.813 ± 0.166 | 0.813 ± 0.167 | 0.812 ± 0.166 |
| transcription_cleanup | bertscore_f1 | 0.915 ± 0.059 | 0.916 ± 0.060 | 0.915 ± 0.059 |
| clinical_summarization | rougeL | 0.412 ± 0.092 | 0.422 ± 0.113 | 0.419 ± 0.084 |
| clinical_summarization | bertscore_f1 | 0.786 ± 0.037 | 0.791 ± 0.053 | 0.787 ± 0.038 |
| differential_diagnosis | f1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.993 ± 0.021 |
| differential_diagnosis | ndcg | 0.952 ± 0.048 | 0.965 ± 0.051 | 0.969 ± 0.039 |
| medication_normalization | f1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| drug_interaction_check | f1 | 0.667 ± 0.471 | 0.607 ± 0.507 | 1.000 ± 0.000 |
| final_report_generation | subjective_rougeL | 0.346 ± 0.071 | 0.377 ± 0.067 | 0.385 ± 0.075 |
| final_report_generation | objective_rougeL | 0.297 ± 0.151 | 0.298 ± 0.098 | 0.339 ± 0.121 |
| final_report_generation | assessment_rougeL | 0.236 ± 0.062 | 0.214 ± 0.025 | 0.211 ± 0.033 |
| final_report_generation | plan_rougeL | 0.225 ± 0.105 | 0.195 ± 0.043 | 0.190 ± 0.048 |
| final_report_generation | subjective_bertscore_f1 | 0.759 ± 0.032 | 0.782 ± 0.036 | 0.789 ± 0.039 |
| final_report_generation | objective_bertscore_f1 | 0.735 ± 0.057 | 0.737 ± 0.037 | 0.748 ± 0.044 |
| final_report_generation | assessment_bertscore_f1 | 0.713 ± 0.027 | 0.710 ± 0.029 | 0.703 ± 0.023 |
| final_report_generation | plan_bertscore_f1 | 0.701 ± 0.037 | 0.693 ± 0.021 | 0.690 ± 0.022 |
