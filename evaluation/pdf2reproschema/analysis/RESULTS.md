# PDF-to-ReproSchema Evaluation Results

## Overview

This evaluation measures how accurately StructSense extracts structured questionnaire data from a PDF into ReproSchema format. The source document is the **Mood and Feelings Questionnaire (MFQ) Long Version** — a 33-item self-report depression measure with a 3-point Likert scale (NOT TRUE / SOMETIMES / TRUE, scored 0/1/2).

## Evaluation Approach

The ground truth is hardcoded from the MFQ PDF. Each result file is scored across 7 dimensions:

| Dimension | Weight | What's Checked |
|-----------|--------|----------------|
| Item Completeness | 20% | All 33 items extracted? |
| Question Text | 25% | Fuzzy text match against PDF (>=85% similarity) |
| Response Options | 15% | Correct values (0,1,2) and labels (NOT TRUE, SOMETIMES, TRUE) per item |
| Scoring | 15% | Per-item scoring maps + computed total formula (sum of all 33 items) |
| Activity Metadata | 10% | Title, description, preamble, citation — keyword presence |
| Item Order | 10% | Items in correct 1-33 sequence |
| Input Type | 5% | All items marked as "radio" input |

## Results

| File | Status | Items | Text | Response | Scoring | Metadata | Order | Overall |
|------|--------|-------|------|----------|---------|----------|-------|---------|
| GPT (nhil) | OK | 100% | 100% | 100% | 100% | 100% | 100% | **100.0%** |
| Gemini (nhil) | OK | 100% | 100% | 100% | 50% | 88% | 100% | **91.2%** |
| Gemini (hil) | OK | 100% | 100% | 100% | 50% | 81% | 100% | **90.6%** |
| Qwen (nhil) | FAIL | — | — | — | — | — | — | **0.0%** |

### Key Findings

1. **GPT achieves perfect extraction**: GPT-4o mini produces a flawless ReproSchema conversion — all 33 items with exact question text, correct response options, per-item scoring maps, computed total formula, and complete activity metadata.

2. **Gemini is near-perfect but misses per-item scoring**: Gemini extracts all items and question text perfectly, but does not include per-item `scoring` maps in the output (only the computed total formula). Activity metadata description is slightly incomplete.

3. **HIL slightly hurts Gemini performance**: Gemini HIL (90.6%) scores slightly lower than NHIL (91.2%) due to a minor reduction in metadata description keyword coverage. This suggests the human feedback step may have edited the description in a way that dropped some keywords.

4. **Qwen fails completely**: Qwen times out after 60 seconds, producing no items at all.

5. **Question text extraction is excellent across models**: Both GPT and Gemini achieve 100% similarity on all 33 question texts, including handling of smart quotes (e.g., `didn't` → `didn\u2019t`).

### Scoring Dimension Detail

- **GPT**: Includes both per-item scoring (`{"NOT TRUE": 0, "SOMETIMES": 1, "TRUE": 2}`) and computed total formula (`Q1 + Q2 + ... + Q33`). Score: 100%.
- **Gemini (nhil/hil)**: Includes computed total formula but no per-item scoring maps. Score: 50%.

### Activity Metadata Detail

All fields checked via keyword presence:
- **prefLabel**: "mood", "feelings", "questionnaire", "long"
- **description**: "feeling", "acting", "recently", "past two weeks"
- **preamble**: "not true", "sometimes", "true", "past two weeks", "feeling or acting"
- **citation**: "angold", "costello", "1987", "duke"

GPT: 100% across all fields. Gemini: preamble and citation perfect, description partially incomplete.

## Reproduction

```bash
# Evaluate all result files
python evaluation/pdf2reproschema/analysis/reproschema_eval.py

# Evaluate single file
python evaluation/pdf2reproschema/analysis/reproschema_eval.py <result.json>

# Verbose (show per-item mismatches)
python evaluation/pdf2reproschema/analysis/reproschema_eval.py --verbose

# Save JSON report
python evaluation/pdf2reproschema/analysis/reproschema_eval.py -o report.json
```

## Output Files

- `reproschema_eval.py` — Evaluation script (Python stdlib only)
- `results_all.json` — Detailed per-dimension evaluation for all result files
- `RESULTS.md` — This document
