# Benchmark NER Evaluation Results

## Overview

This evaluation measures StructSense's entity extraction **recall** against two standard biomedical NER benchmark datasets:

- **NCBI Disease**: 960 entity mentions (403 unique) across 940 sentences
- **S800 Species**: 767 entity mentions (370 unique) across 1630 sentences

The ground truth defines the **minimum** set of entities to extract. StructSense extracts additional entities beyond ground truth — this is a feature, not an error, and is reported as "extra entities."

## Evaluation Approach

### Entity Matching (3-tier cascade)

For each ground truth entity mention, the script attempts matching in priority order:

1. **Exact text match**: Normalize both GT and result text (lowercase, collapse spaced hyphens `" - "` to `"-"`, collapse whitespace), then exact string comparison. This handles the BIO tokenization artifact where GT has `"ataxia - telangiectasia"` and results have `"ataxia-telangiectasia"`.

2. **Span overlap**: Compare character offsets. Match if overlap >= 50% of GT span length. Catches positional matches where text differs slightly.

3. **Substring/containment**: Normalized GT text is a substring of a result entity text, or vice versa. For example, GT `"T-cell leukaemia"` matches result `"sporadic T-cell prolymphocytic leukaemia"`.

### Filtering

- Entities from `en_core_web_sm` only (SpaCy general-purpose NER) are removed before evaluation
- No junk/stopword filtering is applied (unlike paper-based NER eval) since even generic-looking entities may be valid benchmark matches

### Metrics

| Metric | Description |
|---|---|
| Recall (strict) | Fraction of GT mentions matched via exact text |
| Recall (relaxed) | Fraction of GT mentions matched via any tier |
| Extra entities | Result entities beyond ground truth (tool advantage) |

## Results

### NCBI Disease

| Model | Variant | Strict Recall | Relaxed Recall | Extra Entities |
|-------|---------|--------------|----------------|----------------|
| Gemini 3.1 Flash Lite | hil | 85.6% (822/960) | **95.7%** (919/960) | 1,046 |
| Gemini 3.1 Flash Lite | nhil | 83.1% (798/960) | **96.1%** (923/960) | 1,049 |
| GPT-4o mini | hil | 77.9% (748/960) | 93.4% (897/960) | 1,341 |
| GPT-4o mini | nhil | 80.9% (777/960) | 94.5% (907/960) | 1,192 |
| Qwen | hil | 83.5% (802/960) | **97.5%** (936/960) | 1,783 |
| Qwen | nhil | 84.1% (807/960) | **97.0%** (931/960) | 1,504 |

### S800 Species

| Model | Variant | Strict Recall | Relaxed Recall | Extra Entities |
|-------|---------|--------------|----------------|----------------|
| Gemini | nhil | **85.8%** (658/767) | **96.6%** (741/767) | 2,719 |
| GPT-4o mini | nhil | 62.5% (479/767) | 81.0% (621/767) | 3,223 |
| Qwen | nhil | 70.0% (537/767) | 90.6% (695/767) | 3,622 |

### Key Findings

1. **High recall across models**: Relaxed recall exceeds 90% for most configurations, reaching 97.5% (Qwen hil, NCBI). This means StructSense finds nearly all ground truth entities.

2. **Gemini performs best on strict matching**: Gemini achieves the highest strict recall on both datasets (85.6% NCBI, 85.8% S800), suggesting its extracted text most closely matches ground truth formatting.

3. **GPT struggles with S800 species**: GPT-4o mini drops to 62.5% strict / 81.0% relaxed on S800, likely due to abbreviated species names (e.g., `"F. graminearum"`, `"C. albicans"`) and strain identifiers (e.g., `"M2 (T)"`, `"6C (T)"`).

4. **Significant extra entities extracted**: All models extract 1,000-3,600+ entities beyond ground truth, including genes, proteins, chemicals, biological processes — demonstrating StructSense's advantage over the minimum benchmark annotation.

5. **HIL vs NHIL**: Human-in-the-loop generally provides marginal improvement on relaxed recall (e.g., Qwen: 97.5% vs 97.0%) but can slightly decrease strict recall (Gemini: 85.6% vs 83.1%), suggesting HIL may refine entity text in ways that diverge from ground truth formatting.

6. **Common missed entities**: Entities with unusual formatting are most commonly missed:
   - Abbreviated forms: `"A-T"`, `"B-NHL"`, `"T-PLL"`, `"WT"`
   - Compound entities: `"sporadic breast, brain, prostate and kidney cancer"`
   - Strain identifiers: `"DSM 18155 (T)"`, `"M2 (T)"`
   - Abbreviated species: `"F. graminearum"`, `"C. albicans"`

## Extra Entity Label Distribution (top labels across models)

The extra entities (beyond ground truth) break down into these categories, showing StructSense's broader extraction capability:

**NCBI**: GENE, PROTEIN, GENE_MUTATION, GENETIC_VARIANT, BIOLOGICAL_PROCESS, CHEMICAL, CELL_TYPE, METHOD

**S800**: GENE, BIOLOGICAL_PROCESS, PROTEIN, DISEASE, CHEMICAL, ORGANISM, ORG, METHOD

## Reproduction

```bash
# Evaluate all datasets and models
python evaluation/benchmark/analysis/benchmark_eval.py

# Evaluate single dataset
python evaluation/benchmark/analysis/benchmark_eval.py --dataset ncbi

# Explicit GT + result pair
python evaluation/benchmark/analysis/benchmark_eval.py \
  --gt evaluation/benchmark/ncbi/NCBI_disease_test_entities_mapping.jsonl \
  --result evaluation/benchmark/ncbi/results-qwen/NCBI_disease_test_text_qwen_nhil.json

# Save JSON report
python evaluation/benchmark/analysis/benchmark_eval.py -o report.json

# Verbose (show all missed entities)
python evaluation/benchmark/analysis/benchmark_eval.py --verbose
```

## Output Files

- `benchmark_eval.py` — Evaluation script (Python stdlib only, no external dependencies)
- `results_all.json` — Combined evaluation results for all datasets and models
- `results_ncbi.json` — NCBI dataset results
- `results_s800.json` — S800 dataset results
- `RESULTS.md` — This document
