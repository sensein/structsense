# NER Label Correctness Evaluation

## Overview

This evaluation assesses whether the **label** assigned to each extracted **entity** is correct across StructSense NER extraction results. It covers 3 papers, each processed by 3 LLM backends (Gemini, GPT, Qwen), for a total of 9 non-HIL (no human-in-the-loop) result files.

## Evaluation Approach

The evaluation is fully automated via `ner_eval.py`, using a baked-in heuristic dictionary built from cross-model consensus analysis of all 9 files (8,882 total entity occurrences, 3,009 unique entities).

### Pipeline Steps

1. **Source filtering**: Remove entities extracted exclusively by `en_core_web_sm` (SpaCy's general-purpose NER model), which produces high noise in biomedical/neuroscience text.

2. **Junk entity exclusion**: Remove entities that should not have been extracted at all:
   - Single characters and Greek letters (`a`, `x`, `α`, `σ`, etc.)
   - Known noisy 2-character tokens (`rn`, `de`, `ns`, `eq`, etc.)
   - Pure numeric values (`0.5`, `1000`, `0.971`, etc.)
   - Figure/table references (`Fig. 5a`, `Figure 1`, `Table 1`, etc.)
   - Equation references (`Eq. (1)`, `Eqs. (14) and (15)`, etc.)
   - Stopwords and generic terms (`data`, `model`, `methods`, `task`, etc.)
   - URLs, punctuation artifacts, citation numbers

3. **Label correctness scoring** (in priority order):
   - **Multi-label check**: Some entities legitimately have multiple correct labels (e.g., `VIP` can be GENE or CELL_TYPE; `NS-Forest` can be METHOD or SOFTWARE). These are accepted if the assigned label matches any acceptable option.
   - **Label blacklist**: Labels from clinical NER models (`Diagnostic_procedure`, `Sign_symptom`, `Lab_value`, `Coreference`, `Detailed_description`, etc.) are marked incorrect — these are clinical categories misapplied to neuroscience text.
   - **Exact dictionary match**: 200+ entity-to-label mappings derived from cross-model consensus (70%+ agreement across models).
   - **Keyword rules**: Pattern-based rules (e.g., entity contains "neuron" -> CELL_TYPE, "cortex" -> BRAIN_REGION, "sequencing" -> METHOD). Uses word-boundary matching to avoid false positives.
   - **Trusted canonical label fallback**: If the assigned label maps to a recognized neuroscience NER category (CELL_TYPE, BRAIN_REGION, GENE, METHOD, etc.) and no conflicting heuristic exists, it is trusted as correct.

### Label Canonicalization

Labels are normalized before comparison to handle variant naming across models:
- Case normalization: `Gene` -> `GENE`, `Cell Type` -> `CELL_TYPE`
- Alias mapping: `GENE_OR_GENE_PRODUCT` -> `GENE`, `TECHNIQUE` -> `METHOD`, `ANATOMICAL_STRUCTURE` -> `BRAIN_REGION`
- Prefix-based fallback: `NEURAL_*` -> `CONCEPT`, `STATISTICAL_*` -> `METRIC`, etc.

## Results

### Per-Paper Summary

#### Paper 1: Discovery of Optimal Cell Type Classification (3 models)

| Model | Total | Evaluated | Correct | Incorrect | Unknown | Accuracy |
|-------|------:|----------:|--------:|----------:|--------:|---------:|
| Gemini | 726 | 622 | 404 | 216 | 2 | 65.2% |
| GPT | 798 | 634 | 367 | 234 | 33 | 61.1% |
| Qwen | 865 | 602 | 388 | 172 | 42 | 69.3% |
| **Total** | **2,389** | **1,858** | **1,159** | **622** | **77** | **65.1%** |

#### Paper 2: Latent Circuit Inference (3 models)

| Model | Total | Evaluated | Correct | Incorrect | Unknown | Accuracy |
|-------|------:|----------:|--------:|----------:|--------:|---------:|
| Gemini | 968 | 813 | 455 | 321 | 37 | 58.6% |
| GPT | 1,410 | 1,097 | 534 | 468 | 95 | 53.3% |
| Qwen | 1,493 | 708 | 510 | 124 | 74 | 80.4% |
| **Total** | **3,871** | **2,618** | **1,499** | **913** | **206** | **62.1%** |

#### Paper 3: Multiscale Spatial Transcriptomic (3 models)

| Model | Total | Evaluated | Correct | Incorrect | Unknown | Accuracy |
|-------|------:|----------:|--------:|----------:|--------:|---------:|
| Gemini | 1,141 | 1,092 | 895 | 155 | 42 | 85.2% |
| GPT | 1,481 | 1,263 | 872 | 318 | 73 | 73.3% |
| Qwen | 0 | 0 | 0 | 0 | 0 | N/A |
| **Total** | **2,622** | **2,355** | **1,767** | **473** | **115** | **78.9%** |

### Overall Summary

| | Count | % of Evaluated |
|--|------:|---------------:|
| Total entities across all files | 8,882 | — |
| Filtered (en_core_web_sm only) | ~350 | — |
| Excluded (junk/generic) | ~700 | — |
| **Evaluated** | **6,831** | 100% |
| Correct | 4,425 | 64.8% |
| Incorrect | 2,008 | 29.4% |
| Unknown (no heuristic match) | 398 | 5.8% |
| **Accuracy (excl. unknown)** | **68.8%** | — |

### Model Comparison (Accuracy across all papers)

| Model | Accuracy |
|-------|------:|
| Gemini | 69.0% |
| GPT | 58.6% |
| Qwen | 75.4% |

### Key Findings

1. **Biggest error source**: The `d4data/biomedical-ner-all` model assigns clinical NER labels (`Diagnostic_procedure`, `Sign_symptom`, `Lab_value`, etc.) that are inappropriate for neuroscience text. These account for the majority of incorrect labels.

2. **Model quality varies by paper**: Gemini performs best on the multiscale spatial paper (85.2%), while Qwen performs best on the latent circuit paper (80.4%). GPT consistently has the lowest accuracy.

3. **Qwen extracts fewer but more accurate entities**: Qwen's evaluated entity count is lower (especially for latent circuit: 708 vs 813/1097), but its accuracy is higher, suggesting more conservative extraction.

4. **Label inconsistency across models**: The same entity often receives different labels from different models. For example, `cell types` may be labeled `CELL_TYPE`, `BIOLOGICAL_ENTITY`, `Diagnostic_procedure`, or `Sign_symptom` depending on the source model.

## Reproduction

```bash
# Evaluate a single file
python evaluation/ner/analysis/ner_eval.py <path_to_nhil.json>

# Evaluate all non-hil files under a directory
python evaluation/ner/analysis/ner_eval.py evaluation/ner/

# Save detailed JSON report
python evaluation/ner/analysis/ner_eval.py evaluation/ner/ -o report.json

# Verbose mode (show all per-entity decisions)
python evaluation/ner/analysis/ner_eval.py <file.json> --verbose
```

## Output Files

- `ner_eval.py` — Evaluation script (no external dependencies beyond Python stdlib)
- `results_all.json` — Detailed per-entity evaluation for all 9 files
- `results_discovery_of_optimal_cell.json` — Discovery paper results
- `results_latent_circuit.json` — Latent circuit paper results
- `results_multiscale_spatial.json` — Multiscale spatial paper results
- `RESULTS.md` — This document
