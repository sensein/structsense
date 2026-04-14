# Resource Extraction Evaluation Results

## Overview

This evaluation measures how accurately StructSense extracts structured metadata from scientific papers describing computational resources (tools/models). Two papers are evaluated:

- **ViTPose**: A vision transformer-based pose estimation model (Human target)
- **DeepLabCut**: A multi-animal pose estimation and tracking tool (Animal target)

Each paper was processed by 3 LLM backends (Gemini, GPT, Qwen), with some HIL variants.

## Evaluation Approach

Ground truth was built from cross-model consensus and paper content. Each result is scored across 3 dimensions:

### 1. Primary Fields (40% weight)

Checks core resource metadata against expected values:

| Field | What's Checked |
|-------|----------------|
| name | Fuzzy match against known resource name |
| type | Model vs Tool vs Dataset (exact match) |
| category | "Pose Estimation" keyword match |
| target | Human vs Animal (exact match) |
| specific_target | Keyword presence (e.g., "keypoint", "mice") |
| url | Contains expected keywords (e.g., "github", resource name) |

### 2. Mentions Recall (40% weight)

Checks if key related entities were extracted from the paper:

- **Required mentions**: Entities that appear consistently across all model outputs (must-find)
- **Optional mentions**: Entities found by some models (nice-to-have)
- **Extra mentions**: Additional entities found beyond ground truth (informational)

Categories: related_models, datasets, benchmarks, tools

### 3. Ontology Mapping Quality (20% weight)

Flags nonsensical ontology mappings using heuristic detection:
- Pharmaceutical/drug terms mapped to CS resources
- Plant species mapped to dataset names (e.g., "Magnolia coco" for "COCO dataset")
- Genetic disease terms mapped to model names

## Results

### DeepLabCut

| Model | Primary | Mentions | Ontology | Overall |
|-------|--------:|---------:|---------:|--------:|
| Gemini (nhil) | 85% | **100%** | 96% | **93.2%** |
| GPT (nhil) | **100%** | 75% | 94% | 88.9% |
| Qwen (nhil) | 75% | **100%** | **98%** | 89.4% |

### ViTPose

| Model | Primary | Mentions | Ontology | Overall |
|-------|--------:|---------:|---------:|--------:|
| Gemini (hil) | **94%** | 75% | 40% | 75.8% |
| Gemini (nhil) | **94%** | **100%** | 41% | 85.9% |
| GPT (nhil) | **94%** | 75% | **97%** | 87.2% |
| Qwen (nhil) | 72% | 96% | **97%** | 86.6% |

### Key Findings

1. **Primary field extraction is strong across all models**: Name, type, category, and target are correctly identified in nearly all cases. Main failures are Qwen adding extra text to the name ("DeepLabCut for Multi-Animal Pose Estimation") and missing URLs.

2. **Mentions recall varies by model**: Gemini and Qwen tend to extract more complete mention lists (100% required recall on DeepLabCut), while GPT sometimes misses tools like `idtracker.ai` and `mmpose`.

3. **Ontology mapping is the weakest dimension**: The alignment agent uses generic biomedical ontology lookup that produces nonsensical mappings for CS/ML resources:
   - "COCO" dataset → "Magnolia coco" (a plant species in NCBITAXON)
   - Model names → "CALFACTANT 35MG/ML SUSP,INTRATRACHEAL" (a pharmaceutical product)
   - "Human Keypoint Detection" → "Human leucocyte antigen gene detection"

   This is a **systematic issue** with the ontology lookup service, not the extraction itself.

4. **ViTPose Gemini has the worst ontology scores (40-41%)**: The Gemini model's alignment agent consistently maps ML model names to pharmaceutical terms (CALFACTANT), dragging down its overall score despite excellent extraction.

5. **HIL slightly hurts ViTPose Gemini**: The human-in-the-loop variant (75.8%) scores lower than NHIL (85.9%) due to losing the `mmpose` tool mention during feedback processing.

6. **GPT produces the most accurate URLs**: GPT consistently provides the correct GitHub URL (`github.com/ViTAE-Transformer/ViTPose`, `github.com/DeepLabCut/DeepLabCut`), while Gemini/Qwen sometimes use the benchmark URL instead.

### Common Missed Mentions

| Paper | Category | Commonly Missed |
|-------|----------|----------------|
| ViTPose | tools | `mmpose` (missed by GPT, Gemini HIL) |
| ViTPose | models | `PRTR` (missed by Qwen) |
| DeepLabCut | tools | `idtracker.ai` (missed by GPT) |

## Reproduction

```bash
# Evaluate all papers and models
python evaluation/resource_extraction/analysis/resource_eval.py

# Evaluate single file
python evaluation/resource_extraction/analysis/resource_eval.py <result.json>

# Verbose (show missed mentions and ontology issues)
python evaluation/resource_extraction/analysis/resource_eval.py --verbose

# Save JSON report
python evaluation/resource_extraction/analysis/resource_eval.py -o report.json
```

## Output Files

- `resource_eval.py` — Evaluation script (Python stdlib only)
- `results_all.json` — Detailed evaluation for all result files
- `RESULTS.md` — This document
