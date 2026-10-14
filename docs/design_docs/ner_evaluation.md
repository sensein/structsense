# Neuroscientific NER Evaluation: StructSense vs. Direct API Call

## Overview

A comparative evaluation of two neuroscientific entity extraction approaches on the same task: extracting and labeling all neuroscientific entities from neuroscience papers.

| System | Description |
|---|---|
| **StructSense** | Multi-agent pipeline built on CrewAI, with extraction, ontology alignment, quality judging, and optional human-in-the-loop stages |
| **Direct API Call** | Single LiteLLM call to the same underlying model, no orchestration overhead |

---

## System Architecture

### StructSense Call Chain — BioPortal mode (cloud)
```
StructSense → CrewAI → LiteLLM → OpenRouter → LLM provider
                                      ↓
                  alignment agent → BioPortal API
```

### StructSense Call Chain — Fast alignment mode (local)
```
StructSense → CrewAI → LiteLLM → OpenRouter → LLM provider
                                      ↓
                  concept mapping tool (POST /map/batch) → local HTTP service
                                                           (BM25 + dense retrieval)
```

> **Fast alignment bypasses the alignment LLM agent entirely.** It calls the concept mapping tool directly in batch, skipping the LLM reasoning step. No BioPortal API call is made. The local service runs hybrid BM25 + dense retrieval with re-ranking on-premise at `LOCAL_CONCEPT_MAPPING_URL` (default `http://localhost:8000`).

### Direct API Call Chain
```
Direct call → LiteLLM → OpenRouter → LLM provider
```

> **Key insight:** Both systems use LiteLLM as their LLM interface layer. Any observed performance difference is attributable to StructSense's orchestration, chunking, and judging stages — not the underlying model interface.

### StructSense Agent Configuration

StructSense runs two sequential LLM agents, each with a distinct role:

```yaml
extractor_agent:
  # Extracts raw entity spans and labels from input text
  # Output: structured JSON of entity candidates

alignment_agent:
  # Maps extracted entities to ontology terms
  # Both BioPortal and local modes return the same output shape:
  #   { ontology_id: <full IRI>, ontology_label: <label>, ontology: <acronym> }
  # BioPortal mode:  IRI from BioPortal search API  e.g. http://purl.obolibrary.org/obo/MONDO_0007254
  #                  ontology auto-detected per term via BioPortal Recommender API
  #                  throttled to 0.7s per term — source of the ~60 min runtime
  # Local mode:      IRI from local BM25 + dense retrieval service
  #                  e.g. http://purl.obolibrary.org/obo/UBERON_0001870
  # Fast alignment:  alignment_agent LLM is skipped entirely;
  #                  concept mapping tool called directly in batch via POST /map/batch
```

This two-stage design means StructSense's final output is in **ontology space**, while a direct API call produces output in **free-form label space** (LLM-assigned labels, no fixed schema). The exact ontology format depends on which alignment mode is active. See Label Space Problem below.

---

## What to Hold Constant Across Both Systems

| Standardize | Can Differ |
|---|---|
| Input text (same PDF-extracted text) | Prompt wording and structure |
| Output format (same JSON schema for entity spans) | Label space (see Label Space Problem) |
| Evaluation criteria | Number of LLM calls, internal routing |
| Underlying model (same model string via LiteLLM) | Chunking and orchestration strategy |
| Normalization strategy (abbreviation expansion) | Ontology alignment stage |

> Both systems must receive identical input text, produced by the same PDF→text pipeline run (Grobid or pdfplumber) before either system is invoked. Label space is deliberately **not** held constant — the evaluation strategy accounts for this explicitly.

---

## Label Space Problem

The two systems operate in fundamentally different label spaces by design:

| System | Alignment mode | Label space | Example output |
|---|---|---|---|
| StructSense | BioPortal (cloud) | Full IRI + ontology acronym (separate field) | `ontology_id: http://purl.obolibrary.org/obo/MONDO_0007254`, `ontology: MONDO` |
| StructSense | Local / fast alignment | Full OBO IRI + ontology acronym (separate field) | `ontology_id: http://purl.obolibrary.org/obo/UBERON_0001870`, `ontology: UBERON` |
| Direct API call | N/A | Free-form LLM-assigned labels (no fixed schema) | `BRAIN_REGION`, `GENE`, `CELL_TYPE` |

Both StructSense alignment modes return the **same output shape**: `{ontology_id, ontology_label, ontology}`. The `ontology` field always carries the acronym directly — no IRI parsing is needed. The key difference between modes is *which ontologies* are selected and *how*: BioPortal uses a Recommender API to auto-detect ontologies per term (non-deterministic), while the local service uses its own indexed ontology set. Forcing identical labels at extraction time would break StructSense's pipeline — see Evaluation Strategy below.

StructSense's extractor agent also uses its own internal label taxonomy (`MUTATION`, `GENE`, `DISEASE`, etc.). Rather than hand-curating a fixed schema collapse mapping, both systems' labels are normalized at comparison time using `_canonicalize_label()` from `evaluation/ner/analysis/ner_eval.py`. This function resolves aliases and prefix patterns (e.g. `BRAIN_AREA`, `ANATOMY`, `ANATOMICAL_STRUCTURE` all canonicalize to `BRAIN_REGION`) without requiring either system to use a predetermined label set.

---

## Direct API Call — Label Approach

The direct API call (`evaluation/ner_comparison/direct_api.py`) does **not** provide a fixed label list to the model. The prompt instructs the LLM to assign the most precise and descriptive label it thinks is appropriate, with examples (`BRAIN_REGION`, `CELL_TYPE`, `GENE`, `DISEASE`, `METHOD`, `CHEMICAL`, `SPECIES`, `SOFTWARE`) as guidance only.

This means the direct API and StructSense's extractor agent operate in the same unconstrained label space. Label comparison uses canonical normalization — not schema collapse — so no pre-agreed taxonomy is required from either system.

## StructSense Final Output Schema

Each entity in StructSense's output contains the following fields:

```json
{
  "entity":        "missense mutations",   // extracted text span
  "label":         "MUTATION",             // extractor agent label (StructSense taxonomy)
  "start":         14,                     // char offset in chunk
  "end":           32,
  "global_start":  14,                     // char offset in full document
  "global_end":    32,
  "weighted_score": 1.0,                  // multi-model voting score
  "model_count":   1,
  "occurrences":   [...],                  // all span locations with sentence context
  "provenance":    [...],                  // per-source vote weights

  // Entity ontology mapping (maps the entity text)
  "ontology_id":    "http://purl.jp/bio/4/id/...",
  "ontology_label": "missense mutation",
  "ontology":       "IOBC",
  "concept_mapping_provenance": "tool",

  // Label ontology mapping (maps the label type itself)
  "label_ontology_id":    "http://purl.jp/bio/4/id/...",
  "label_ontology_label": "Frameshift Mutation",
  "label_ontology":       "IOBC",
  "label_concept_mapping_provenance": "tool",

  // Judge stage output
  "judge_score": 1.0,
  "remarks":     "Exact span and label match a mutation mention."
}
```

> **The judge stage is already part of StructSense's output.** `judge_score` and `remarks` are produced by the quality judging agent. A separate judge call does not need to be added to the evaluation harness — use `judge_score` and `remarks` directly. The "cancer" example in the real output (judge_score: 0.78, remarks: *"ontology mapping is poor and overly specific/incorrect"*) shows the judge correctly catching alignment failures.

> **Two ontology mappings are present per entity.** `ontology_*` maps the entity text; `label_ontology_*` maps the label type itself. For the evaluation, use `ontology` (entity mapping) as the primary alignment signal. `label_ontology` is supplementary and maps the category concept, not the entity.

---

## Label Canonicalization

Rather than hand-curating fixed schema collapse mappings, label comparison uses `_canonicalize_label()` from `evaluation/ner/analysis/ner_eval.py`. This function normalizes free-form label strings to canonical forms via an alias table and prefix rules, and is shared across both Layer 1A and Layer 1B.

Examples of what canonicalization handles:

| Raw label | Canonical form |
|---|---|
| `BRAIN_AREA`, `ANATOMY`, `ANATOMICAL_STRUCTURE`, `NEUROANATOMY` | `BRAIN_REGION` |
| `GENE_OR_PROTEIN`, `PROTEIN`, `GENETIC_MARKER`, `BIOLOGICAL_MARKER` | `GENE` |
| `TECHNIQUE`, `METHODOLOGY`, `EXPERIMENTAL_TECHNIQUE`, `ALGORITHM` | `METHOD` |
| `CELL`, `CELLTYPE`, `CELL_CLUSTER`, `CELL_SUBCLASS` | `CELL_TYPE` |
| `ORGANISM` | `SPECIES` |

Entity filtering before any comparison uses `is_excluded()` from the same module, which removes noise, stopwords, generic meta-terms, punctuation artifacts, and citation/figure references.

> **Ontology field note:** StructSense's `result["ontology"]` (the acronym, e.g. `UBERON`, `IOBC`, `MONDO`) is not used in the Layer 1 label comparison. Layer 1 compares the extractor agent's `label` field from both systems. The ontology field is relevant for Layer 2 qualitative analysis only (e.g. checking whether the Recommender selected a sensible ontology for a given entity). Always read `result["ontology"]` directly — never parse the IRI.

---

## Test Corpus

- **Size:** 20–30 papers
- **Stratify by:**
  - Domain: cellular/molecular, systems, cognitive, clinical neuro
  - Entity density: sparse (methods papers) vs. dense (results-heavy papers)
  - LLM familiarity: papers likely in training data vs. post-cutoff

---

## Ground Truth Strategy

**Option A — Human annotation (gold standard)**
Manually annotate 5–10 papers. Expensive but enables true precision/recall.

**Option B — LLM-as-judge consensus (recommended for initial eval)**
Run a third judge LLM call treating the union or consensus of both systems as a soft reference. Faster and sufficient for a comparative study.

---

## Evaluation Dimensions

| Dimension | Description |
|---|---|
| Coverage | Does the extractor find all entities? (recall) |
| Precision | Are extracted entities genuinely neuroscientific? |
| Label accuracy | Are entity types assigned correctly? |
| Granularity | "prefrontal cortex" vs. "cortex" — surface form specificity |
| Consistency | Given the same paper twice, is output stable? |
| Hallucination rate | Entities extracted that don't appear in the source text |
| Duplicate rate | Same entity extracted multiple times with minor surface variation |

---

## Evaluation Strategy

The label space mismatch means comparison must happen across two layers, each with sub-steps. Do not collapse these into a single metric.

### Layer 1 — Entity Span and Label Comparison

Layer 1 has two sub-steps that each answer a distinct question.

---

**Layer 1A — Post-extraction, pre-alignment (entity span + label comparison)**

Answers: *do both systems find the same entities, and do they agree on the category?*

Run before the alignment agent. Compare StructSense's extractor output directly against the direct API call. Both systems assign free-form labels; agreement is determined by canonicalizing both labels via `_canonicalize_label()` and checking for equality. Noise entities are filtered by `is_excluded()` before comparison. Spans are normalized (lowercased, abbreviation-expanded) before matching.

```
PDF text
    ↓
StructSense extractor agent                        Direct API call
  entity: "missense mutations"                     entity: "missense mutations"
  label:  MUTATION → _canonicalize_label → GENE    label: gene_protein → _canonicalize_label → GENE
    ↓
  ← Layer 1A: compare normalized spans + canonical labels →
    ↓
StructSense alignment agent (not yet run)
```

Metrics at this checkpoint:
- Jaccard overlap on normalized entity text spans
- Label agreement rate (fraction of shared spans where canonical labels match)
- Entities where spans match but canonical labels disagree

---

**Layer 1B — Post-alignment, full pipeline (entity span + label comparison)**

Answers: *after full StructSense pipeline runs, do both systems still agree on entity coverage and category?*

Use the complete StructSense output. The label compared is still the extractor agent's `label` field (not the ontology), canonicalized via `_canonicalize_label()` — keeping this layer directly comparable to Layer 1A. Split entities by `judge_score` before computing overlap — low-confidence entities (< 0.8) are flagged separately since StructSense itself has marked them as poor alignments.

```
StructSense full output                            Direct API call
  entity: "missense mutations"                     entity: "missense mutations"
  label: "MUTATION" → _canonicalize_label → GENE   label: "gene_protein" → _canonicalize_label → GENE
  judge_score: 1.0 → high confidence
    ↓
  ← Layer 1B: compare normalized spans + canonical labels (high-conf only) →
```

```python
high_conf = [e for e in structsense_output["entities"] if (e.get("judge_score") or 0) >= 0.8]
low_conf  = [e for e in structsense_output["entities"] if (e.get("judge_score") or 0) <  0.8]

structsense_spans = {normalize_span(e["entity"]) for e in high_conf}
api_spans         = {normalize_span(e["entity"]) for e in api_output["entities"]}

jaccard = len(structsense_spans & api_spans) / len(structsense_spans | api_spans)
```

Metrics at this checkpoint:
- Jaccard overlap on normalized entity text spans (high-confidence only)
- Label agreement rate (fraction of shared spans where canonical labels match)
- Low-confidence entity rate — proportion of StructSense entities excluded due to poor alignment
- Entities where spans match but canonical labels disagree

---

### Layer 2 — Label Quality (System-Specific, Independent)

Evaluate each system's labels independently on their own terms. Do **not** cross-compare labels directly — this layer is about correctness within each system's label space, not agreement between systems.

**For StructSense:** `judge_score` and `remarks` are already present in the output — produced by the quality judging agent. Use these directly rather than running a separate judge call.

| System | Primary signal | What is evaluated |
|---|---|---|
| StructSense (any mode) | `judge_score` + `remarks` (built-in) | Did the judging agent flag poor alignment? Aggregate `judge_score` distribution per paper. Flag entities with `judge_score < 0.8` for manual review using the `remarks` field. |
| StructSense (BioPortal mode) | `ontology` acronym + `ontology_label` | Did the Recommender select a sensible ontology? e.g. did "hippocampus" map to RADLEX/UBERON vs. an unrelated ontology? |
| StructSense (local/fast mode) | `ontology` acronym + `ontology_label` | Is the local service's chosen ontology appropriate for the entity type? |
| Direct API | `score_entity()` from `ner_eval.py` | Is the LLM-assigned free-form label correct for this entity? Uses the heuristic dictionary and keyword rules in `ner_eval.py`. |

---

## Metrics

**Layer 1A — Span and label comparison, pre-alignment (per paper)**
```
Jaccard (spans):              |A ∩ B| / |A ∪ B|   (spans normalized + abbreviation-expanded)
StructSense-only spans:       |A \ B|
API-only spans:               |B \ A|
Label agreement rate:         shared spans where _canonicalize_label(ss_label) == _canonicalize_label(api_label) / total shared spans
Label disagreement instances: shared spans where canonical labels differ (raw labels recorded for inspection)
```

**Layer 1B — Span and label comparison, post-alignment (per paper)**
```
Jaccard (spans, high-conf):         computed on StructSense entities with judge_score >= 0.8 only
StructSense-only spans:             |A \ B|
API-only spans:                     |B \ A|
Label agreement rate:               shared spans where canonical labels match / total shared spans
Low-confidence entity rate:         entities with judge_score < 0.8 / total StructSense entities
Label disagreement instances:       shared spans where canonical labels differ
```

**Layer 2 — Label quality (per system)**
```
StructSense:
  judge_score distribution (mean, median, % below 0.8 threshold)
  ontology coverage: fraction of entities where result["ontology"] is a known neuroscience ontology
  IOBC rate: fraction of entities mapped to IOBC (broad ontology, warrants manual review)

Direct API:
  score_entity() verdict distribution (correct / incorrect / unknown) from ner_eval.py
  LLM-judge correctness rate (if using consensus approach)
```

Aggregate all metrics across papers.

---

## Evaluation Prompt Framing

Two valid approaches — run both:

**Framing A — Best vs. Best (primary)**
Give each system its optimal prompt. Evaluates systems as they'd be used in production. Answers: *"which should we use?"*

**Framing B — Controlled (diagnostic, 5 papers)**
Give both systems the same prompt. Isolates the architectural contribution of multi-agent orchestration from prompt engineering. Answers: *"does the orchestration overhead add value?"*

---

## Capability Gap Analysis

Adding tools and chunking to the direct API call closes the gap progressively:

```
Direct API call          Direct API +           Direct API +
(no tools, no chunking)  tools + chunking        tools + chunking
                                                 + judge call
        ↑                       ↑                      ↑
    baseline              partial parity          near-StructSense
                                                  (minus CrewAI glue)
```

Running evaluations at each level identifies which StructSense components earn their complexity.

### Remaining gaps even with tools + chunking + judge call

| Gap | Notes |
|---|---|
| Agent specialization | StructSense uses separate system prompts and roles per stage |
| State management | CrewAI passes context between stages automatically |
| Parallelism | StructSense chunks run in parallel; naive implementation is sequential |
| Retry / error handling | Built into CrewAI; must be hand-rolled otherwise |
| Config-driven pipeline | StructSense is declarative (YAML); direct approach is imperative (Python) |

---

## Implementation Notes

```python
# Normalization runs before ANY span comparison (Layer 1A or Layer 1B)
# "PFC" and "prefrontal cortex" are treated as a match via abbreviation expansion.
# normalize_span() in layer1_metrics.py handles this.
# is_excluded() from ner_eval.py filters noise before comparison.

# Label comparison uses _canonicalize_label() from ner_eval.py — no fixed schema needed.

# Recommended outputs:
# 1. per-paper span overlap CSV (Layer 1)
#    columns: paper_id, checkpoint (A|B), jaccard,
#             structsense_only_count, api_only_count,
#             structsense_low_conf_count  # entities excluded due to judge_score < 0.8
#
# 2. per-paper label disagreements CSV (Layer 1, for qualitative review)
#    columns: paper_id, entity_normalized, in_structsense, in_api,
#             structsense_label, structsense_canonical,
#             api_label, api_canonical
#
# 3. StructSense entity quality CSV (Layer 2)
#    columns: paper_id, entity_text, extractor_label, canonical_label,
#             ontology_acronym, ontology_label,
#             judge_score, remarks, concept_mapping_provenance,
#             alignment_mode (bioportal|local|fast)
#
# 4. Direct API label quality CSV (Layer 2)
#    columns: paper_id, entity_text, assigned_label, canonical_label,
#             score_entity_verdict (correct|incorrect|unknown), expected_label
#
# 5. StructSense post-alignment entity CSV (Layer 1B input, StructSense only)
#    columns: paper_id, entity_text, extractor_label, canonical_label,
#             ontology_id (full IRI), ontology_acronym, ontology_label,
#             judge_score, remarks, alignment_mode, confidence_band (high|low)
```

---

## Key Risks

**Canonicalization gaps** — `_canonicalize_label()` in `ner_eval.py` covers observed label variants but is not exhaustive. A label the LLM invents that has no alias entry will not match even a semantically equivalent label from the other system — it will appear as a disagreement rather than an unknown. Monitor the label disagreement records for patterns of uncovered labels and extend `_LABEL_ALIASES` or `_LABEL_PREFIX_RULES` in `ner_eval.py` as needed.

**IOBC is the dominant ontology in observed output** — real StructSense output shows `IOBC` (Integrated Object-Based Corpus, `purl.jp/bio/4/`) as the returned ontology for GENE, MUTATION, and DISEASE entities. IOBC is a broad Japanese bioinformatics ontology and is not a standard neuroscience resource. It is not used in the Layer 1 label comparison (which uses the extractor `label` field), but it is relevant to Layer 2 ontology quality analysis. Treat IOBC-mapped entities as requiring manual review.

**StructSense label taxonomy is not fully documented** — the extractor agent's label set (MUTATION, GENE, DISEASE, etc.) depends on the extractor agent's prompt and config. Run the extractor on a small sample and inspect unique `label` values to verify they canonicalize correctly before running at scale.

**Both tools return full IRIs — use the `ontology` field, never parse the IRI** — both BioPortal and the local service return `{ontology_id: <full IRI>, ontology_label: <str>, ontology: <acronym>}`. The acronym is always in `result["ontology"]`. Never attempt to extract it from the IRI string — IRI formats vary widely across ontologies (OBO, BioPortal, RADLEX, MEDDRA, custom) and parsing is fragile. Always look up `result["ontology"]` in the collapse mapping.

**BioPortal Recommender non-determinism** — BioPortal calls `/recommender` before each term lookup to auto-select ontologies. The same term can be mapped to different ontologies on different runs depending on recommender output. "hippocampus" may return RADLEX on one run and UBERON on another. This makes BioPortal alignment results non-deterministic and the collapsed schema label unstable across runs. Log the `ontology` field for every entity in every run.

**BioPortal latency confound** — the BioPortal tool throttles to a minimum of 0.7s between per-term requests (sequential, not batched). This is the source of the ~60 min runtime for large papers. Do not compare raw latency between BioPortal mode and local/fast mode — they are architecturally incomparable on this dimension. Latency comparison is only meaningful between the direct API call and StructSense in local/fast alignment mode.

**Local service ontology coverage** — the local service is indexed against a specific set of ontologies that may differ from BioPortal's. A concept BioPortal maps to UBERON may map to SNOMEDCT in the local service, changing the collapsed label. Document which ontologies the local service indexes before finalising the mapping.

**In-memory cache confound** — `ConceptMappingLocalTool` maintains an in-memory cache keyed by `local|{term}|{max_results}`. After the first run on a corpus, repeated runs serve from cache, making latency measurements artificially fast and consistency scores artificially high. Clear the cache (or use a fresh process) between evaluation runs when measuring latency or consistency.

**Prompt drift (pre-alignment comparison only)** — if StructSense's extractor agent and the direct API call use sufficiently different prompts, Layer 1A comparison may measure prompt engineering rather than architecture. For Framing B (controlled), use the same system prompt text for both systems. For Framing A (best-vs-best), document each system's prompt alongside results so differences can be attributed.

**BioPortal query variability (cloud mode only)** — BioPortal results can vary based on query phrasing and API version. If the alignment agent is free to rephrase entity strings before querying, alignment output is not fully deterministic. Log all BioPortal queries and responses for reproducibility.