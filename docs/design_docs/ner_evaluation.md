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

This two-stage design means StructSense's final output is in **ontology space**, while a direct API call produces output in **schema space** (fixed category labels). The exact ontology format depends on which alignment mode is active — this affects the collapse mapping. See Label Space Problem below.

---

## StructSense Feature Set

| Feature | Description |
|---|---|
| Multi-agent pipeline | Extraction → ontology alignment → quality judging → optional human-in-the-loop |
| Task-type auto-detection | Detects NER, resource extraction, or structured extraction from config |
| Chunking | Splits large PDFs into sentence-aligned chunks; runs extraction in parallel |
| Fast alignment | Skips the alignment LLM agent entirely; calls `POST /map/batch` on the local concept mapping service directly. No BioPortal query, no Recommender API call. Returns `{ontology_id, ontology_label, ontology}` — same shape as BioPortal tool. In-memory cache means repeated runs on the same corpus bypass HTTP entirely. |
| Pluggable concept mapping | BioPortal (cloud) or local hybrid BM25 + dense retrieval, switchable via env var |
| Partial pipeline | Run any subset of stages via `--skip_stage` / `--preload_stage` |
| Any LLM via OpenRouter | Model configured per agent in YAML |
| Single config file | One YAML drives the entire pipeline |

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
| Direct API call | N/A | Fixed 8-category schema | `brain_region`, `neurotransmitter` |

Both StructSense alignment modes return the **same output shape**: `{ontology_id, ontology_label, ontology}`. The `ontology` field always carries the acronym directly — no IRI parsing is needed for the collapse mapping. The key difference between modes is *which ontologies* are selected and *how*: BioPortal uses a Recommender API to auto-detect ontologies per term (non-deterministic), while the local service uses its own indexed ontology set. Forcing identical labels at extraction time would break StructSense's pipeline — see Evaluation Strategy below.

There is also a **third label space dimension**: StructSense's extractor agent uses its own internal label taxonomy (`MUTATION`, `GENE`, `DISEASE`, etc.) which is distinct from both the ontology space and the direct API's 8-category schema. Two collapse mappings are therefore required — see Ontology Collapse Mapping below.

---

## Entity Schema (Direct API Call and Pre-Alignment Comparison)

The fixed schema used by the direct API call, and for pre-alignment comparison against StructSense's extractor agent output:

| Label | Examples |
|---|---|
| `brain_region` | prefrontal cortex, CA1, striatum |
| `cell_type` | pyramidal neuron, astrocyte, interneuron |
| `neurotransmitter` | dopamine, GABA, serotonin |
| `receptor` | NMDA-R, D2 receptor |
| `technique` | fMRI, patch clamp, optogenetics |
| `behavior_task` | fear conditioning, Morris water maze |
| `disorder` | Parkinson's disease, schizophrenia |
| `gene_protein` | BDNF, tau, PSD-95 |

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

## Ontology Collapse Mapping (StructSense Post-Alignment)

Two collapse mappings are required to bring StructSense's output into the same space as the direct API call.

### Mapping A — Ontology acronym → schema label

Use `result["ontology"]` directly as the lookup key — **never parse the IRI**.

```python
# Both BioPortal and local tools return: {ontology_id: <IRI>, ontology_label: <str>, ontology: <acronym>}

ONTOLOGY_TO_SCHEMA = {
    # Brain regions
    "UBERON":   "brain_region",
    "FMA":      "brain_region",
    "RADLEX":   "brain_region",     # BioPortal maps many neuroanatomy terms to RADLEX
                                    # e.g. hippocampus → RADLEX:RID6529
    # Cell types
    "CL":       "cell_type",        # Cell Ontology
    # Neurotransmitters / small molecules
    "CHEBI":    "neurotransmitter",
    # Genes / proteins / mutations
    "PR":       "gene_protein",     # Protein Ontology
    "NCIT":     "gene_protein",     # NCI Thesaurus
    "GO":       "gene_protein",     # Gene Ontology — in BioPortal fallback set
    "IOBC":     "gene_protein",     # Integrated Object-Based Corpus (Japanese bioinformatics ontology)
                                    # observed in real StructSense output for GENE, MUTATION entities
                                    # broad mapping — review unclassified IOBC entries manually
    # Disorders / phenotypes
    "DOID":     "disorder",
    "MONDO":    "disorder",
    "HP":       "disorder",         # Human Phenotype Ontology
    "SNOMEDCT": "disorder",         # in BioPortal fallback ontology set
    "MEDDRA":   "disorder",         # BioPortal returns MEDDRA for clinical phrases
    # Behaviour / tasks
    "NBO":      "behavior_task",
    # Techniques
    "ERO":      "technique",
    "OBI":      "technique",        # Ontology for Biomedical Investigations
}

def collapse_ontology_to_schema(ontology_acronym: str) -> str:
    return ONTOLOGY_TO_SCHEMA.get(ontology_acronym, "unclassified")
```

### Mapping B — StructSense label → schema label

The extractor agent uses its own label taxonomy. This must also be mapped to the 8-category schema for pre-alignment comparison (Layer 1A) and as a cross-check against the ontology mapping.

```python
STRUCTSENSE_LABEL_TO_SCHEMA = {
    "GENE":          "gene_protein",
    "PROTEIN":       "gene_protein",
    "MUTATION":      "gene_protein",
    "BRAIN_REGION":  "brain_region",
    "ANATOMY":       "brain_region",
    "CELL_TYPE":     "cell_type",
    "NEUROTRANSMITTER": "neurotransmitter",
    "RECEPTOR":      "receptor",
    "DISEASE":       "disorder",
    "DISORDER":      "disorder",
    "BEHAVIOR":      "behavior_task",
    "TASK":          "behavior_task",
    "TECHNIQUE":     "technique",
    "METHOD":        "technique",
}

def collapse_label_to_schema(structsense_label: str) -> str:
    return STRUCTSENSE_LABEL_TO_SCHEMA.get(structsense_label.upper(), "unclassified")
```

> **Note:** The StructSense label taxonomy is derived from the extractor agent's prompt/config and may not match the keys above exactly. Inspect actual extractor output to confirm label strings before finalising this mapping.

> **BioPortal fallback ontologies** (used when Recommender API fails):
> `SNOMEDCT`, `MONDO`, `NCIT`, `GO`, `HP`, `CHEBI` — all covered in Mapping A above.

> Track the `unclassified` rate in both mappings as a health signal. A high rate in Mapping A indicates new ontologies appearing in output; a high rate in Mapping B indicates new extractor labels.

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

The label space mismatch means comparison must happen across two independent layers. Do not collapse these into a single metric.

### Layer 1 — Entity Span Comparison (Label-Agnostic)

Compare entity text strings only, before any label is considered. Answers: *do the systems find the same entities?*

This is valid at **two checkpoints**:

**Checkpoint A — Post-extraction, pre-alignment**
Compare StructSense's extractor agent output directly against the direct API call using the `entity` text field. Apply Mapping B (`STRUCTSENSE_LABEL_TO_SCHEMA`) to StructSense labels and compare against the direct API's 8-category labels. This isolates whether multi-agent extraction finds more entities, independent of alignment.

```
PDF text
    ↓
StructSense extractor agent  ←── Layer 1A comparison ──→  Direct API call
  (entity + label fields)        entity text + schema label   (entity text + schema label)
    ↓
StructSense alignment agent
```

**Checkpoint B — Post-alignment (full pipeline)**
Use the full StructSense output. Apply Mapping A (`collapse_ontology_to_schema`) to `result["ontology"]` to bring entity ontology terms into schema space. Filter to entities where `judge_score` is available — low-scoring entities (e.g. < 0.8) should be flagged separately rather than included in the main overlap calculation, since StructSense itself has flagged them as poor alignments.

```python
# Use entity text spans for overlap, ontology collapse for label comparison
structsense_entities = [
    {"text": e["entity"], "schema_label": collapse_ontology_to_schema(e["ontology"]),
     "judge_score": e.get("judge_score")}
    for e in structsense_output["entities"]
]

# Split by judge confidence before computing overlap
high_conf = [e for e in structsense_entities if (e["judge_score"] or 0) >= 0.8]
low_conf  = [e for e in structsense_entities if (e["judge_score"] or 0) <  0.8]

structsense_spans = {e["text"] for e in high_conf}
api_spans         = {e["text"] for e in api_output["entities"]}

jaccard = len(structsense_spans & api_spans) / len(structsense_spans | api_spans)
```

### Layer 2 — Label Quality (System-Specific, Independent)

Evaluate each system's labels independently. Do **not** cross-compare labels directly.

**For StructSense:** `judge_score` and `remarks` are already present in the output — produced by the quality judging agent. Use these directly rather than running a separate judge call.

| System | Primary signal | What is evaluated |
|---|---|---|
| StructSense (any mode) | `judge_score` + `remarks` (built-in) | Did the judging agent flag poor alignment? Aggregate `judge_score` distribution per paper. Flag entities with `judge_score < 0.8` for manual review using the `remarks` field. |
| StructSense (BioPortal mode) | `ontology` acronym + `ontology_label` | Did the Recommender select a sensible ontology? e.g. did "hippocampus" map to RADLEX/UBERON vs. an unrelated ontology? |
| StructSense (local/fast mode) | `ontology` acronym + `ontology_label` | Is the local service's chosen ontology appropriate for the entity type? |
| Direct API | Schema label vs. LLM-judge | Is the assigned 8-category label correct for this entity? |

### Layer 3 — BCKB Downstream Utility (Primary Decision Metric)

Since the ultimate purpose is HOMBA alignment, the most actionable metric is: **are StructSense's BioPortal terms crosswalkable to HOMBA?**

```
StructSense BioPortal output → MNI → AHRA → HOMBA crosswalk → resolvable? (Y/N)
```

This reframes the comparison from *"do the labels match?"* to *"which system produces output more useful to BCKB downstream?"*

---

## Metrics

**Layer 1 — Span overlap (per paper)**
```
Jaccard:             |A ∩ B| / |A ∪ B|
StructSense-only:    |A \ B|   # multi-agent catches; direct API misses
API-only:            |B \ A|   # direct API catches; multi-agent misses
```

**Layer 2 — Label quality (per system)**
```
StructSense:
  judge_score distribution (mean, median, % below 0.8 threshold)
  low-confidence entity rate: entities with judge_score < 0.8 / total entities
  unclassified rate (Mapping A): ontology acronym not in ONTOLOGY_TO_SCHEMA
  unclassified rate (Mapping B): extractor label not in STRUCTSENSE_LABEL_TO_SCHEMA

Direct API:
  Per-category F1 (if human ground truth available)
  LLM-judge correctness rate (if using consensus approach)
```

**Layer 3 — BCKB utility (StructSense only)**
```
HOMBA resolvability rate:  entities successfully crosswalked / total entities
Unclassified rate:         entities where collapse mapping returned "unclassified" / total entities
                           (health signal — high rate indicates mapping gaps or unexpected ontology namespaces)
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
# Normalization runs before ANY comparison (Layer 1 or Layer 2)
# "PFC" and "prefrontal cortex" must be treated as a match, not a miss
# Strategies: alias dictionary, abbreviation expansion, or embedding similarity

# Recommended outputs:
# 1. per-paper span overlap CSV (Layer 1)
#    columns: paper_id, checkpoint (A|B), jaccard,
#             structsense_only_count, api_only_count,
#             structsense_low_conf_count  # entities excluded due to judge_score < 0.8
#
# 2. per-paper disagreements CSV (Layer 1, for qualitative review)
#    columns: paper_id, entity_text, in_structsense, in_api,
#             structsense_label, structsense_schema_label,  # Mapping B output
#             api_schema_label
#
# 3. StructSense entity quality CSV (Layer 2)
#    columns: paper_id, entity_text, extractor_label, schema_label (Mapping B),
#             ontology_acronym, ontology_label, ontology_schema_label (Mapping A),
#             judge_score, remarks, concept_mapping_provenance,
#             alignment_mode (bioportal|local|fast)
#
# 4. Direct API label quality CSV (Layer 2)
#    columns: paper_id, entity_text, assigned_schema_label, judge_correct (Y/N)
#
# 5. HOMBA resolvability CSV (Layer 3, StructSense only)
#    columns: paper_id, entity_text, ontology_id (full IRI), ontology_acronym,
#             ontology_label, judge_score, alignment_mode,
#             collapsed_schema_label, homba_resolvable (Y/N)
```

---

## Key Risks

**Collapse mapping error** — the ontology-to-schema mapping is hand-curated and incomplete. A bad mapping silently conflates alignment quality with extraction quality. Validate on a small sample before running the full corpus and track the `unclassified` rate per run as a health signal.

**IOBC is the dominant ontology in observed output** — real StructSense output shows `IOBC` (Integrated Object-Based Corpus, `purl.jp/bio/4/`) as the returned ontology for GENE, MUTATION, and DISEASE entities. IOBC is a broad Japanese bioinformatics ontology and is not a standard neuroscience resource. It is mapped to `gene_protein` in Mapping A as a starting point, but this is a coarse approximation — IOBC entries can span genes, diseases, and cell components. Treat IOBC-mapped entities as requiring manual review until the mapping is validated.

**StructSense label taxonomy is not documented in the spec** — the extractor agent's label set (MUTATION, GENE, DISEASE, etc.) is inferred from observed output, not from a schema definition. The full set of possible labels depends on the extractor agent's prompt and config. Run the extractor on a small sample and collect all unique `label` values before finalising Mapping B.

**Both tools return full IRIs — use the `ontology` field, never parse the IRI** — both BioPortal and the local service return `{ontology_id: <full IRI>, ontology_label: <str>, ontology: <acronym>}`. The acronym is always in `result["ontology"]`. Never attempt to extract it from the IRI string — IRI formats vary widely across ontologies (OBO, BioPortal, RADLEX, MEDDRA, custom) and parsing is fragile. Always look up `result["ontology"]` in the collapse mapping.

**BioPortal Recommender non-determinism** — BioPortal calls `/recommender` before each term lookup to auto-select ontologies. The same term can be mapped to different ontologies on different runs depending on recommender output. "hippocampus" may return RADLEX on one run and UBERON on another. This makes BioPortal alignment results non-deterministic and the collapsed schema label unstable across runs. Log the `ontology` field for every entity in every run.

**BioPortal latency confound** — the BioPortal tool throttles to a minimum of 0.7s between per-term requests (sequential, not batched). This is the source of the ~60 min runtime for large papers. Do not compare raw latency between BioPortal mode and local/fast mode — they are architecturally incomparable on this dimension. Latency comparison is only meaningful between the direct API call and StructSense in local/fast alignment mode.

**Local service ontology coverage** — the local service is indexed against a specific set of ontologies that may differ from BioPortal's. A concept BioPortal maps to UBERON may map to SNOMEDCT in the local service, changing the collapsed label. Document which ontologies the local service indexes before finalising the mapping.

**In-memory cache confound** — `ConceptMappingLocalTool` maintains an in-memory cache keyed by `local|{term}|{max_results}`. After the first run on a corpus, repeated runs serve from cache, making latency measurements artificially fast and consistency scores artificially high. Clear the cache (or use a fresh process) between evaluation runs when measuring latency or consistency.

**Schema drift (pre-alignment comparison only)** — if StructSense's extractor agent and the direct API call are implicitly using different entity definitions at extraction time, Layer 1A comparison measures prompt engineering rather than architecture. Inspect the extractor agent's prompt and explicitly provide the fixed schema to both systems at this stage.

**BioPortal query variability (cloud mode only)** — BioPortal results can vary based on query phrasing and API version. If the alignment agent is free to rephrase entity strings before querying, alignment output is not fully deterministic. Log all BioPortal queries and responses for reproducibility.