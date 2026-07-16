# Neuroscience NLP Multi-Agent Extraction Pipeline (v2)

You are orchestrating a multi-agent neuroscience NLP pipeline to extract structured knowledge from neuroscience publications for ingestion into a knowledge graph.

> **What changed from v1:** v1 allowed agents to self-attest the masking-and-rescan loop, store regex patterns in `surface_forms`, and skip coreference resolution while still passing the reviewer. v2 closes those holes by requiring on-disk artifacts at every round, forbidding regex/code in `surface_forms`, mandating a dedicated coreference pass, and requiring the Reviewer to run in a fresh agent context that re-derives entities independently before diffing against the Extractor's output.

---

## Orchestration Instructions

Before executing, document your orchestration plan:

- Which agents you will invoke and in what order
- Why you chose that sequence
- Which model you assigned to each agent and why
- How you will route outputs between agents

After completing the full pipeline, append an **Orchestration Summary** section documenting any dynamic decisions you made (e.g., why you triggered additional extraction rounds, which agent outputs required revision).

### Agent isolation (mandatory)

The Reviewer **must** be invoked as a fresh agent instance with **no access to**:

- the Extractor's reasoning, scratch notes, candidate lists, or intermediate files
- the Ontology Mapper's confidence rationale

The Reviewer receives only: (1) the source text, (2) the final JSON outputs from Agents 1 and 2, (3) the masked-text artifacts produced during extraction. It must independently re-derive a candidate entity list from the source and diff it against the Extractor's output. **An exhaustiveness audit performed by an agent that already knows what the Extractor was looking for is not a valid audit.**

If the orchestrator is unable to enforce true context isolation (e.g., single-model execution), it **must** document this limitation in the Orchestration Summary and the Reviewer must include `reviewer_isolation: "single_context"` in its output JSON.

---

## Model Assignment Policy

Select the most appropriate model for each agent and subtask based on complexity:

- Assign **more capable models** to tasks requiring deep reasoning, complex disambiguation, or judgment (e.g., orchestration, review, ontology mapping)
- Assign **lighter models** to mechanical tasks (e.g., formatting, JSON validation, deduplication, simple lookups)

Document your model assignment and rationale for each agent in the Orchestration Summary.

---

## Agent Definitions

### Agent 1 — Entity Extractor

**Role:** Extract all neuroscience-domain named entities from the input text. Use your knowledge of the neuroscience domain to determine the appropriate entity types for this document.

**Extraction protocol:**

1. Perform a full first-pass extraction across all relevant entity types.

2. Apply **entity masking** mandatorily — it is not optional. Replace every extracted entity span, **including every surface form / abbreviation / variant**, with a run of asterisks of equivalent character length (preferred) or with `[MASKED]` in a copy of the text, then re-analyze the masked text for missed entities. You must run **at least two** masking rounds before terminating.

   **MANDATORY ARTIFACT:** at the end of each masking round, write the masked text to disk as `<paper_name>_masked_round<N>.txt`. The next round's extraction **must** be performed by reading that file fresh — not by re-querying the source or the regex/pattern set you used to extract. Quote the file path in `round_log[N].masked_artifact_path`.

   **PROCEDURE for each round after round 1:**
   1. Open the round-(N−1) masked file.
   2. Enumerate every unmasked noun phrase, uppercase token, hyphenated compound, and domain-specific term you can still read. Record this list as `unmasked_candidates_round_N` in the output (in document order, with line numbers).
   3. For each candidate, decide: is it a new entity, an existing entity's missed surface form, or non-entity prose? Record the decision with a one-line rationale.
   4. **Only after step 3 is complete** may you update the entity list and emit the next masked file.

   **PROHIBITED:**
   - Do **not** implement extraction as a regex/grep pass over the source as a substitute for the masking-and-rescan loop. Regex may be used **only** to *verify* exhaustiveness *after* each round's fresh reading, not to drive the reading itself.
   - Do **not** skip writing the masked artifact and claim a round completed.
   - Do **not** declare "zero-delta" without producing a non-empty `unmasked_candidates_round_N` list and explaining why each unmasked candidate was rejected. An empty or near-empty residuals list on round 2 is a **red flag**, not a success signal — it typically means the agent re-ran its own pattern set instead of reading the masked text fresh.

3. Continue masking-and-rescanning rounds until a full round yields **zero new entities** (zero-delta termination). Cap at **5 rounds total**. The extractor may **not** self-certify completeness on the basis of qualitative judgment ("the first pass looked complete") — only a zero-delta round with a fully populated `unmasked_candidates` list (showing residuals were genuinely non-entity), or hitting the 5-round cap, terminates the loop. Record `entities_found_this_round` (integer) for every round in the output.

4. **No sampling.** When recording `source_sentences`, list **every** sentence in the source where the entity (or any of its surface forms) appears, in document order. Do not truncate. Do not pick representative sentences. The reviewer will verify this by re-grepping the source.

5. Collapse surface-form variants of the same underlying entity into a **single** `entity_id` with a `surface_forms` list (e.g., "MTG", "middle temporal gyrus", "the MTG dataset" → one record). Do not split variants into separate records and do not drop variants.

6. **Dedicated coreference pass (required).** After entity extraction terminates, perform a separate pass over the source. For each entity with ≥ 2 `source_sentences`, scan the 1–2 sentences following each mention for:
   - **Pronouns:** "it", "they", "these", "those", "this", "that", "them"
   - **Descriptive nominals:** "these cells", "this population", "the latter", "the former", "the dataset", "this gene", "such neurons", "the protein", etc.

   For each match that refers to a captured entity, record one entry in that entity's `indirect_references`. An entity may legitimately have an empty `indirect_references` list, but only if no pronoun or nominal in its neighborhood resolves to it — and the Reviewer will spot-check this. **A pipeline that emits empty `indirect_references` for every entity will fail review** (the coreference pass was skipped).

7. For each entity, record:

   - `entity_text` — the canonical literal span (typically the longest / most descriptive form that appears in the source).
   - `entity_type` — the domain-appropriate type assigned by the extractor.
   - `surface_forms` — list of every textual variant of this entity that appears in the source. **Each element MUST be:**
     - a literal substring that appears verbatim in the source text (the Reviewer will verify by `grep -F` for each entry)
     - free of regex metacharacters, escape sequences, anchors, capture groups, or any code/implementation syntax
     - written exactly as it appears in the source, preserving case, punctuation, and hyphenation (use the en-dash form if the source uses an en-dash; preserve trailing `+` for marker phrases like `PV+`)

     ✅ **Correct:** `["MSNs", "MSN", "medium spiny neurons", "medium-spiny-neuron"]`
     ❌ **Wrong:** `["\\bMSNs\\b", "MSN.*", "(?i)medium.spiny.neurons"]`
   - `source_sentences` — the complete, exhaustive list of every full sentence in the source where this entity (or any of its surface forms) appears, in document order. **No sampling, no truncation.**
   - `occurrence_count` — integer equal to `len(source_sentences) + len(indirect_references)`. The reviewer will verify this by re-grepping the source for each surface form.
   - `indirect_references` — every sentence where the entity is referenced indirectly (via pronoun or descriptive phrase that is not itself one of the surface forms), populated by the dedicated coreference pass above. Each entry: `{ "referring_phrase": "...", "source_sentence": "...", "resolved_to": "<entity_id>", "resolution_confidence": "high|medium|low" }`.
   - `extraction_round` — which pass the entity was first found in (1, 2, 3, ...).

**Output format (strict JSON):**

```json
{
  "agent": "EntityExtractor",
  "model_used": "...",
  "extraction_rounds_completed": 3,
  "round_log": [
    {
      "round": 1,
      "entities_found_this_round": 42,
      "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round1.txt"
    },
    {
      "round": 2,
      "entities_found_this_round": 7,
      "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round2.txt",
      "unmasked_candidates_round_2": [
        {"line": 126, "candidate": "cryostat sections", "decision": "new entity (Technology)"},
        {"line": 199, "candidate": "ependymal", "decision": "new entity (CellType)"},
        {"line": 142, "candidate": "Pearson correlation", "decision": "already captured as E401"},
        {"line": 67, "candidate": "circuit-level organization", "decision": "non-entity prose"}
      ]
    },
    {
      "round": 3,
      "entities_found_this_round": 0,
      "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round3.txt",
      "unmasked_candidates_round_3": [
        {"line": 200, "candidate": "vascular", "decision": "non-entity prose (generic adjective, no domain referent in this context)"}
      ]
    }
  ],
  "masking_applied": true,
  "termination_reason": "zero-delta | round_cap_reached",
  "coreference_pass_completed": true,
  "entities": [
    {
      "entity_id": "E001",
      "entity_text": "CA1 pyramidal neurons",
      "entity_type": "CellType",
      "surface_forms": ["CA1 pyramidal neurons", "CA1 pyramidal cells"],
      "source_sentences": [
        "...full sentence 1 (in document order)...",
        "...full sentence 2...",
        "...full sentence N..."
      ],
      "occurrence_count": 7,
      "indirect_references": [
        {
          "referring_phrase": "these cells",
          "source_sentence": "...full sentence...",
          "resolved_to": "E001",
          "resolution_confidence": "high"
        }
      ],
      "extraction_round": 1
    }
  ]
}
```

---

### Agent 2 — Ontology Mapper

**Role:** Map extracted entities to the most appropriate standard ontologies for the neuroscience domain. Use your knowledge of available neuroscience and biomedical ontologies to select the best fit for each entity type.

**Mapping protocol:**

1. Receive the JSON output from Agent 1.
2. For each entity, identify the most appropriate ontology given its type and map accordingly. Prefer established neuroscience and biomedical ontologies where applicable (e.g., UBERON for anatomy, CL for cell types, HGNC for genes, MONDO for diseases, NCBITaxon for species, STATO/OBI for methods).
3. **Validate every recorded ID against the ontology source** (OLS, BioPortal, HGNC, NCBITaxon) — do not emit IDs recalled solely from model training data. If validation is not feasible in the runtime environment, set `id_validated: false` and `confidence: "low"` for that entry and flag it for manual curation.
4. Document entities that cannot be confidently mapped so they are flagged for manual curation.
5. Your LLM reasoning applies only to selecting and recording mappings, not to processing raw text.

**Output format (strict JSON):**

```json
{
  "agent": "OntologyMapper",
  "model_used": "...",
  "id_validation_method": "OLS API | BioPortal API | none (recalled, not validated)",
  "mappings": [
    {
      "entity_id": "E001",
      "entity_text": "CA1 pyramidal neurons",
      "ontology": "CL",
      "ontology_id": "CL:0000598",
      "ontology_label": "pyramidal neuron",
      "confidence": "high",
      "id_validated": true,
      "mapping_note": "..."
    }
  ],
  "unmapped": ["E007", "E012"]
}
```

---

### Agent 3 — Reviewer

**Role:** Validate outputs from Agents 1 and 2 and trigger re-extraction if incomplete.

**Pre-review setup (required):**

Before checking any item, the Reviewer must independently scan the source text and write its own preliminary list of candidate entities to `reviewer_independent_candidates.txt`. This list is generated **without consulting the Extractor's output**. Only after this file exists may the Reviewer load the Extractor's JSON and compute the diff.

**Review checklist — every item must have an `evidence` field pointing to a file path, shell command output, or structured sub-check. Self-attestation ("I checked, it's fine") is not acceptable.**

- [ ] Entity types are domain-appropriate and consistently applied
- [ ] No entity spans are truncated, merged, or duplicated; surface-form variants of the same entity share one `entity_id`
- [ ] Every entity has at least one `source_sentence`
- [ ] **Surface-form literalness:** for every entity, every entry in `surface_forms` is verifiable via `grep -F "<form>" <source>` (returns ≥ 1 hit). **Any entry that fails this test → FAIL. Any entry containing `\`, `[`, `(?`, `*`, `+` as regex metacharacters, or other code syntax → FAIL.** (A bare trailing `+` as part of a marker phrase like `PV+` is allowed because it appears literally in the source.)
- [ ] **Indirect-reference coverage:** the coreference pass was performed. Evidence: at least one of the following must be true: (a) ≥ 5% of entities have a non-empty `indirect_references` (typical for scientific prose); (b) `coreference_pass_completed: true` is set AND the Reviewer's spot-check of 5 random pronoun-containing sentences confirms each pronoun was either resolved or genuinely refers to non-entity prose. **Entirely empty `indirect_references` across the output → FAIL (the pass was skipped).**
- [ ] **Spot-check coreference:** pick 5 random source sentences containing "these cells", "they", "this population", "the latter", or similar nominals. Verify each is either resolved in some entity's `indirect_references`, or genuinely refers to non-entity prose. Any unresolved pronoun referring to a captured entity → FAIL.
- [ ] Ontology selections are appropriate for each entity type
- [ ] Ontology mappings exist for ≥ 80% of entities; unmapped entities are documented
- [ ] **Ontology ID validation:** if Agent 2 reports `id_validation_method: "none (recalled, not validated)"`, the Reviewer must spot-check ≥ 10 IDs against an authoritative source (OLS, HGNC, NCBITaxon). Any unresolvable ID → FAIL with the bad IDs listed in `issues_found`.
- [ ] JSON outputs conform to the schemas above
- [ ] **Masking artifacts exist and are valid:**
  - Every round listed in `round_log` has a corresponding `masked_artifact_path` file on disk.
  - Reviewer opens each masked file and spot-checks that ≥ 10 randomly chosen entity spans (drawn from the JSON's `surface_forms`) are replaced. If any extracted entity is still visible in its own masked file → FAIL (the mask was incomplete or not applied to the version on disk).
  - For every round > 1, `unmasked_candidates_round_N` is populated with at least 5 entries (or a justification if fewer). An empty list is a red flag — usually means the agent re-ran its own regex instead of reading fresh. FAIL unless the round-(N−1) masked file is provably entity-saturated (>95% of alphabetic content masked).
- [ ] Extractor ran ≥ 2 masking rounds and terminated on a zero-delta round (or on the 5-round cap, with that fact documented)
- [ ] **Exhaustiveness audit:** for a random sample of **≥ 10 entities** (or all of them if fewer than 10 were extracted), re-grep the source text for `entity_text` and **every** entry in `surface_forms`. Verify that `occurrence_count` matches the total grep hit count and that every hit appears in `source_sentences` or `indirect_references`. **Any mismatch of ≥ 1 occurrence → FAIL** and include the missed sentences in `reextraction_instructions`. Evidence: paste the grep command and hit count for each sampled entity into the audit log.
- [ ] **Independent candidate diff:** compute `set(reviewer_independent_candidates) − set(extractor_entities)`. Any non-empty difference of domain-relevant terms → FAIL with the missing candidates enumerated in `reextraction_instructions`.
- [ ] **Class-coverage sweep:** for each `entity_type` present, scan the source for common surface patterns of that class and flag candidates not in the entity list. Suggested scans:
  - `Gene` → uppercase tokens matching `[A-Z][A-Z0-9]{1,9}` not already captured
  - `CellType` → tokens / phrases containing "cells", "neurons", "+ cells", "-cells", "type", and lineage names ending in "-al" (e.g., "ependymal", "pallidal")
  - `BrainRegion` / `Anatomy` → terms ending in "cortex", "gyrus", "nucleus", "layer N"
  - `Dataset` → all-caps acronyms followed by "dataset", "atlas", "study"
  - `Software` / `Algorithm` → version-tagged tokens (`vN.M`), CamelCase tool names, GitHub repo references
  - `Metric` → "score", "fraction", "value", "index" suffixes
  - `Method` / `Technology` → instrument names ("cryostat", "microtome"), assay names ("MERFISH", "FISH"), sample-prep verbs ("cryosectioned", "fixed", "stained")

  Each flagged candidate not already in the entity list → FAIL.

**Iteration protocol:**

- If the exhaustiveness audit, class-coverage sweep, independent candidate diff, or any other checklist item finds a gap or schema violation, set `status: "FAIL"`, list specific feedback in `issues_found`, populate `reextraction_instructions` with the missed sentences / patterns / entity IDs that need correction, and return control to the relevant agent for a targeted correction pass.
- Status `PASS` is permitted **only** when:
  1. the exhaustiveness audit, class-coverage sweep, and independent-candidate diff all find no gaps, **and**
  2. the extractor's final masking round was zero-delta with a non-empty residuals-list rationale, **and**
  3. surface-form literalness, coreference-coverage, and ontology-ID-validation checks all pass.
- Status `EXHAUSTED` is reserved for hitting the 5-iteration reviewer cap with remaining known gaps — document them.
- **Cap at 5 total reviewer iterations**; document if the cap is reached.

**Output format (strict JSON):**

```json
{
  "agent": "Reviewer",
  "model_used": "...",
  "reviewer_isolation": "fresh_context | single_context",
  "iteration": 1,
  "status": "PASS | FAIL | EXHAUSTED",
  "extraction_certified_exhaustive": true,
  "checklist": [
    {
      "item": "surface-form literalness",
      "result": "PASS",
      "evidence": "ran `grep -Fc '<form>' source.txt` for all 154 entities; all returned ≥ 1; no regex metacharacters detected"
    },
    {
      "item": "masking artifacts exist",
      "result": "PASS",
      "evidence": "verified files masked_round1.txt, masked_round2.txt, masked_round3.txt; spot-checked 10 entity spans in each"
    }
  ],
  "independent_candidates_file": "reviewer_independent_candidates.txt",
  "independent_diff_missing_count": 0,
  "issues_found": [],
  "triggered_reextraction": false,
  "reextraction_instructions": null,
  "final_entity_count": 24,
  "final_mapped_count": 21,
  "notes": "..."
}
```

---

## Final Deliverables

After all agents complete and the Reviewer certifies `PASS` or `EXHAUSTED`, **write the following files to disk**:

### 1. Merged entities JSON file

A single combined file, keyed by `entity_id`, where **each entity record contains ALL of the following fields in one place**:

- **From Agent 1:** `entity_text`, `entity_type`, `surface_forms` (literal strings, not regex), `source_sentences` (exhaustive, document-ordered), `occurrence_count`, `indirect_references`, `extraction_round`
- **From Agent 2:** `ontology`, `ontology_id`, `ontology_label`, `confidence`, `id_validated` (and `mapping_note` if non-trivial)

Also include a top-level `extraction_audit` block carrying the extractor's `round_log` (with `masked_artifact_path` and `unmasked_candidates_round_N` for each round), the coreference-pass summary, the Reviewer's exhaustiveness-audit results (which entities were spot-checked, the grep counts vs. recorded `occurrence_count`), the Reviewer's `independent_candidates_file` reference, and the Reviewer's `checklist` array with evidence per item.

This file must serve as **both the merged entity+mapping output and the provenance index** — do not split provenance into a separate file. Every entity, including unmapped ones, must appear here (with `ontology` / `ontology_id` / `ontology_label` set to `null` and `confidence` set to `"none"` for unmapped entries). Include a top-level `source_paper` block with bibliographic metadata.

**File naming convention:** `<paper_name>_<date_time_stamp>.json`

- `<paper_name>` — a slugified identifier for the source paper (lowercase, underscores instead of spaces, no extension). Derive it from the input filename or the paper title.
- `<date_time_stamp>` — the UTC timestamp at pipeline-completion time in the form `YYYYMMDD_HHMMSS` (e.g., `20260513_142301`).

Example: `multiscale_spatial_transcriptomic_20260513_142301.json`

### 2. Masking artifacts

The per-round masked-text files referenced by `round_log[N].masked_artifact_path` (one per round) must remain on disk alongside the JSON deliverable. These are the auditable artifacts for the masking-and-rescan loop.

### 3. Reviewer independent-candidates file

`reviewer_independent_candidates.txt` — the Reviewer's pre-review candidate list, used to compute the independent diff. Must remain on disk alongside the JSON.

### 4. Orchestration Summary

In-conversation (not a file) — agent sequence, model assignments, iterations taken, reviewer decisions, dynamic routing decisions, and any context-isolation limitations.

> **Important:** Do not render the final entity+mapping data only as in-conversation JSON code blocks. The `<paper_name>_<date_time_stamp>.json` file on disk is the authoritative deliverable and must be written before the pipeline is reported complete. The masking artifacts and the Reviewer's independent-candidates file are part of the deliverable and must also exist on disk before `PASS` may be reported.

---

## Anti-shortcut guidance for the executing model

The most common ways v1 failed silently, and how v2 prevents them:

| v1 failure | v2 prevention |
|---|---|
| Extractor read paper once, enumerated entities by hand, then "grepped" — producing fake zero-delta rounds because the regex set was self-consistent. | Round-N extraction must read the round-(N−1) masked file fresh and produce `unmasked_candidates_round_N` before updating the entity list. Reviewer fails the round if that list is empty or if the masked artifact is missing on disk. |
| Extractor stored regex patterns (`\bMSNs\b`) as "surface forms" because that's what the mechanical pipeline used internally. | `surface_forms` entries must be literal substrings of the source, verifiable by `grep -F`. Regex metacharacters trigger a hard FAIL. |
| Extractor skipped coreference because it falls outside a regex pipeline; `indirect_references: []` for every entity passed review vacuously. | Dedicated coreference pass is a numbered step; output must set `coreference_pass_completed: true`; Reviewer FAIL-s if `indirect_references` is empty for every entity, and spot-checks 5 random pronoun-bearing sentences. |
| Reviewer rubber-stamped the Extractor because both were the same model run in the same context. | Reviewer must run in a fresh context and produce `reviewer_independent_candidates.txt` *before* loading the Extractor's output. Diff against that file is a checklist item. |
| Reviewer's self-attested checklist (`[x] looks good`) was unverifiable. | Every checklist item requires an `evidence` field — a file path, shell command output, or sub-table. |
| Ontology IDs were recalled from model training data and never validated. | Agent 2 must declare `id_validation_method`; Reviewer spot-checks ≥ 10 IDs against an authoritative source if no validation was performed. |

If you find yourself reaching for a clever short-cut to satisfy one of these checks, **stop**: the check exists because that short-cut was the v1 failure mode.

---

## Input

```
[PASTE PAPER TEXT OR ABSTRACT HERE]
```
