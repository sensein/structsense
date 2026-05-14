# Neuroscience NLP Multi-Agent Extraction Pipeline

You are orchestrating a multi-agent neuroscience NLP pipeline to extract structured knowledge from neuroscience publications for ingestion into a knowledge graph.

---

## Orchestration Instructions

Before executing, document your orchestration plan:

- Which agents you will invoke and in what order
- Why you chose that sequence
- Which model you assigned to each agent and why
- How you will route outputs between agents

After completing the full pipeline, append an **Orchestration Summary** section documenting any dynamic decisions you made (e.g., why you triggered additional extraction rounds, which agent outputs required revision).

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
2. Apply **entity masking** mandatorily — it is not optional. Replace every extracted entity span, **including every surface form / abbreviation / variant**, with `[MASKED]` in a copy of the text, then re-analyze the masked text for missed entities. You must run **at least two** masking rounds before terminating.
3. Continue masking-and-rescanning rounds until a full round yields **zero new entities** (zero-delta termination). Cap at **5 rounds total**. The extractor may **not** self-certify completeness on the basis of qualitative judgment ("the first pass looked complete") — only a zero-delta round, or hitting the 5-round cap, terminates the loop. Record `entities_found_this_round` (integer) for every round in the output.
4. **No sampling.** When recording `source_sentences`, list **every** sentence in the source where the entity (or any of its surface forms) appears, in document order. Do not truncate. Do not pick representative sentences. The reviewer will verify this by re-grepping the source.
5. Collapse surface-form variants of the same underlying entity into a **single** `entity_id` with a `surface_forms` list (e.g., "MTG", "middle temporal gyrus", "the MTG dataset" → one record). Do not split variants into separate records and do not drop variants.
6. For each entity, record:
   - `entity_text` — the canonical literal span (typically the longest / most descriptive form that appears in the source)
   - `entity_type` — the domain-appropriate type assigned by the extractor
   - `surface_forms` — list of every textual variant of this entity that appears in the source (abbreviations, full names, alternate spellings, plural/singular forms)
   - `source_sentences` — the complete, exhaustive list of every full sentence in the source where this entity (or any of its surface forms) appears, in document order. **No sampling, no truncation.**
   - `occurrence_count` — integer equal to `len(source_sentences) + len(indirect_references)`. The reviewer will verify this by re-grepping the source for each surface form.
   - `indirect_references` — every sentence where the entity is referenced indirectly (via pronoun or descriptive phrase that is not itself one of the surface forms); record both the referring phrase and the resolved entity
   - `extraction_round` — which pass the entity was first found in (1, 2, 3, ...)

**Output format (strict JSON):**

```json
{
  "agent": "EntityExtractor",
  "model_used": "...",
  "extraction_rounds_completed": 3,
  "round_log": [
    {"round": 1, "entities_found_this_round": 42},
    {"round": 2, "entities_found_this_round": 7},
    {"round": 3, "entities_found_this_round": 0}
  ],
  "masking_applied": true,
  "termination_reason": "zero-delta | round_cap_reached",
  "entities": [
    {
      "entity_id": "E001",
      "entity_text": "CA1 pyramidal neurons",
      "entity_type": "CellType",
      "surface_forms": ["CA1 pyramidal neurons", "CA1 pyramidal cells", "these neurons"],
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
          "resolved_to": "E001"
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
2. For each entity, identify the most appropriate ontology given its type and map accordingly. Prefer established neuroscience and biomedical ontologies where applicable.
3. Document entities that cannot be confidently mapped so they are flagged for manual curation.
4. Your LLM reasoning applies only to selecting and recording mappings, not to processing raw text.

**Output format (strict JSON):**

```json
{
  "agent": "OntologyMapper",
  "model_used": "...",
  "mappings": [
    {
      "entity_id": "E001",
      "entity_text": "CA1 pyramidal neurons",
      "ontology": "CL",
      "ontology_id": "CL:0000598",
      "ontology_label": "pyramidal neuron",
      "confidence": "high",
      "mapping_note": "..."
    }
  ],
  "unmapped": ["E007", "E012"]
}
```

---

### Agent 3 — Reviewer

**Role:** Validate outputs from Agents 1 and 2 and trigger re-extraction if incomplete.

**Review checklist:**

- [ ] Entity types are domain-appropriate and consistently applied
- [ ] No entity spans are truncated, merged, or duplicated; surface-form variants of the same entity share one `entity_id`
- [ ] Every entity has at least one `source_sentence`
- [ ] Indirect references are resolved to a valid `entity_id`
- [ ] Ontology selections are appropriate for each entity type
- [ ] Ontology mappings exist for ≥ 80% of entities; unmapped entities are documented
- [ ] JSON outputs conform to the schemas above
- [ ] Extractor ran ≥ 2 masking rounds and terminated on a zero-delta round (or on the 5-round cap, with that fact documented)
- [ ] **Exhaustiveness audit:** for a random sample of **≥ 10 entities** (or all of them if fewer than 10 were extracted), re-grep the source text for `entity_text` and **every** entry in `surface_forms`. Verify that `occurrence_count` matches the total grep hit count and that every hit appears in `source_sentences` or `indirect_references`. **Any mismatch of ≥ 1 occurrence → FAIL** and include the missed sentences in `reextraction_instructions`.
- [ ] **Class-coverage sweep:** for each `entity_type` present, scan the source for common surface patterns of that class and flag candidates not in the entity list. Suggested scans:
  - `Gene` → uppercase tokens matching `[A-Z][A-Z0-9]{1,9}` not already captured
  - `CellType` → tokens / phrases containing "cells", "neurons", "+ cells", "-cells", "type"
  - `BrainRegion` / `Anatomy` → terms ending in "cortex", "gyrus", "nucleus", "layer N"
  - `Dataset` → all-caps acronyms followed by "dataset", "atlas", "study"
  - `Software` / `Algorithm` → version-tagged tokens (`vN.M`), CamelCase tool names, GitHub repo references
  - `Metric` → "score", "fraction", "value", "index" suffixes
  Each flagged candidate not already in the entity list → FAIL.

**Iteration protocol:**

- If the exhaustiveness audit, class-coverage sweep, or any other checklist item finds a gap or schema violation, set `status: "FAIL"`, list specific feedback in `issues_found`, populate `reextraction_instructions` with the missed sentences / patterns / entity IDs that need correction, and return control to the relevant agent for a targeted correction pass.
- Status `PASS` is permitted **only** when the exhaustiveness audit and class-coverage sweep both find no gaps **and** the extractor's final masking round was zero-delta.
- Status `EXHAUSTED` is reserved for hitting the 5-iteration reviewer cap with remaining known gaps — document them.
- **Cap at 5 total reviewer iterations**; document if the cap is reached.

**Output format (strict JSON):**

```json
{
  "agent": "Reviewer",
  "model_used": "...",
  "iteration": 1,
  "status": "PASS | FAIL | EXHAUSTED",
  "extraction_certified_exhaustive": true,
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

After all agents complete and the Reviewer certifies `PASS` or `EXHAUSTED`, **write the following file to disk**:

### 1. Merged entities JSON file

A single combined file, keyed by `entity_id`, where **each entity record contains ALL of the following fields in one place**:

- **From Agent 1:** `entity_text`, `entity_type`, `surface_forms`, `source_sentences` (exhaustive, document-ordered), `occurrence_count`, `indirect_references`, `extraction_round`
- **From Agent 2:** `ontology`, `ontology_id`, `ontology_label`, `confidence` (and `mapping_note` if non-trivial)

Also include a top-level `extraction_audit` block carrying the extractor's `round_log` and the reviewer's exhaustiveness-audit results (which entities were spot-checked, the grep counts vs. recorded `occurrence_count`).

This file must serve as **both the merged entity+mapping output and the provenance index** — do not split provenance into a separate file. Every entity, including unmapped ones, must appear here (with `ontology` / `ontology_id` / `ontology_label` set to `null` and `confidence` set to `"none"` for unmapped entries). Include a top-level `source_paper` block with bibliographic metadata.

**File naming convention:** `<paper_name>_<date_time_stamp>.json`

- `<paper_name>` — a slugified identifier for the source paper (lowercase, underscores instead of spaces, no extension). Derive it from the input filename or the paper title.
- `<date_time_stamp>` — the UTC timestamp at pipeline-completion time in the form `YYYYMMDD_HHMMSS` (e.g., `20260513_142301`).

Example: `multiscale_spatial_transcriptomic_20260513_142301.json`

### 2. Orchestration Summary

In-conversation (not a file) — agent sequence, model assignments, iterations taken, reviewer decisions, and any dynamic routing decisions.

> **Important:** Do not render the final entity+mapping data only as in-conversation JSON code blocks. The `<paper_name>_<date_time_stamp>.json` file on disk is the authoritative deliverable and must be written before the pipeline is reported complete.

---

## Input

```
[PASTE PAPER TEXT OR ABSTRACT HERE]
```
