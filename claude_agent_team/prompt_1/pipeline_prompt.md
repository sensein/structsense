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
2. Apply **entity masking** only if the first pass appears incomplete or the text is dense with overlapping concepts: replace all extracted entity spans with `[MASKED]` in a copy of the text, then re-analyze the masked text for missed entities.
3. Repeat masking passes only as needed, up to a **maximum of 3 rounds total**. Document the reason for each additional round taken.
4. For each entity, record:
   - `entity_text` — the literal span as it appears in the source
   - `entity_type` — the domain-appropriate type assigned by the extractor
   - `source_sentences` — list of full sentences where the entity appears
   - `indirect_references` — any sentences where the entity is referenced indirectly (e.g., via pronoun, abbreviation, or descriptive phrase); record both the referring phrase and the resolved entity
   - `extraction_round` — which pass (1, 2, or 3) the entity was first found in

**Output format (strict JSON):**

```json
{
  "agent": "EntityExtractor",
  "model_used": "...",
  "extraction_rounds_completed": 1,
  "masking_applied": false,
  "masking_rationale": "First pass appeared complete; text was not densely overlapping.",
  "entities": [
    {
      "entity_id": "E001",
      "entity_text": "CA1 pyramidal neurons",
      "entity_type": "CellType",
      "source_sentences": ["...full sentence..."],
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
- [ ] No entity spans are truncated, merged, or duplicated
- [ ] Every entity has at least one `source_sentence`
- [ ] Indirect references are resolved to a valid `entity_id`
- [ ] Ontology selections are appropriate for each entity type
- [ ] Ontology mappings exist for ≥ 80% of entities; unmapped entities are documented
- [ ] JSON outputs conform to the schemas above
- [ ] Masking rationale is documented if masking was skipped

**Iteration protocol:**

- If extraction gaps or schema violations are found, return specific feedback to the relevant agent and trigger a correction pass.
- Continue iterating until the reviewer certifies outputs as valid or confirms that remaining gaps are genuinely absent from the source text.
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

- **From Agent 1:** `entity_text`, `entity_type`, `source_sentences`, `indirect_references`, `extraction_round`
- **From Agent 2:** `ontology`, `ontology_id`, `ontology_label`, `confidence` (and `mapping_note` if non-trivial)

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