# 🧠 ReproSchema Conversion Pipeline

This project uses `structsense` to automate the extraction, transformation, and validation of questionnaire content from PDF files into ReproSchema-compatible schema files. The goal is to standardize survey-based data collection workflows by producing machine-readable, interoperable, and reusable JSON-LD representations of assessments.

---

## 🔧 What This System Does

Given a questionnaire PDF, this pipeline:

1. **Extracts structured information** (e.g., items, response options, languages, scoring) from the raw document.
2. **Generates a ReproSchema-compatible folder structure**, including:
   - `items/`: individual question files (`item.jsonld`)
   - `activities/`: composite assessment (`activity.jsonld`)
   - `activity_schema.jsonld`: metadata and UI behavior
3. **Evaluates schema fidelity** against the original questionnaire.
4. **Applies human-in-the-loop feedback** to revise and improve schema quality.

---

## 🧑‍💼 Agents & Their Roles

| Agent Name           | Description |
|----------------------|-------------|
| `extractor_agent`    | Parses the questionnaire PDF and extracts questions, options, logic, and metadata. |
| `alignment_agent`    | Converts the extracted structure into a ReproSchema JSON-LD folder layout. |
| `judge_agent`        | Evaluates whether the generated schema matches the source content and highlights issues. |
| `humanfeedback_agent`| Applies human-provided feedback to revise and align the schema output more accurately. |

---

## 📁 Expected Output Structure

The final output is a directory that follows ReproSchema conventions:
```
protocol_name/
├── activities/
│   └── example_activity.jsonld
├── items/
│   ├── q1.jsonld
│   ├── q2.jsonld
│   └── …
├── activity_schema.jsonld
```

Each JSON-LD file conforms to [ReproSchema specifications](https://www.repronim.org/reproschema/), including proper use of:
- `@context`, `@id`, `@type`
- multilingual support (`skos:prefLabel`, `skos:altLabel`)
- input types, response options, scoring
- UI behavior and branching logic

---

## 🧠 Human-in-the-Loop

After the automated processing and evaluation steps, human feedback can be used to:
- Correct extraction mistakes
- Improve ontology mappings (e.g., add SSSOM-aligned terms)
- Clarify ambiguous response options or language labels
- Trigger regeneration of files with updated logic

---

## 🔍 Technologies Used

- **CrewAI** for multi-agent orchestration
- **ReproSchema** for schema-based assessment modeling
- **OpenRouter GPT-4o-mini** as the LLM backend
- **Ollama/Nomic** embeddings for memory and search
- **Local vector store** for knowledge integration (e.g., label/entity lookups)

---

## 📦 Configuration Files

- `agent_config`: Defines the roles, goals, and models for each AI agent
- `task_config`: Step-by-step logic for extracting, converting, validating, and revising schema data
- `embedder_config`: Text embedding model and backend for semantic memory
- `knowledge_config`: Defines searchable metadata keys (`entity`, `label`)
- `human_in_loop_config`: Activates the `humanfeedback_agent` for schema revision

---

## 🧪 Usage Tips

- The `{literature}` variable must be initialized with a questionnaire PDF.
- Outputs are JSON files organized in folders; ensure file writing permissions in your workspace.
- Best used with source questionnaires similar to PHQ-9, GAD-7, or eCOBIDAS formats.
- Consider mounting Git integration to track schema version control.

---

## 📚 References

- [ReproSchema Documentation](https://www.repronim.org/reproschema/)
- [CrewAI Docs](https://docs.crewai.com/)
- [LinkML](https://linkml.io)
- [FAIR Principles](https://www.go-fair.org/fair-principles/)
