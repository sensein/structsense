# StructSense

`structsense` is a multi-agent system for extracting structured information from unstructured text and documents. It orchestrates a configurable pipeline of AI agents — extractor → alignment → judge → human feedback — each driven by a single YAML config file.

**Documentation:** [docs.brainkb.org](http://docs.brainkb.org/structsense_overview.html)
**License:** [Apache 2.0](LICENSE.txt)

---

## Using StructSense (CLI and Python)

### Command-line (CLI)

After installing (`pip install -e .`), the entry point is **`structsense-cli`**.

#### Full pipeline (extract)

Runs extraction → alignment → judge → optional human feedback and returns the final structured result.

```bash
structsense-cli extract \
  --config path/to/config.yaml \
  --source path/to/file.pdf \
  --env_file .env \
  --save_file result.json
```

| Option | Description                                                                          |
|--------|--------------------------------------------------------------------------------------|
| `--config` | **(Required)** Path to YAML config (agent + task + embedder).                        |
| `--source` | Path to a PDF, CSV, or TXT file to process. Mutually exclusive with `--source_text`. |
| `--source_text` | Raw text string to use as _. Mutually exclusive with `--source`.                     |
| `--api_key` | OpenRouter (or other) API key; can also be set in `.env` as `OPENROUTER_API_KEY`.    |
| `--env_file` | Path to `.env` (default: `.env` in current directory).                               |
| `--save_file` | Save the result JSON to this path.                                                   |
| `--enable_chunking` | Enable chunking for long documents (flag).                                           |
| `--chunk_size` | Chunk size in characters (e.g. `2000`); used when chunking is enabled.               |
| `--max_workers` | Max parallel workers for chunked extraction.                                         |
| `--downstream_max_input_chars` | Max input length for alignment/judge (default 80000).                                |
| `--max_extraction_chunk_chars` | Cap per-chunk size for extraction (default 25000).                                   |

**With OpenRouter (API key):**

```bash
structsense-cli extract \
  --source somefile.pdf \
  --api_key <YOUR_OPENROUTER_API_KEY> \
  --config someconfig.yaml \
  --env_file .env \
  --save_file result.json
```

**With Ollama (local, no API key):**

```bash
structsense-cli extract \
  --source somefile.pdf \
  --config someconfig.yaml \
  --env_file .env \
  --save_file result.json
```

**With chunking (recommended for long PDFs):**

```bash
structsense-cli extract \
  --config config.yaml \
  --source file.pdf \
  --enable_chunking \
  --chunk_size 2000 \
  --save_file result.json
```

#### Single agent–task (run-agent)

Run one agent and one task only (e.g. extractor only), without the full pipeline:

```bash
structsense-cli run-agent \
  --config path/to/config.yaml \
  --agent_key extractor_agent \
  --task_key extraction_task \
  --source path/to/file.pdf \
  --env_file .env \
  --save_file result.json
```

Use the same chunking/worker options as `extract` when needed.

---

### Python (programmatic)

Use **StructSenseFlow** as the single entry point. Run the **full pipeline** with `information_extraction_task()`, or a **single agent** with `kickoff(agent_key, task_key)` or `extraction()`.

**API key when running via Python:** For OpenRouter (or other cloud LLMs), either pass `api_key="your-key"` to `StructSenseFlow(...)` or set `OPENROUTER_API_KEY` in a `.env` file and pass `env_file=".env"`. The key is injected into the agent LLM config so all agents use it. Get an OpenRouter key at [openrouter.ai/keys](https://openrouter.ai/keys). If you get `401 User not found`, the key is missing or invalid.

#### Full pipeline (recommended)

```python
import asyncio
from structsense.app import StructSenseFlow

# Config can be paths to YAML files or dicts
flow = StructSenseFlow(
    agent_config="path/to/config.yaml",
    task_config="path/to/config.yaml",
    embedder_config="path/to/config.yaml",
    source="path/to/file.pdf",   # or source_text for raw text
    enable_chunking=True,
    chunk_size=2000,
    max_workers=8,
    env_file=".env",
    api_key=None,   # or set OPENROUTER_API_KEY in .env
)

# Run full pipeline: extraction → alignment → judge → human feedback (if enabled)
result = asyncio.run(flow.information_extraction_task())

# Result is a dict: entities, key_terms, resources, judged_terms, concept_mapping, etc.
print(result.get("task_type"), result.get("elapsed_time"))

# Save to file
import json
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
```

#### Single agent (one agent–task pair)

You can run **any** single agent–task pair with `kickoff(agent_key=..., task_key=...)`. For the extractor only, the convenience method is `extraction()`. For the **full pipeline** (extraction → alignment → judge → humanfeedback), use `information_extraction_task()`.

```python
import asyncio
from structsense.app import StructSenseFlow

flow = StructSenseFlow(
    agent_config="path/to/config.yaml",
    task_config="path/to/config.yaml",
    embedder_config="path/to/config.yaml",
    source="path/to/file.pdf",  # or source_text for raw text
    enable_chunking=True,
    chunk_size=2000,
)

# Run only the extractor (convenience method)
result = asyncio.run(flow.extraction())

# Or run any specific agent–task pair
result = asyncio.run(flow.kickoff(
    agent_key="extractor_agent",
    task_key="extraction_task",
))
# Other pairs: alignment_agent/alignment_task, judge_agent/judge_task,
# humanfeedback_agent/humanfeedback_task
```

**Note:** Alignment, judge, and humanfeedback tasks are designed to receive **output from the previous stage** when run in the full pipeline. When you run them alone via `kickoff(...)`, they receive the raw `source_text` as input (useful for debugging or custom flows).

#### Passing config as dicts

```python
import asyncio
import yaml
from structsense.app import StructSenseFlow

with open("ner-config.yaml") as f:
    all_config = yaml.safe_load(f)

flow = StructSenseFlow(
    agent_config=all_config["agent_config"],
    task_config=all_config["task_config"],
    embedder_config=all_config.get("embedder_config", {}),
    source="path/to/file.pdf",  # or source_text for raw text
    enable_chunking=True,
    chunk_size=2000,
    max_workers=8,
    env_file=".env",           # optional; loads OPENROUTER_API_KEY etc.
    api_key=None,              # or pass key here; injected into LLM config
)
result = asyncio.run(flow.information_extraction_task())

import json
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
```

---
## Configuration

Example configuration files and documentation on how to write them can be found in the [`config_template/`](config_template/) directory. It includes:

- `config.yaml` — a general configuration template
- `ner-config.yaml` — a configuration for NER tasks
- `resource-extraction-config.yaml` — a configuration for resource extraction tasks
- `pdf2_reproschema.yaml` — a configuration for converting survey PDFs to JSON-LD

See the [`config_template/readme.md`](config_template/readme.md) for details on agent, task, and embedder configuration options.

## Tutorials

This repository includes example tutorials demonstrating how to run StructSense:

- **[`tutorial/cli/`](tutorial/cli/)**
  Examples for running StructSense in **CLI mode**.

- **[`tutorial/python-example/`](tutorial/python-example/)**
  Tutorials demonstrating how to run StructSense in **programmatic (Python) mode**.

## Known Issues

<details>
<summary><strong>pip "resolution-too-deep" when installing <code>structsense</code></strong></summary>

**Symptom**

- During `pip install structsense` (or when it's a transitive dep), pip backtracks for a long time across many `opentelemetry-*` packages and eventually fails with a dependency resolution error.

**Resolution**

- `pip install --use-deprecated=legacy-resolver structsense`

</details>

<details>
<summary><strong>Python version</strong></summary>

**Symptom**

- `ERROR: Could not find a version that satisfies the requirement structsense (from versions: none) ERROR: No matching distribution found for structsense`

**Resolution**

- Your Python version should be `>=3.10,<3.13`.

</details>

<details>
<summary><strong>Agent execution traces</strong></summary>

**Symptom**

- The agent when running shows the execution trace prompt `Would you like to view your execution traces? [y/N] (20s timeout)`. For more see [How to disable execution trace prompt?](https://community.crewai.com/t/how-to-disable-execution-trace-prompt/7150) and [[BUG] How can disable tracing prompt? #3789](https://github.com/crewAIInc/crewAI/issues/3789).

**Resolution**
- Please set the following environment variables with provided values.

```bash
CREWAI_TRACING_ENABLED=false
CREWAI_DISABLE_TELEMETRY=true
CREWAI_DISABLE_TRACING=true
CREWAI_TELEMETRY=false
OTEL_SDK_DISABLED=true
ENABLE_CREW_MEMORY=false
```
</details>

<details>
<summary><strong>Agent Memory Issue</strong></summary>

**Symptom**

- You may get non-fatal error regarding agents and some discussion suggest disabling memory or provide Open AI key.

**Resolution**
- Please set the following environment variable with provided values to disable memory.

```bash
ENABLE_CREW_MEMORY=false
```
</details>

<details>
<summary><strong>Performance vs. Accuracy Trade-offs</strong></summary>

**Trade-off**

- Using smaller chunk sizes can improve extraction accuracy, but as chunk size decreases, processing time increases for the agent.
</details>

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Configuration](#configuration)
  - [Config file structure](#config-file-structure)
  - [Task types](#task-types)
  - [Chunking for large documents](#chunking-for-large-documents)
- [Concept Mapping](#concept-mapping)
  - [BioPortal (default)](#bioportal-default)
  - [Local service](#local-service)
- [Environment Variables](#environment-variables)
- [Examples](#examples)

---

## Features

- **Multi-agent pipeline** — extraction, ontology alignment, quality judging, and optional human-in-the-loop feedback, all in one command
- **Task-type auto-detection** — detects NER, resource extraction, or structured extraction from your config; detected once and reused across all pipeline stages
- **Chunking** — splits large PDFs into sentence-aligned chunks and runs extraction in parallel
- **Pluggable concept mapping** — BioPortal (cloud) or a local hybrid BM25 + dense retrieval service, switchable via env var
- **Any LLM via OpenRouter** — configure model per agent in YAML
- **Single config file** — one YAML drives the entire pipeline

---

## Installation

```bash
pip install structsense
```

Requires Python 3.10–3.12.

---

## Quick Start

### CLI

**Full pipeline** (extraction → alignment → judge):

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --api_key sk-or-v1-...
```

**With chunking** (recommended for large PDFs):

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --enable_chunking \
  --chunk_size 2000 \
  --max_workers 8 \
  --save_file result.json \
  --api_key sk-or-v1-...
```

**Run a single agent/task directly:**

```bash
structsense-cli run-agent \
  --config ner-config.yaml \
  --agent_key extractor_agent \
  --task_key extraction_task \
  --source "Hippocampal neurons in CA1 were recorded during spatial navigation." \
  --api_key sk-or-v1-...
```

**Save output to file:**

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --save_file output.json \
  --api_key sk-or-v1-...
```

### Python API

```python
import asyncio, json, yaml
from structsense.app import StructSenseFlow

with open("ner-config.yaml") as f:
    cfg = yaml.safe_load(f)

flow = StructSenseFlow(
    agent_config=cfg["agent_config"],
    task_config=cfg["task_config"],
    embedder_config=cfg.get("embedder_config", {}),
    input_source="paper.pdf",
    enable_chunking=True,
    chunk_size=2000,
    max_workers=8,
    api_key="sk-or-v1-...",
)

result = asyncio.run(flow.information_extraction_task())

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
```

---

## Configuration

### Config file structure

All pipeline settings live in a single YAML file:

```yaml
agent_config:
  extractor_agent:
    role: >
      Neuroscience NER Extractor Agent
    goal: >
      Extract named entities and key terms from {input_text}. Return structured JSON.
    backstory: >
      You are an AI assistant for neuroscience NER. Output strict JSON.
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

  alignment_agent:
    role: >
      Neuroscience NER Concept Alignment Agent
    goal: >
      Map entities in {extracted_structured_information} to ontologies. Add ontology_id,
      ontology_label, ontology, and concept_mapping_provenance.
    backstory: >
      You align extracted terms to ontologies (CL, UBERON, NCBITaxon). Use the Concept
      Mapping Tool. Set concept_mapping_provenance to "tool" or "llm_knowledge".
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

  judge_agent:
    role: >
      Neuroscience NER Judge Agent
    goal: >
      Extend {aligned_structured_information} with judge_score (0–1) and remarks.
    backstory: >
      You evaluate alignment quality. Do not remove existing fields. Output strict JSON.
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

task_config:
  extraction_task:
    description: >
      Extract entities and key_terms from {input_text}. Return JSON with
      entities (entity, label, sentence, start, end, paper_title, doi) and key_terms.
    expected_output: >
      JSON: { "entities": [...], "key_terms": [...] }
    agent_id: extractor_agent

  alignment_task:
    description: >
      Map each entity/key_term from {extracted_structured_information} to an ontology.
      Add ontology_id, ontology_label, ontology, concept_mapping_provenance.
    expected_output: >
      Same structure as input with ontology fields added.
    agent_id: alignment_agent

  judge_task:
    description: >
      Evaluate {aligned_structured_information}. Add judge_score and remarks per item.
    expected_output: >
      Same structure as input with judge_score and remarks added.
    agent_id: judge_agent

embedder_config:
  provider: ollama
  config:
    api_base: http://localhost:11434
    model: nomic-embed-text
```

Ready-to-use config templates are in [config_template/](config_template/).

### Task types

The pipeline auto-detects task type from your config description. The three supported types are:

| Task type | Detected when | Output keys |
|---|---|---|
| `ner` | config mentions `entity`, `named entity`, `ner` | `entities`, `key_terms` |
| `resource` | config mentions `resource` + extraction-related terms | `resources` |
| `structured_extraction` | config mentions `structured extraction` | raw pass-through |

Task type is detected **once** at the extraction stage and reused for all downstream stages.

### Chunking for large documents

For large PDFs, enable chunking to split the text into sentence-aligned chunks and run extraction in parallel:

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source large_paper.pdf \
  --enable_chunking \
  --chunk_size 2000 \
  --max_workers 8
```

Or in Python:

```python
flow = StructSenseFlow(
    ...
    enable_chunking=True,
    chunk_size=2000,       # characters per chunk
    max_workers=8,         # parallel workers
    max_extraction_chunk_chars=25000,   # cap chunk size for model context
    downstream_max_input_chars=80000,   # cap input to alignment/judge/humanfeedback
)
```

---

## Concept Mapping

The alignment agent uses a Concept Mapping Tool to map extracted terms to ontology IRIs and labels. Two backends are available, switchable via the `CONCEPT_MAPPING_BACKEND` environment variable.

### Local service (default)

It uses an in-house Ontology Concept Mapping service that combines hybrid **BM25** and **dense retrieval**, enhanced with re-ranking for improved accuracy.

All requests are processed concurrently via the `POST /map/batch` endpoint.

To use this feature, ensure the [concept mapping service](https://github.com/sensein/search_hybrid) is running locally.

```bash
CONCEPT_MAPPING_BACKEND=local   # default — can be omitted
LOCAL_CONCEPT_MAPPING_URL=http://localhost:8000
```

| Variable                                   | Default | Description                                                                                        |
|--------------------------------------------|---|----------------------------------------------------------------------------------------------------|
| `LOCAL_CONCEPT_MAPPING_URL`                | `http://localhost:8000` | Base URL of the local service                                                                      |
| `LOCAL_CONCEPT_MAPPING_API_KEY` (Optional) | — | API/OpenRouter key for LLM re-ranking (falls back to `OPENROUTER_API_KEY`). Note this is optional. |
| `LOCAL_CONCEPT_MAPPING_MODEL`  (Optional)      | — | OpenRouter model for LLM re-ranking (falls back to `OPENROUTER_MODEL`)                             |
| `LOCAL_CONCEPT_MAPPING_TIMEOUT`            | `30` | Request timeout in seconds                                                                         |
| `MAX_CONCEPT_MAPPING_RESULTS`              | `1` | Results per term (1–20)                                                                            |

 
### BioPortal

Uses the [BioPortal](https://bioportal.bioontology.org/) REST API for ontology lookup with automatic ontology detection.

```bash
CONCEPT_MAPPING_BACKEND=bioportal
BIOPORTAL_API_KEY=your-key-here
```

Get a free API key at [bioportal.bioontology.org/account](https://bioportal.bioontology.org/account).

Optional tuning:

| Variable | Default | Description |
|---|---|---|
| `BIOPORTAL_REQUEST_INTERVAL` | `0.7` | Seconds between requests (increase to avoid 429s) |
| `BIOPORTAL_BACKOFF_AFTER_429` | `2.0` | Retry backoff in seconds after a 429 |
| `MAX_CONCEPT_MAPPING_RESULTS` | `1` | Results per term (1–20) |
| `CONCEPT_MAPPING_CACHE_SIZE` | `2000` | In-memory cache entries |

**Switching backends** is a one-line change — the output format is identical so no pipeline changes are needed.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM calls |
| `ENABLE_HUMAN_FEEDBACK` | `true`/`false` — enable human-in-the-loop feedback stage |
| `ENABLE_CREW_MEMORY` | `true`/`false` — enable CrewAI long/short/entity memory (requires embedder) |
| `CONCEPT_MAPPING_BACKEND` | `bioportal` (default) or `local` |
| `BIOPORTAL_API_KEY` | Required when `CONCEPT_MAPPING_BACKEND=bioportal` |
| `LOCAL_CONCEPT_MAPPING_URL` | Local service URL (default `http://localhost:8000`) |
| `LOCAL_CONCEPT_MAPPING_API_KEY` | API key for local service LLM re-ranking |
| `MAX_CONCEPT_MAPPING_RESULTS` | Concept mapping results per term (default `1`) |

Store these in a `.env` file and pass it with `--env_file .env` (CLI) or `env_file=".env"` (Python).

---

## Examples

Ready-to-run examples are in [example/](example/):

| Example | Description |
|---|---|
| [NER_EXAMPLE_OPENROUTER/](example/NER_EXAMPLE_OPENROUTER/) | Named entity recognition from neuroscience text using OpenRouter |
| [resource_extraction/](example/resource_extraction/) | BBQS resource extraction (tools, datasets, models, benchmarks) |
| [pdf2_reproschema/](example/pdf2_reproschema/) | Structured extraction into ReproSchema format |

Python tutorial with full pipeline and extraction-only examples: [tutorial/python-example/](tutorial/python-example/)
