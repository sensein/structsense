# StructSense

`structsense` is a multi-agent system for extracting structured information from unstructured text and documents. It orchestrates a configurable pipeline of AI agents — extractor → alignment → judge → human feedback — each driven by a single YAML config file.

**Documentation:** [docs.brainkb.org](http://docs.brainkb.org/structsense_overview.html)
**License:** [Apache 2.0](LICENSE.txt)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python](#python)
- [Configuration](#configuration)
  - [Config file structure](#config-file-structure)
  - [Task types](#task-types) 
- [Pipeline Options](#pipeline-options)
  - [Skip pipeline stages](#skip-pipeline-stages)
  - [Alignment options](#alignment-options)
  - [Judge options](#judge-options)
  - [Resume from a saved stage](#resume-from-a-saved-stage)
  - [Agent execution controls](#agent-execution-controls)
- [Concept Mapping](#concept-mapping)
- [Environment Variables](#environment-variables)
- [Human Feedback](#human-feedback)
- [Examples & Tutorials](#examples--tutorials)
- [Known Issues](#known-issues)

---

## Features

- **Multi-agent pipeline** — extraction, ontology alignment, quality judging, and optional human-in-the-loop feedback, all in one command
- **Task-type auto-detection** — detects NER, resource extraction, or structured extraction from your config; applied consistently across all pipeline stages
- **Chunking** — splits large PDFs into sentence-aligned chunks and runs extraction in parallel; downstream stages split automatically based on model context window
- **Fast alignment** — skips the alignment LLM entirely for local concept mapping; calls the concept mapping tool directly in batch (~seconds vs ~60 min)
- **Pluggable concept mapping** — BioPortal (cloud) or a local hybrid BM25 + dense retrieval service, switchable via env var
- **Partial pipeline** — run any subset of stages; combine `--skip_stage` with `--preload_stage` to resume from any checkpoint
- **Any LLM via OpenRouter** — configure model per agent in YAML
- **Single config file** — one YAML drives the entire pipeline

---

## Installation

```bash
pip install structsense
```

Requires Python 3.10–3.12.

> **Tip — dependency resolution error:** If pip fails with a "resolution-too-deep" error on `opentelemetry-*` packages, use:
> ```bash
> pip install --use-deprecated=legacy-resolver structsense
> ```

---

## Quick Start

### CLI

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --api_key sk-or-v1-... \
  --save_file result.json
```

**With chunking** (recommended for large inputs):

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --enable_chunking \
  --chunk_size 600 \
  --max_workers 8 \
  --save_file result.json \
  --api_key sk-or-v1-...
```

### Python

```python
import asyncio, json, yaml
from structsense.app import StructSenseFlow

# read the config file
with open("ner-config.yaml") as f:
    cfg = yaml.safe_load(f)

# initialize and run StructSense
flow = StructSenseFlow(
    agent_config=cfg["agent_config"],
    task_config=cfg["task_config"],
    embedder_config=cfg.get("embedder_config", {}),
    source="paper.pdf",
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

## Advanced Usage

### Using CLI
 
#### Full pipeline

Runs extraction → alignment → judge → optional human feedback and returns the final structured result.

```bash
structsense-cli extract \
  --config path/to/config.yaml \
  --source path/to/file.pdf \
  --env_file .env \
  --save_file result.json
```

| Option | Description                                                                                                                                                                                                                                                        |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config` | **(Required)** Path to YAML config.                                                                                                                                                                                                                                |
| `--source` | Path to a PDF, CSV, or TXT file. Mutually exclusive with `--source_text`.                                                                                                                                                                                          |
| `--source_text` | Raw text string. Mutually exclusive with `--source`.                                                                                                                                                                                                               |
| `--api_key` | OpenRouter API key; can also be set in `.env` as `OPENROUTER_API_KEY`.                                                                                                                                                                                             |
| `--env_file` | Path to `.env` (default: `.env` in current directory).                                                                                                                                                                                                             |
| `--save_file` | Save result JSON to this path.                                                                                                                                                                                                                                     |
| `--enable_chunking` | Enable chunking for long documents (flag).                                                                                                                                                                                                                         |
| `--chunk_size` | Chunk size in characters (e.g. `2000`).                                                                                                                                                                                                                            |
| `--max_workers` | Max parallel workers for chunked extraction.                                                                                                                                                                                                                       |
| `--skip_alignment_llm` | `auto`/`true`/`false` — bypass alignment LLM.                                                                                                                                                                                                                      |
| `--skip_judge_llm` | `true`/`false` — bypass judge LLM, inject default scores.                                                                                                                                                                                                          |
| `--skip_stage` | Omit a pipeline stage (repeatable). Note, while  `--skip_alignment_llm` and `--skip_judge_llm` allows you to skip individual agent, here you can specify multiple agents to skip (example below).                                                                  |
| `--preload_stage` | Load a saved stage output instead of running it (repeatable).                                                                                                                                                                                                      |
| `--agent_max_iter` | **Maximum iterations per task (`max_iter`).** Limits the number of iterations a task can execute to prevent infinite loops. Defaults to `20` in our case. For more information, see the Crew.ai documentation: https://docs.crewai.com/en/learn/customizing-agents |
| `--agent_max_execution_time` | **Maximum wall-clock time per agent run (in seconds).** This value is passed to the agent’s `max_execution_time` setting in Crew.ai. For more information, see the Crew.ai documentation: https://docs.crewai.com/en/learn/customizing-agents                      |
| `--agent_max_retry_limit` |**Maximum agent retries on errors (`max_retry_limit`).** Sets the maximum number of retry attempts for an agent when errors occur. Defaults to `5`. For more information, see the Crew.ai documentation: https://docs.crewai.com/en/learn/customizing-agents                                                                                                                                                                                                                                 |
| `--model_context_window` | Override auto-detected context window in tokens.                                                                                                                                                                                                                   |
| `--downstream_max_input_chars` | Max input length for alignment/judge (default 80000).                                                                                                                                                                                                              |
| `--downstream_chunk_size` | Entities per chunk for downstream stages (auto if omitted).                                                                                                                                                                                                        |

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

**Note on using Ollama/other providers:**

To use `StructSense` with Ollama, update your configuration so it matches the format expected by CrewAI.

For example, when using OpenRouter, you would set the model as `openrouter/<model-name>` and configure `base_url` to point to the OpenRouter API.

Similarly, for Ollama, set the model as `ollama/<model-name>` and use:

`base_url=http://localhost:11434`

This is the default Ollama local endpoint, unless you changed it during installation or configuration. As an example, you can refer to the config template directory, where Ollama is used for embeddings.

To learn more about provider prefixes and configuration formats, see:
[https://docs.crewai.com/en/learn/llm-connections](https://docs.crewai.com/en/learn/llm-connections)

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

**API key:** Pass `api_key="your-key"` or set `OPENROUTER_API_KEY` in `.env`. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys). If you see `401 User not found`, the key is missing or invalid.

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

#### Config as dicts

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

Example config files are in [`config_template/`](config_template/). See [`config_template/readme.md`](config_template/readme.md) for full details.

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
      Map entities in {extracted_structured_information} to ontologies.
    backstory: >
      You align extracted terms to ontologies. Use the Concept Mapping Tool.
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

  judge_agent:
    role: >
      Neuroscience NER Judge Agent
    goal: >
      Extend {aligned_structured_information} with judge_score (0–1) and remarks.
    backstory: >
      You evaluate alignment quality. Do not remove existing fields.
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

task_config:
  extraction_task:
    description: >
      Extract entities and key_terms from {input_text}.
    expected_output: >
      JSON: { "entities": [...], "key_terms": [...] }
    agent_id: extractor_agent

  alignment_task:
    description: >
      Map each entity from {extracted_structured_information} to an ontology.
    expected_output: >
      Same structure with ontology fields added.
    agent_id: alignment_agent

  judge_task:
    description: >
      Evaluate {aligned_structured_information}. Add judge_score and remarks.
    expected_output: >
      Same structure with judge_score and remarks added.
    agent_id: judge_agent

embedder_config:
  provider: ollama
  config:
    api_base: http://localhost:11434
    model: nomic-embed-text
```

### Task types

The pipeline auto-detects the task type from your config description:

| Task type | Detected when config mentions | Output keys |
|---|---|---|
| `ner` | `entity`, `named entity`, `ner` | `entities`, `key_terms` |
| `resource` | `resource` + extraction-related terms | `resources` |
| `structured_extraction` / `generic` | `structured extraction` or other | task-specific keys |

Task type is detected **once** at extraction and reused for all downstream stages.

Ready-to-use configs:
- `ner-config.yaml` — named entity recognition
- `resource-extraction-config.yaml` — tool/dataset/model/benchmark extraction
- `pdf2_reproschema.yaml` — structured extraction into ReproSchema JSON-LD

---




## Skip Pipeline Stages

You can skip or bypass stages three ways: CLI flags, environment variables, or Python parameters.

---

### `--skip_stage` — omit entire stages

Use `--skip_stage` to remove one or more stages from the pipeline entirely. When skipped, the previous stage's output is forwarded directly to the next non-skipped stage.

```bash
# Extraction + alignment only — skip judge and human feedback
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_stage judge_task \
  --skip_stage humanfeedback_task \
  --save_file result.json
```

```python
flow = StructSenseFlow(
    ...,
    skip_stages=["judge_task", "humanfeedback_task"],
)
```

Via env var (comma-separated):
```bash
SKIP_STAGES=judge_task,humanfeedback_task
```

| Stage | `task_key` |
|---|---|
| Alignment | `alignment_task` |
| Judge | `judge_task` |
| Human feedback | `humanfeedback_task` |

> **Note:** `extraction_task` cannot be used with `--skip_stage`. Extraction is always the
> first stage. To skip it, use `--preload_stage extraction_task:<file.json>` to load a
> previously saved extraction result instead.

---

### Skipping crewai-based alignment

Running the alignment task through CrewAI can be costly and time-consuming, especially for large inputs, where execution may take more than 6 hours. This option lets you bypass the CrewAI-based alignment step and use the non-CrewAI alignment approach instead.

By default (`skip_alignment_llm=None`), the alignment LLM is **automatically bypassed** when both of the following conditions are true:

- `CONCEPT_MAPPING_BACKEND=local` (which is the default)
- Task type is `ner`, `keyphrase_extraction`, `resource`, or `structured_extraction`

When bypassed, the concept mapping tool is called directly from Python in one batch (4000 concept/request--see [https://github.com/sensein/search_hybrid/tree/dev](https://github.com/sensein/search_hybrid/tree/dev)) — much
faster than running the LLM. The output records `alignment_method: "direct_tool_call"`.

```bash
# .env
# Auto is the default — no variable needed if using local backend
CONCEPT_MAPPING_BACKEND=local

SKIP_ALIGNMENT_LLM=true   # force bypass regardless of backend or task type
# SKIP_ALIGNMENT_LLM=false  # force the alignment LLM even when local backend is active
# SKIP_ALIGNMENT_LLM=auto   # same as omitting the variable (default behavior)
```

```bash
# CLI
structsense-cli extract --config ner-config.yaml --source paper.pdf \
  --skip_alignment_llm true   # force bypass
```

```python
flow = StructSenseFlow(..., skip_alignment_llm=None)   # auto (default)
# skip_alignment_llm=True  → always bypass
# skip_alignment_llm=False → always run alignment LLM
```

| Value | Behaviour |
|---|---|
| `None` / `auto` (default) | Bypass when `CONCEPT_MAPPING_BACKEND=local` **and** task type is `ner`, `keyphrase_extraction`, `resource`, or `structured_extraction` |
| `True` | Always bypass — direct tool call, alignment LLM never called |
| `False` | Always run the alignment LLM regardless of backend or task type |

---

### Judge stage options

There are two independent settings for the judge stage:

| Setting | What it controls |
|---|---|
| `skip_judge_llm` | Whether the judge runs **at all** |
| `direct_judge_api` | Whether to use CrewAI-based agent for  judge task or not for the same reason as alignment agent (see above).|

#### Skip the judge entirely (`skip_judge_llm`)

When `skip_judge_llm=True`, no LLM call is made. Every entity is automatically stamped
with `judge_score=1.0` and `remarks="auto-approved"`, and `judge_method: "auto_approved"`
is recorded in the output.

Use this when you trust the alignment output and do not need per-entity quality scoring.

```bash
# .env
SKIP_JUDGE_LLM=true
```

```bash
# CLI
structsense-cli extract --config ner-config.yaml --source paper.pdf --skip_judge_llm true
```

```python
flow = StructSenseFlow(..., skip_judge_llm=True)
```

| Value | Behaviour |
|---|---|
| `False` / `None` (default) | Run judge |
| `True` | No LLM call — all entities receive `judge_score=1.0`, `remarks="auto-approved"` |
 
#### Run the custom judge agent or through CrewAI (`direct_judge_api`)

When `direct_judge_api=True` (default), the judge LLM is still used, but it does **not** run through the CrewAI agent loop. Instead, StructSense uses a custom implementation that calls the LLM directly through `AsyncOpenAI` in parallel batches with retry support.

This avoids the overhead of the CrewAI-based judge flow, which can trigger more LLM calls than necessary and become very expensive for large inputs. In our testing, that overhead could sometimes push runtime beyond 6 hours, making it impractical in both time and cost.

Using `direct_judge_api=True` is therefore significantly faster and more efficient for large documents.

Set `direct_judge_api=False` only if you need the full CrewAI ReAct agent behavior.

```bash
# .env
DIRECT_JUDGE_API=false   # revert to CrewAI agent
```

```python
flow = StructSenseFlow(..., direct_judge_api=False)
```

| Value | Behaviour |
|---|---|
| `True` (default) | Direct `AsyncOpenAI` call — fast, parallel, no CrewAI overhead |
| `False` | Full CrewAI judge agent |

The same pattern applies to the humanfeedback stage via `direct_humanfeedback_api` /
`DIRECT_HUMANFEEDBACK_API` (default `True`).


## Resume from a saved stage

If the pipeline crashes after extraction, use `--preload_stage` to skip already-completed stages and load their saved output instead.

Stage output files are written automatically when `stage_output_dir` is set, named:
```
00_extractor_agent_extraction_task.json
01_alignment_agent_alignment_task.json
02_judge_agent_judge_task.json
```

```bash
# Skip extraction; re-run from alignment
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --preload_stage extraction_task:00_extractor_agent_extraction_task.json \
  --save_file result.json
```

```python
result = asyncio.run(
    flow.information_extraction_task(
        preloaded_stages={"extraction_task": extraction_result}
    )
)
```

You can preload multiple stages. `--source` / `--source_text` is still required even when all upstream stages are preloaded.

*Preloading multiple stages:*

```bash
structsense-cli extract --env_file=.env --save_file=output.json --chunk_size=2000 --max_workers=8 --enable_chunking --config=some-config.yaml --source=vitpose.pdf --preload_stage extraction_task:00_extractor_agent_extraction_task.json --preload_stage alignment_task:01_alignment_agent_alignment_task.json --api_key=sk-or-v 
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

Store these in a `.env` file and pass with `--env_file .env` (CLI) or `env_file=".env"` (Python).

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM calls |
| `ENABLE_HUMAN_FEEDBACK` | `true`/`false` — enable human-in-the-loop feedback stage |
| `ENABLE_CREW_MEMORY` | `true`/`false` — enable CrewAI memory (requires embedder) |
| `CONCEPT_MAPPING_BACKEND` | `local` (default) or `bioportal` |
| `BIOPORTAL_API_KEY` | Required when using BioPortal backend |
| `LOCAL_CONCEPT_MAPPING_URL` | Local service URL (default `http://localhost:8000`) |
| `LOCAL_CONCEPT_MAPPING_API_KEY` | API key for local service LLM re-ranking |
| `MAX_CONCEPT_MAPPING_RESULTS` | Results per term (default `1`) |
| `SKIP_ALIGNMENT_LLM` | `auto`/`true`/`false` — bypass alignment LLM |
| `SKIP_JUDGE_LLM` | `true`/`false` — bypass judge LLM, inject default scores |
| `SKIP_STAGES` | Comma-separated task keys to omit, e.g. `judge_task,humanfeedback_task` |
| `AGENT_MAX_ITER` | Max reasoning iterations per agent (CrewAI default 20) |
| `AGENT_MAX_EXECUTION_TIME` | Max wall-clock seconds per agent run (default 30) |
| `AGENT_MAX_RETRY_LIMIT` | Max agent-level retries on errors (default 0) |
| `DIRECT_JUDGE_API` | `true`/`false` — use direct API calls for judge stage (default `true`) |
| `DIRECT_HUMANFEEDBACK_API` | `true`/`false` — use direct API calls for humanfeedback stage (default `true`) |


### Agent execution controls

Control how long each agent may work and how many times it retries.

```bash
# .env
AGENT_MAX_ITER=5
AGENT_MAX_EXECUTION_TIME=60   # seconds
AGENT_MAX_RETRY_LIMIT=1
```

```bash
# CLI
structsense-cli extract ... --agent_max_iter 5 --agent_max_execution_time 60
```

```python
flow = StructSenseFlow(..., agent_max_iter=5, agent_max_execution_time=60, agent_max_retry_limit=1)
```

| Parameter | Default | Notes |
|---|---|---|
| `agent_max_iter` | 20 (CrewAI default) | Lower = faster/cheaper; raise for complex tasks |
| `agent_max_execution_time` | 30 s | Raise for slow models or complex tasks |
| `agent_max_retry_limit` | 0 (fail fast) | Set to 1–3 to allow retries on tool/parse errors |

---

## Human Feedback

Enable the human-in-the-loop feedback stage by setting `ENABLE_HUMAN_FEEDBACK=true`. After the judge stage, the pipeline pauses and presents a menu:

```
1. Approve and continue
2. Abort pipeline
3. Open editor to provide feedback
4. Skip feedback for this step
```

Choosing option **3** opens your default terminal editor with the feedback area at the top of the file. Replace `[WRITE YOUR FEEDBACK HERE]` with your feedback text, then save and close. The current output JSON is shown below as a read-only reference (commented out). Closing the editor without writing anything returns you to the menu.

```bash
# .env
ENABLE_HUMAN_FEEDBACK=true
```

---


## Examples & Tutorials

Ready-to-run examples are in [example/](example/):

| Example | Description |
|---|---|
| [NER_EXAMPLE_OPENROUTER/](example/NER_EXAMPLE_OPENROUTER/) | Named entity recognition from neuroscience text using OpenRouter |
| [resource_extraction/](example/resource_extraction/) | BBQS resource extraction (tools, datasets, models, benchmarks) |
| [pdf2_reproschema/](example/pdf2_reproschema/) | Structured extraction into ReproSchema format |

Step-by-step tutorials: [tutorial/](tutorial/)
- CLI examples: [tutorial/cli/](tutorial/cli/)
- Python examples: [tutorial/python-example/](tutorial/python-example/)

---


## Known Issues

<details>
<summary><strong>pip "resolution-too-deep" when installing structsense</strong></summary>

**Symptom:** pip backtracks across many `opentelemetry-*` packages and fails.

**Fix:**
```bash
pip install --use-deprecated=legacy-resolver structsense
```
</details>

<details>
<summary><strong>Python version</strong></summary>

**Symptom:** `No matching distribution found for structsense`

**Fix:** Use Python `>=3.10,<3.13`.
</details>

<details>
<summary><strong>Agent execution trace prompt</strong></summary>

**Symptom:** Agent shows `Would you like to view your execution traces? [y/N] (20s timeout)`

**Fix:** Add to `.env`:
```bash
CREWAI_TRACING_ENABLED=false
CREWAI_DISABLE_TELEMETRY=true
CREWAI_DISABLE_TRACING=true
CREWAI_TELEMETRY=false
OTEL_SDK_DISABLED=true
ENABLE_CREW_MEMORY=false
```
or use
```bash
export CREWAI_TRACING_ENABLED=false \
CREWAI_DISABLE_TELEMETRY=true \
CREWAI_DISABLE_TRACING=true \
CREWAI_TELEMETRY=false \
OTEL_SDK_DISABLED=true
```
</details>

<details>
<summary><strong>Agent memory errors</strong></summary>

**Symptom:** Non-fatal errors about agent memory.

**Fix:**
```bash
ENABLE_CREW_MEMORY=false
```
</details>

<details>
<summary><strong>Performance vs. accuracy trade-offs</strong></summary>

Smaller chunk sizes improve extraction accuracy but increase processing time.
</details>
