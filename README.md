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
| `--downstream_chunk_size` | Entities per chunk for parallel alignment/judge/humanfeedback (auto if omitted).     |
| `--max_extraction_chunk_chars` | Cap per-chunk size for extraction (default 25000).                                   |
| `--model_context_window` | Override auto-detected context window in tokens (e.g. `200000`). Use for unknown/proxy models. |
| `--skip_alignment_llm` | `auto`/`true`/`false` — bypass alignment LLM. See [Fast Alignment](#fast-alignment-skip-the-alignment-llm). Also settable via `SKIP_ALIGNMENT_LLM` env var. |
| `--skip_judge_llm` | `true`/`false` — bypass judge LLM, inject default scores. See [Fast Judge](#fast-judge-skip-the-judge-llm). Also settable via `SKIP_JUDGE_LLM` env var. |
| `--skip_stage` | Omit a pipeline stage entirely (repeatable). See [Skipping Pipeline Stages](#skipping-pipeline-stages). Also settable via `SKIP_STAGES` env var. |
| `--preload_stage` | Skip a stage by loading saved output from a JSON file (repeatable). See [Resuming a Pipeline](#resuming-a-pipeline-from-a-saved-stage). |

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

  
### Fast Alignment (bypass the alignment LLM)

**What the alignment stage normally does:** After extraction, the alignment agent calls a concept mapping tool (e.g. the local hybrid BM25 + dense retrieval service) to look up ontology IDs and labels for every extracted entity or resource name. The agent is an LLM whose job is to orchestrate those tool calls — but in practice LLMs call the tool for only a handful of terms and stop, leaving most entities unmapped.

**What fast alignment does instead:** StructSense calls the concept mapping tool directly from Python, in one batch containing every extracted term, then injects the results (`ontology_id`, `ontology_label`, `ontology`) into the entity or resource dicts. The alignment LLM is never invoked. This reduces alignment time from tens of minutes to a few seconds for typical documents.

The local service supports up to 4000 terms per batch request by default (configurable — see [search_hybrid](https://github.com/sensein/search_hybrid)).

**When it applies:**
- For **NER and keyphrase** tasks: entity texts are sent to the tool.
- For **resource and structured_extraction** tasks: resource `name` fields are sent to the tool.
- In both cases the same ontology fields are injected and `alignment_method: "direct_tool_call"` is recorded in the output.

**Auto-enable:** Fast alignment is switched on automatically when `CONCEPT_MAPPING_BACKEND=local` and the task type is one of the above. No flag is needed. You can override this with `SKIP_ALIGNMENT_LLM=false` to force the LLM back on, or `SKIP_ALIGNMENT_LLM=true` to force the bypass regardless of backend.

#### .env (recommended — no CLI flags needed)

```bash
CONCEPT_MAPPING_BACKEND=local   # auto-enables fast alignment for NER/resource tasks
# SKIP_ALIGNMENT_LLM=true       # force bypass regardless of task type or backend
# SKIP_ALIGNMENT_LLM=false      # force the alignment LLM even on local backend
```

#### CLI

```bash
# Auto (default — bypass fires automatically when local backend + NER/resource)
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf

# Force bypass for any task type / backend
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_alignment_llm true

# Force the alignment LLM even when local backend is active
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_alignment_llm false
```

#### Python

```python
flow = StructSenseFlow(
    ...
    skip_alignment_llm=None,   # auto (default) — bypass when local backend + NER/resource
    # skip_alignment_llm=True  # always bypass
    # skip_alignment_llm=False # always run alignment LLM
)
```

| `SKIP_ALIGNMENT_LLM` / `skip_alignment_llm` | Behaviour |
|---|---|
| `auto` / `None` (default) | Bypass when `CONCEPT_MAPPING_BACKEND=local` and task is NER, keyphrase, resource, or structured_extraction |
| `true` / `True` | Always bypass — direct tool call only, no LLM |
| `false` / `False` | Always run the alignment LLM |

---

### Skip the Judge LLM

**What the judge stage normally does:** The judge agent reviews every entity produced by the alignment stage and adds a `judge_score` (float 0–1) and `remarks` field evaluating the quality of the extraction and alignment. The judge **always runs by default** whenever `judge_task` is present in the config.

**What skipping the judge does:** When `SKIP_JUDGE_LLM=true`, the judge LLM is not called. Instead, StructSense copies the alignment output directly and stamps every entity with `judge_score=1.0` and `remarks="auto-approved"`. This is appropriate when you trust the alignment output and do not need quality scoring, or when you want the fastest possible run.

> **When to use this:** Speed-focused pipelines where LLM-as-judge overhead is not acceptable, or when you are post-processing results programmatically and do not need per-entity quality scores.

#### .env (recommended)

```bash
SKIP_JUDGE_LLM=true
```

#### CLI

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_judge_llm true
```

#### Python

```python
flow = StructSenseFlow(..., skip_judge_llm=True)
```

| `SKIP_JUDGE_LLM` / `skip_judge_llm` | Behaviour |
|---|---|
| `false` / `None` / `False` (default) | Judge LLM runs normally — entities receive scored `judge_score` and `remarks` |
| `true` / `True` | Judge LLM is skipped — all entities receive `judge_score=1.0` and `remarks="auto-approved"` |

When skipped, `judge_method: "auto_approved"` and `judge_llm_skipped: true` are recorded in the stage output.

---

### Fastest possible run (zero post-extraction LLM calls)

If you only need extracted and ontology-mapped entities, and do not require scored quality judgements, set these three variables in your `.env` and run the pipeline as normal — no CLI flags needed:

```bash
# .env
CONCEPT_MAPPING_BACKEND=local
SKIP_ALIGNMENT_LLM=true   # replace alignment LLM with direct batch tool call
SKIP_JUDGE_LLM=true       # replace judge LLM with default scores
```

```bash
structsense-cli extract --config ner-config.yaml --source paper.pdf --save_file result.json
```

The pipeline becomes:

```
extraction (LLM)  →  concept mapping tool call (~seconds)  →  default judge scores  →  done
```

Only the extraction LLM runs. Alignment and judge complete in milliseconds.

---

## Skipping Pipeline Stages

Use `--skip_stage` (CLI) or `skip_stages=` (Python) to omit specific pipeline stages. The previous stage's output is passed directly to the next non-skipped stage — no intermediate result is needed.

This is useful when you want a fast partial run, e.g. just extraction + alignment, without judge overhead.

### Which stages can be skipped?

| Stage to skip | `task_key` | Notes |
|---|---|---|
| Alignment | `alignment_task` | Output of extraction is passed directly to judge |
| Judge | `judge_task` | Output of alignment is the final result |
| Human feedback | `humanfeedback_task` | Same as leaving `enable_human_feedback=False` (the default) |
| Extraction | *(not supported)* | Use `--preload_stage extraction_task:file.json` instead |

### CLI

```bash
# Extraction + alignment only (skip judge and humanfeedback)
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_stage judge_task \
  --skip_stage humanfeedback_task \
  --save_file result.json

# Extraction + judge only (skip alignment — fast when you already have ontology data)
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_stage alignment_task \
  --save_file result.json
```

### Python

```python
flow = StructSenseFlow(
    agent_config=cfg["agent_config"],
    task_config=cfg["task_config"],
    embedder_config=cfg.get("embedder_config", {}),
    source="paper.pdf",
    api_key="sk-or-v1-...",
    # Run only extraction + alignment; stop before judge
    skip_stages=["judge_task", "humanfeedback_task"],
)
result = asyncio.run(flow.information_extraction_task())
```

**Combining with fast alignment:** `--skip_stage judge_task` + `CONCEPT_MAPPING_BACKEND=local` gives the fastest possible run — extraction followed immediately by a direct batch concept mapping call, no LLM calls at all.

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --skip_stage judge_task \
  --skip_stage humanfeedback_task \
  --skip_alignment_llm true \
  --save_file result.json
```

**Combining with preloaded stages:** `--skip_stage` and `--preload_stage` are independent and can be used together. For example, preload extraction and also skip judge:

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --preload_stage extraction_task:00_extractor_agent_extraction_task.json \
  --skip_stage judge_task \
  --save_file result.json
```

---

## Resuming a Pipeline from a Saved Stage

The pipeline can take hours on large documents. If it crashes after extraction or alignment, you do not need to re-run stages that already succeeded. Use **preloaded stages** to skip any stage and load its output from a saved JSON file instead.

Stage output files are written automatically when you set `stage_output_dir` in `StructSenseFlow`. They are named by order and agent/task:

```
00_extractor_agent_extraction_task.json
01_alignment_agent_alignment_task.json
02_judge_agent_judge_task.json
```

### Python — skip extraction, re-run from alignment

```python
import asyncio, json, yaml
from structsense.app import StructSenseFlow

with open("ner-config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load the extraction output that was already saved
with open("stage_outputs/00_extractor_agent_extraction_task.json") as f:
    extraction_result = json.load(f)

flow = StructSenseFlow(
    agent_config=cfg["agent_config"],
    task_config=cfg["task_config"],
    embedder_config=cfg.get("embedder_config", {}),
    source="paper.pdf",
    enable_chunking=True,
    chunk_size=2000,
    api_key="sk-or-v1-...",
    stage_output_dir="stage_outputs",   # saves new stage outputs here
)

# Pass the saved extraction output; the pipeline skips extraction and starts at alignment
result = asyncio.run(
    flow.information_extraction_task(
        preloaded_stages={"extraction_task": extraction_result}
    )
)
```

You can preload multiple stages. For example, to re-run only the judge:

```python
result = asyncio.run(
    flow.information_extraction_task(
        preloaded_stages={
            "extraction_task": extraction_result,
            "alignment_task": alignment_result,
        }
    )
)
```

### CLI — skip extraction, re-run from alignment

Pass `--preload_stage TASK_KEY:FILE` for each stage you want to skip. Repeat the flag for multiple stages.

```bash
# Skip extraction; re-run alignment → judge → humanfeedback
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --api_key sk-or-v1-... \
  --preload_stage extraction_task:stage_outputs/00_extractor_agent_extraction_task.json \
  --save_file result.json
```

```bash
# Skip extraction + alignment; re-run only judge → humanfeedback
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --api_key sk-or-v1-... \
  --preload_stage extraction_task:stage_outputs/00_extractor_agent_extraction_task.json \
  --preload_stage alignment_task:stage_outputs/01_alignment_agent_alignment_task.json \
  --save_file result.json
```

### Which stages can be preloaded?

Any stage — including judge and humanfeedback:

| `preloaded_stages` key | Stage skipped | Next stage to run |
|---|---|---|
| `extraction_task` | Extraction | Alignment |
| `alignment_task` | Alignment | Judge |
| `judge_task` | Judge | Human feedback (if enabled) |
| `humanfeedback_task` | Human feedback | — (pipeline is done) |

You can preload any combination. The pipeline uses each saved output as `prev_output` for the following stage, so the chain stays consistent.

### Chunk size when preloading

Chunking **only applies to extraction**. Alignment, judge, and humanfeedback always receive a single payload regardless of `chunk_size`.

| Scenario | `chunk_size` / `enable_chunking` needed? |
|---|---|
| `extraction_task` preloaded | No — extraction is skipped, no chunking happens |
| Any later stage preloaded, extraction still runs | Yes — set these as usual for extraction |
| All stages preloaded | No — nothing runs, values are ignored |

If you are re-running from alignment onwards (extraction preloaded), you can omit `--enable_chunking` and `--chunk_size` entirely:

```bash
structsense-cli extract \
  --config ner-config.yaml \
  --source paper.pdf \
  --api_key sk-or-v1-... \
  --preload_stage extraction_task:00_extractor_agent_extraction_task.json \
  --save_file result.json
  # no --enable_chunking or --chunk_size needed
```

**Important:** `--source` / `--source_text` is still required even when all upstream stages are preloaded, because the flow needs it for context.

---

## Testing

All tests are fully offline — no API key, no LLM, no network calls required.

```bash
cd structsense
pytest src/tests/test_no_truncation_regression.py \
       src/tests/test_model_context.py \
       src/tests/test_parallel_downstream_chunking.py -v
```

| Test file | What it covers | Tests |
|---|---|---|
| `test_no_truncation_regression.py` | Regression guard for issue #17 — verifies 2176 entities survive the full prepare chain (alignment → judge → humanfeedback) without any truncation. Tests every old truncation path: 70%/80% token-budget caps, `_extract_essentials` (10-entity fallback), `_ensure_list_keys_preserved` (100-entity hard cap), 20% feedback reservation, `context_manager` / `max_tokens` params, large-sentence chunk overflow. | 28 |
| `test_model_context.py` | Model family detection (`get_model_context_window`): 25 model families, case-insensitivity, specificity ordering (scout > maverick > llama4), fallback. Token-aware chunk sizing (`compute_downstream_chunk_size`): fits-in-one-call for 1M models, must-split minimum-chunks logic, max_workers cap, `context_window_override`, `extraction_chunk_count` as target, adaptive `prompt_overhead_tokens`. Adaptive prompt overhead (`estimate_agent_prompt_tokens`): config text counted, placeholder stripping, larger config → larger estimate, adaptive overhead integrated into chunk sizing. OpenRouter probe: 400 error parsed/cached, 200 lower bound, fallbacks, cache hit. | 67 |
| `test_parallel_downstream_chunking.py` | `unify_ontology_across_entities`: tool beats llm_knowledge, real IRI beats N/A, preserves all instances, different labels kept separate. `split_structured_payload` + `merge_structured_chunk_results`: no data loss for 50/500 entities, chunk count/metadata, key_terms dedup, resources preserved, key_terms must not inflate chunk count (regression for empty-entity chunk bug). Full round-trip integration. | 17 |

**Total: 112 offline tests.**

---

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
- **Partial pipeline** — run any subset of stages with `--skip_stage`; combine with `--preload_stage` to resume from any checkpoint
- **Task-type auto-detection** — detects NER, resource extraction, or structured extraction from your config; detected once and reused across all pipeline stages
- **Chunking** — splits large PDFs into sentence-aligned chunks and runs extraction in parallel; downstream alignment/judge/humanfeedback split automatically when the payload exceeds the model's context window
- **Fast alignment** — for local concept mapping backend, skips the alignment LLM entirely for NER, keyphrase, resource, and structured_extraction tasks; calls the concept mapping tool directly in batch (~seconds vs ~60 min)
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

For large PDFs, enable chunking to split the text into sentence-aligned chunks and run extraction in parallel. When `enable_chunking=True`, **all pipeline stages run in parallel** — extraction, alignment, judge, and humanfeedback are all chunked and dispatched with `asyncio.gather`:

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
    chunk_size=2000,            # characters per extraction chunk
    max_workers=8,              # parallel workers for all stages
    downstream_chunk_size=100,  # entities per alignment/judge chunk (explicit override; auto if omitted)
    model_context_window=200000,# override auto-detected context window (optional)
)
```

#### How downstream parallelism works

Alignment/judge/humanfeedback use **token-aware** chunk sizing based on model token size. 

**Decision logic (per stage):**

| Condition | Result |
|---|---|
| `explicit downstream_chunk_size` set | Use that value directly |
| `payload_tokens ≤ payload_budget` | **Single call** — no splitting, full payload in one LLM call |
| `payload_tokens > payload_budget` | Split into minimum chunks needed, capped at `max_workers` |

```
input_budget   = model_context_window × 0.70      (30 % reserved for output)
payload_budget = input_budget − prompt_overhead    (adaptive: read from actual config)
prompt_overhead = len(role + goal + backstory + task_description_template) + 3 000 CrewAI boilerplate
```

**Adaptive prompt overhead:** The agent config content (role, goal, backstory, task description) varies enormously — a minimal config may be ~1 k tokens while a richly annotated one with many entity-type examples may exceed 50 k. StructSense reads the actual YAML fields at runtime and estimates their size, so the payload budget automatically adjusts to your specific config. Template placeholders like `{aligned_structured_information}` are stripped to avoid double-counting the payload.

**Model context window auto-detection:** The model string from the YAML config is matched against a built-in registry of LLM families. No configuration needed for common models:

| Model family | Context window |
|---|---|
| Gemini 2.x / 1.5 | 1 000 000 |
| Llama 4 Scout | 10 000 000 |
| Llama 4 Maverick | 1 000 000 |
| GPT-4.1 | 1 000 000 |
| GPT-5 | 400 000 |
| Claude 3+ / 4 | 200 000 |
| Mistral Large 3 | 256 000 |
| Qwen 3 | 256 000 |
| Nemotron 3 | 1 000 000 |
| DeepSeek / GPT-4o / Llama 3.x | 128 000 |

For unrecognised models the default is 128 000. Override with `--model_context_window N` (CLI) or `model_context_window=N` (Python).

**OpenRouter live context-window probe:** When an OpenRouter API key is present and the model is served through OpenRouter, StructSense probes the real context window once at pipeline startup by sending an intentionally oversized dummy chat-completion request. OpenRouter returns a 400 error whose message contains the actual maximum:

```
"This endpoint's maximum context length is 1048576 tokens.
 However, you requested about 1152000 tokens (1152000 of text input)."
```

The real limit is parsed from the error and cached for the rest of the run. Subsequent stages use the confirmed value instead of the static dictionary estimate. If the probe fails (network error, non-context 400, or no API key) the static estimate is used as a fallback — no pipeline disruption.

**Example — Gemini 1M model, 480k-token payload, 15k-token config:**
```
input_budget   = 1 000 000 × 0.70 = 700 000
payload_budget = 700 000 − 15 000 = 685 000
480 000 ≤ 685 000 → single call, no chunking
```

**Example — DeepSeek 128k, 400k-token payload, 8k-token config, 8 workers:**
```
input_budget   = 128 000 × 0.70 = 89 600
payload_budget = 89 600 − 8 000 = 81 600
min_chunks     = ceil(400 000 / 81 600) = 5
n_chunks       = min(5, 8) = 5 → 5 parallel calls
```

 
**Ontology consistency:** After parallel alignment chunks are merged, entities with the same text may have received different ontology IDs from different LLM calls. A consistency pass unifies them: tool-backed mappings beat `llm_knowledge`, real IRIs beat `N/A`. All individual entity instances (different sentences/positions) are preserved.

**Data-loss guard:** After each parallel merge, entity count is compared to the pre-split count. If any chunk caused entities to be dropped, the pipeline falls back to the richest available previous stage result.

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

 
### BioPortal (mutually exclusive with local service)

Uses the [BioPortal](https://bioportal.bioontology.org/) REST API for ontology lookup with automatic ontology detection. Make sure you are using either Bioportal or local service, not both.

```bash
CONCEPT_MAPPING_BACKEND=bioportal
BIOPORTAL_API_KEY=your-key-here
```

Get a free API key at [bioportal.bioontology.org/account](https://bioportal.bioontology.org/account).

**Important Note: Bioportal has rate limit.**

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
| `SKIP_ALIGNMENT_LLM` | `true`/`false`/`auto` — bypass alignment LLM (equivalent to `--skip_alignment_llm`) |
| `SKIP_JUDGE_LLM` | `true`/`false` — bypass judge LLM, inject default scores (equivalent to `--skip_judge_llm`) |
| `SKIP_STAGES` | Comma-separated task keys to omit, e.g. `judge_task,humanfeedback_task` (equivalent to `--skip_stage`) |

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
