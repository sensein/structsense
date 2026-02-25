# 🧩 StructSense

Welcome to `structsense`!

`structsense` is a powerful multi-agent system designed to extract structured information from unstructured data. By orchestrating intelligent agents, it helps you make sense of complex information — hence the name *structsense*.

Whether you're working with scientific texts, documents, or messy data, `structsense` enables you to transform it into meaningful, structured insights.

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

| Option | Description |
|--------|-------------|
| `--config` | **(Required)** Path to YAML config (agent + task + embedder). |
| `--source` | **(Required)** Input: path to a PDF/text file, a folder, or a text string. |
| `--api_key` | OpenRouter (or other) API key; can also be set in `.env` as `OPENROUTER_API_KEY`. |
| `--env_file` | Path to `.env` (default: `.env` in current directory). |
| `--save_file` | Save the result JSON to this path. |
| `--enable_chunking` | Enable chunking for long documents (flag). |
| `--chunk_size` | Chunk size in characters (e.g. `2000`); used when chunking is enabled. |
| `--max_workers` | Max parallel workers for chunked extraction. |
| `--downstream_max_input_chars` | Max input length for alignment/judge (default 80000). |
| `--max_extraction_chunk_chars` | Cap per-chunk size for extraction (default 25000). |

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
    input_source="path/to/file.pdf",   # or a text string, or path to .txt
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
    input_source="path/to/file.pdf",  # or source_text="raw text"
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
    input_source="path/to/file.pdf",  # or source_text="raw text"
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

- During `pip install structsense` (or when it's a transitive dep), pip backtracks for a long time across many `opentelemetry-*` packages and eventually fails with:

- `pip install --use-deprecated=legacy-resolver structsense`

</details>

<details>
<summary><strong>Python version</strong></summary>

**Symptom**

- ERROR: Could not find a version that satisfies the requirement structsense (from versions: none) ERROR: No matching distribution found for structsense

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

### License
[Apache License Version 2.0](LICENSE.txt)
