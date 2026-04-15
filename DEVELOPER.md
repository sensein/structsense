# StructSense Developer Guide

## Development Setup

### Requirements

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/) >= 2.0

### Install

```bash
git clone https://github.com/sensein/structsense.git
cd structsense
poetry install --with dev
```

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your-key-here        # required for LLM calls
BIOPORTAL_API_KEY=your-key-here         # optional, only for BioPortal concept mapping
```

### Pre-commit Hooks

```bash
poetry run pre-commit install
```

### Running Tests

```bash
poetry run pytest                                      # all tests (requires OPENROUTER_API_KEY)
poetry run pytest -m "not requires_openrouter"         # unit tests only
```

---

## Architecture Overview

This document provides the overview of the major **StructSense** component and flow.

## Key Functions

### Pipeline entry and flow
- **`StructSenseFlow.information_extraction_task()`** (`app.py`)
  Main entry: runs extraction → alignment → judge → humanfeedback in order; builds `final`, injects `aligned_resources` for resource task, calls `promote_canonical_resources_for_resource_task`, handles skip end concept mapping and provenance.
- **`StructSenseFlow.run_agent_task()`** (`app.py`)
  Runs one agent–task pair (with optional chunking). Used by the pipeline for each stage and by CLI `run-agent`. Passes `pick_richest_alignment` for resource alignment so multiple alignment blobs are returned for merge.
- **`StructSenseFlow._get_detected_task_type()`** (`app.py`)
  Returns `"ner"`, `"resource"`, `"structured_extraction"`, or `"extraction"` from task config (LLM or heuristic). Drives post-processor, merger, and whether alignment blobs are extracted for merge.
###  Crew run and parsing

- **`run_crew_extraction()` / `run_crew_extraction_async()`** (`crew_utils.py`)
  Generic crew runner: no-chunk path (single run) or chunked path. No-chunk path can return multiple results when alignment returns multiple blobs (`pick_richest_alignment` + `_parse_crew_output_alignment_blobs`).

- **`_run_crew_on_retry()` / `_run_crew_on_retry_async()`** (`crew_utils.py`)
  One (or two on retry) crew kickoff; parse string output via `_parse_raw_string` → `_parse_crew_output_alignment_blobs` (if alignment resource) or `_parse_crew_output_string`.

- **`_parse_crew_output_alignment_blobs(s)`** (`crew_utils.py`)
  Extracts all top-level JSON blobs with `aligned_resources` and returns a list (2+) or single dict so the app can merge them in postprocessing.

- **`_extract_all_json_blobs(s)`** (`crew_utils.py`)
  Brace-matching extraction of every `{...}` in a string; used to get multiple alignment blocks from one LLM response.

The other functions include tool registration and invocation and merging results.

## Module Overview

- **`src/structsense/app.py`**: Main entry point for **StructSense**, responsible for initializing the pipeline and orchestrating agent execution.
- **`src/structsense/cli.py`**: Defines and handles command-line interface (CLI) options for running **StructSense**.
- **`src/structsense/humanloop.py`**: Implements the `human-in-the-loop` workflow, collecting human feedback (Approve, View, Modify, Abort), opening an editor for modifications, and routing execution based on user input.
- **`src/utils/crew_utils.py`**: Utilities for executing CrewAI agents, extracting and modularizing agent orchestration logic previously embedded in `app.py`.
- **`src/utils/downstream_agent_helper.py`**: Manages token-aware context construction and data flow between agents by preparing, compressing, splitting, merging, and normalizing structured outputs for downstream inputs.
- **`src/utils/context_window_manager.py`**: Provides token-aware context management by estimating token usage and compressing structured or unstructured inputs to fit within defined context window limits.
- **`src/utils/agent_context.py`**: Manages per-agent state, results, and confidence tracking with thread-safe storage, chunk-level aggregation, and token-bounded summaries for parallel and multi-stage execution.
- **`src/utils/task_detection.py`**: Implements LLM-based + heuristics-based (default fallback) task detection, defining the task taxonomy, associated tools, and prompt construction logic to classify inputs with confidence and rationale.
- **`src/utils/task_tools.py`**: Centralizes tool resolution and orchestration, mapping task types and agent configurations to concrete tool implementations while handling runtime configuration and API dependencies.
- **`src/utils/ner_tool.py`**: Provides NER-specific tooling and execution logic for entity extraction tasks.
- **`src/utils/conceptmappingtool.py`**: Implements the **Concept Mapping tool**, offering cached and rate-limited integration with external ontology services (e.g., BioPortal) to map extracted concepts and capture provenance-aware alignment outputs.
- **`src/utils/postprocessing.py`**: Implements task-specific post-processing, validation, and merging logic, including weighted voting, chunk-level result consolidation, provenance-aware resource normalization, and final output normalization.
- **`src/utils/text_chunking.py`**: Handles text chunking and span management by splitting documents into sentence-bounded chunks, validating and globalizing entity spans, and merging entity occurrences across chunks.
- **`src/utils/utils.py`**: Provides shared utilities for input handling, configuration loading, external service integration (e.g., `Grobid PDF extraction`, `Weaviate` (not used see [Change Log](CHANGE_LOG.md)), `Ollama`), and data transformation across the pipeline.
- **`src/utils/mlops.py`**: Provides lightweight MLOps integration by conditionally enabling experiment monitoring (e.g., `Weights & Biases` or `MLflow`) based on environment configuration.

## Adding a New Tool

### Overview

Tools are attached **per stage (agent) and per task type**. The flow is:
- **Task type** is detected from the extractor task description (`task_detection.py`).
- **Tool names** for that (agent_key, task_type) come from `task_tools.py` (generic + task-specific).
- **Tool names** are resolved to **CrewAI tool instances** in `_resolve_tool()` in `task_tools.py`.
- The agent is built with those tools; CrewAI runs the agent and the LLM can **call** the tools during execution.

### Steps to Add a New Tool
-  Implement the tool. For organization purpose, please place tools under `src/utils/` directory and please ensure the tool’s **name** and **description** are clear so the LLM knows when to call it.
- Register the tool name for a task type and/or stage.

| Purpose                                                           | File | What to change |
|-------------------------------------------------------------------|------|----------------|
| Tool only for **extractor** for a specific task type (e.g. `ner`) | `src/utils/task_detection.py` | Add an entry to **`TOOLS_BY_TASK_TYPE`**: e.g. `"ner": ["extract_ner_terms", "your_new_tool"]`. Keys must match the **taxonomy** task types (ner, resource, extraction, keyphrase_extraction, etc.). |
| Tool for **all task types** of an agent (e.g. alignment)          | `src/utils/task_tools.py` | Add the tool name to **`GENERIC_TOOLS_BY_STAGE`**: e.g. `"alignment_agent": ["concept_mapping_tool", "your_new_tool"]`. |
| Tool for a specific (agent, task_type) not covered above          | `src/utils/task_tools.py` | Ensure the task type exists in **`TOOLS_BY_STAGE_AND_TASK`** (it references `TOOLS_BY_TASK_TYPE` for extractor). For other agents, add a mapping under **`TOOLS_BY_STAGE_AND_TASK`** for that agent and task type. |
- To add a new tool, update `_resolve_tool(name, ...)` in `src/utils/task_tools.py` to map the tool name to its corresponding function or instantiated class.

    **Example (function tool):**

    ```python
    # In task_tools.py _resolve_tool()
    if name == "your_new_tool":
        if name not in _TOOL_REGISTRY:
            from .your_module import your_tool_func
            _TOOL_REGISTRY[name] = your_tool_func
        return _TOOL_REGISTRY.get(name)
    ```
    **Example (e.g. ConceptMappingTool):**

    ```python
        if name == "concept_mapping_tool":
            if name not in _TOOL_REGISTRY:
                try:
                    from .conceptmappingtool import ConceptMappingTool
                    _TOOL_REGISTRY[name] = ConceptMappingTool()
                except ValueError as e:
                    logger.warning(f"ConceptMappingTool not registered (missing BIOPORTAL_API_KEY): {e}")
                    return None
            return _TOOL_REGISTRY.get(name)
    ```
- For tools supporting new task types, update the taxonomy to include the corresponding task definition.

###  Task types & post-processing Information

The pipeline detects **task type** from the extractor task description (LLM or heuristic) and applies matching post-processors and mergers:

| Task type                | Post-processor   | Merge behavior |
|--------------------------|------------------|----------------|
| **ner**                  | NER              | Weighted voting; entities with `occurrences`, `provenance`, `weighted_score` |
| **resource** / **structured_extraction** | Resource | All resources merged into **one** aggregated resource with **list-valued** fields (`resource_name`, `description`, `type`, etc. as lists; `mentions` merged and deduped) |
| **extraction** (generic) | Pass-through     | Concatenate list values per key |

- **NER output schema** (per entity): `text`, `label`, `start`, `end`, `weighted_score`, `model_count`, `occurrences` (list of `{start, end, global_start, global_end, sentence}`), `provenance` (labels and source models with weights).
- **Resource output**: Single aggregated resource with list fields (e.g. `resource_name`: list of names, `mentions.datasets` / `mentions.related_models` / `mentions.related_papers` merged across all extracted resources).

**Final output shape (all tasks):** The pipeline returns only **task-specific** keys (no intermediate agent containers). NER: `entities`, `key_terms`, `verification`, `errors`, `task_type`, `elapsed_time`. Resource: `resources`, `verification`, `errors`, `task_type`, `elapsed_time`. Provenance is added by default for all stages (extractor, alignment, judge, human feedback).

---

### Tools Information

Tools are resolved **per stage (agent) and task type** so different agents and tasks can use different tools.

- **Stage (agent)**
  Only the **extractor** stage receives tools by default; **alignment**, **judge**, and **human_feedback** use no tools (LLM-only).

- **Generic tools**
  An agent can have **generic** tools that apply to **all** task types (e.g. a shared search or lookup). Configured in `GENERIC_TOOLS_BY_STAGE` in `src/utils/task_tools.py`.

- **Task-specific tools**
  Task types (e.g. `ner`, `keyphrase_extraction`) get additional tools (e.g. `extract_ner_terms`). Configured in `TOOLS_BY_TASK_TYPE` (in `task_detection`) for the extractor and in `TOOLS_BY_STAGE_AND_TASK` for other agents.

**Current mapping:**

| Stage             | Task types with tools     | Tools              |
|-------------------|---------------------------|--------------------|
| extractor_agent   | ner, keyphrase_extraction | extract_ner_terms   |
| extractor_agent   | extraction, resource, …   | (none)             |
| alignment_agent   | (all)                     | (none)             |
| judge_agent       | (all)                     | (none)             |
| human_feedback    | (all)                     | (none)             |

To add tools: extend `TOOLS_BY_TASK_TYPE` (extractor) or `TOOLS_BY_STAGE_AND_TASK` (any agent), and/or add generic tools in `GENERIC_TOOLS_BY_STAGE`; register new tool names in `_resolve_tool` in `src/utils/task_tools.py`.
