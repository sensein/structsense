# StructSense Information for Developer 

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

