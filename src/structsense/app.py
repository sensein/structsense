"""StructSense main application and pipeline orchestration.

This module provides the primary entry point for structured information extraction
via the :class:`StructSenseFlow` class. It runs a multi-agent CrewAI pipeline
(extraction → alignment → judge → humanfeedback) with context management and
token-window handling.

Key classes and functions
-------------------------
- :class:`StructSenseFlow` – Main orchestrator; use :meth:`~StructSenseFlow.information_extraction_task`
  for the full pipeline or :meth:`~StructSenseFlow.kickoff` for a single agent-task.
- :meth:`StructSenseFlow.information_extraction_task` – Recommended entry point for production
  (full pipeline with token management).
- :meth:`StructSenseFlow.kickoff` – Run a single agent-task pair.
- :meth:`StructSenseFlow.extraction` – Run only the extraction agent.
- :meth:`StructSenseFlow.run_agent_task` – Low-level single agent-task with full control.

Key constants (used when chaining pipeline stages)
--------------------------------------------------
- :data:`PIPELINE_INPUT_KEY_MAP` – Maps each task_key to the input placeholder name
  the next stage expects (e.g. extraction_task → input_text, alignment_task → extracted_structured_information).
- :data:`TASK_KEY_TO_CONTAINER_KEY` – Maps task_key to the container key in that task's
  output for merging and provenance (e.g. extraction_task → resources or extracted_terms).
- :data:`DOWNSTREAM_CONTAINER_KEYS` – All known container keys used in NER/resource pipelines.

See also
--------
- :mod:`utils.task_tools` – Tool resolution per agent and task type.
- :mod:`utils.postprocessing` – Post-processors and result mergers per task type.
- :mod:`utils.task_detection` – Task-type detection from config.
"""

import json
import logging
import os
import time
import tracemalloc
import asyncio
import math
import threading
from pathlib import Path
from datetime import datetime

# Filter warnings at the beginning
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Disable all warnings including Pydantic serialization warnings
warnings.filterwarnings("ignore")
# Specifically suppress Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress Pydantic-related warnings using string patterns
warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*serialization.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields.*")
warnings.filterwarnings("ignore", message=".*PydanticSerialization.*")

# Disable CrewAI tracing and telemetry BEFORE importing crewai
# This must be set before any crewai imports
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
# Also disable interactive prompts and telemetry messages
os.environ["CREWAI_DISABLE_INTERACTIVE"] = "true"
# Suppress telemetry output messages
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from crewai import Crew, Process
from crewai.hooks import before_llm_call, after_llm_call
from dotenv import load_dotenv

from utils.utils import (
    load_config,
    process_file,
    replace_api_key,
    str_to_bool,
    check_ollama_health,
)
from utils.task_detection import detect_task_type, DEFAULT_TAXONOMY
from utils.task_tools import get_tools_for_agent
from utils.crew_utils import initialize_memory
from utils.mlops import setup_monitoring


from utils.crew_utils import run_crew_extraction, run_crew_extraction_async, initialize_agent_and_task
from utils.postprocessing import (
    get_post_processor,
    get_result_merger,
    merge_downstream_chunk_results_with_provenance,
    add_provenance_to_result,
    verify_merged_result,
    apply_concept_mapping_to_result,
    normalize_final_result_for_output,
    promote_canonical_resources_for_resource_task,
    ensure_resource_mapped_concepts_provenance,
    promote_stage_output_to_canonical,
    # FIX: inject concept mapping (class_uri/ontology_label/ontology_id) from alignment
    # agent tool calls directly into NER entity dicts after the alignment stage.
    # Previously these ontology fields were captured by the tool but never surfaced to
    # the caller because the injection block only ran for task_type == "extraction".
    inject_alignment_concept_mapping_into_ner_entities,
    inject_alignment_concept_mapping_into_resources,
    _flatten_container_to_list,
    unify_ontology_across_entities,
)
from .humanloop import HumanInTheLoop

# Import enhanced context management
from utils.agent_context import AgentContext, ThreadSafeMemory
from utils.context_window_manager import ContextWindowManager
from utils.downstream_agent_helper import (
    prepare_alignment_agent_input,
    prepare_judge_agent_input,
    prepare_humanfeedback_agent_input,
    split_structured_payload,
    merge_structured_chunk_results,
)
from utils.model_context import (
    compute_downstream_chunk_size,
    estimate_agent_prompt_tokens,
    probe_openrouter_context_window,
)
from utils.conceptmappingtool import (
    clear_alignment_tool_outputs,
    get_alignment_tool_outputs,
    format_alignment_tool_outputs_as_concept_mapping,
)


# Start memory tracking
tracemalloc.start()

# Configure logging - filter out warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warning logs from specific modules
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("pydantic_core").setLevel(logging.ERROR)
logging.getLogger("crewai").setLevel(logging.INFO)  # Keep INFO, suppress WARNING
# Suppress CrewAI telemetry and tracing messages
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.tracing").setLevel(logging.CRITICAL)
logging.getLogger("crewai.memory").setLevel(logging.WARNING)  # Only show warnings/errors from memory
# Suppress all UserWarning messages
logging.captureWarnings(True)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", category=UserWarning)
# Suppress Pydantic serialization warnings specifically (using string patterns)
warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*serialization.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields.*")


# Setup timing logger to separate file
def setup_timing_logger():
    """Setup a separate logger for timing information."""
    timing_log_dir = Path(os.getcwd()) / "timing_logs"
    timing_log_dir.mkdir(exist_ok=True)

    timing_log_file = timing_log_dir / f"timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    timing_logger = logging.getLogger("timing")
    timing_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in timing_logger.handlers[:]:
        timing_logger.removeHandler(handler)

    # File handler for timing log
    file_handler = logging.FileHandler(timing_log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    timing_logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    timing_logger.addHandler(console_handler)

    timing_logger.propagate = False

    return timing_logger, str(timing_log_file)


# ---------------------------------------------------------------------------
# LLM call tracking — separate log file per run, thread-safe counter
# ---------------------------------------------------------------------------
_llm_call_count = {"n": 0}
_llm_call_lock = threading.Lock()
# Per-stage tracking: task_key → {start_n, agent_key, n_chunks, calls, calls_per_chunk, elapsed}
_llm_stage_tracker: Dict[str, Dict[str, Any]] = {}


def setup_llm_call_logger() -> tuple:
    """Set up a dedicated logger that records every LLM call made by CrewAI.

    Creates ``llm_call_logs/llm_calls_YYYYMMDD_HHMMSS.log`` in the current
    working directory. Also resets the global call counter so each pipeline
    run starts from #1.

    Returns
    -------
    tuple[logging.Logger, str]
        The configured logger and the absolute path to the log file.
    """
    log_dir = Path(os.getcwd()) / "llm_call_logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    llm_logger = logging.getLogger("llm_calls")
    llm_logger.setLevel(logging.INFO)

    # Remove any handlers left from a previous run
    for handler in llm_logger.handlers[:]:
        llm_logger.removeHandler(handler)

    # delay=False opens the file immediately so the first flush has a real fd
    file_handler = logging.FileHandler(log_file, delay=False)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    llm_logger.addHandler(file_handler)

    # Avoid duplicating records in the root logger
    llm_logger.propagate = False

    # Reset counter for this run
    with _llm_call_lock:
        _llm_call_count["n"] = 0

    return llm_logger, str(log_file)


def _flush_llm_logger() -> None:
    """Flush all handlers of the llm_calls logger to disk immediately."""
    for h in logging.getLogger("llm_calls").handlers:
        h.flush()


def _begin_stage_llm_tracking(task_key: str, agent_key: str) -> None:
    """Snapshot the current call count so the stage delta can be computed later."""
    with _llm_call_lock:
        _llm_stage_tracker[task_key] = {
            "agent_key": agent_key,
            "start_n": _llm_call_count["n"],
        }


def _end_stage_llm_tracking(task_key: str, n_chunks: int, stage_elapsed: float) -> None:
    """Compute and log per-stage call summary immediately when a stage finishes.

    Logs to the ``llm_calls`` log file (real-time flush) and to the main logger.
    Formula: total_calls = number_of_chunks × calls_per_chunk
    """
    tracker = _llm_stage_tracker.get(task_key)
    if not tracker:
        return

    with _llm_call_lock:
        current_n = _llm_call_count["n"]

    stage_calls = current_n - tracker["start_n"]
    calls_per_chunk = stage_calls / n_chunks if n_chunks > 0 else 0
    agent_key = tracker.get("agent_key", "unknown")

    # Store for final summary
    tracker.update({
        "n_chunks": n_chunks,
        "calls": stage_calls,
        "calls_per_chunk": calls_per_chunk,
        "elapsed": stage_elapsed,
    })

    llm_logger = logging.getLogger("llm_calls")
    if llm_logger.handlers:
        llm_logger.info("-" * 70)
        llm_logger.info(f"[STAGE COMPLETE] {agent_key} / {task_key}")
        llm_logger.info(f"  Chunks          : {n_chunks}")
        llm_logger.info(f"  LLM calls       : {stage_calls}")
        llm_logger.info(f"  Calls/chunk     : {calls_per_chunk:.1f}  "
                        f"(= {n_chunks} chunks × {calls_per_chunk:.1f} calls/chunk)")
        llm_logger.info(f"  Stage elapsed   : {stage_elapsed:.1f}s")
        llm_logger.info(f"  Running total   : {current_n} calls so far this run")
        llm_logger.info("-" * 70)
        _flush_llm_logger()

    logger.info(
        "[llm_tracking] %s/%s: %d chunks × %.1f calls/chunk = %d LLM calls (%.1fs)",
        agent_key, task_key, n_chunks, calls_per_chunk, stage_calls, stage_elapsed,
    )


@before_llm_call
def _on_before_llm_call(context):
    """Hook: log every LLM call before it is dispatched.

    Increments the global call counter (thread-safe) and writes a structured
    record — agent role, task description, iteration number, message count,
    and a preview of the last message — to the ``llm_calls`` logger.
    Flushes to disk immediately so the file is readable as the pipeline runs.
    """
    with _llm_call_lock:
        _llm_call_count["n"] += 1
        n = _llm_call_count["n"]

    llm_logger = logging.getLogger("llm_calls")
    if not llm_logger.handlers:
        return None  # Logger not yet configured; silently skip

    _agent_obj = getattr(context, "agent", None)
    agent_role = getattr(_agent_obj, "role", "unknown")
    agent_max_iter = getattr(_agent_obj, "max_iter", "unknown")
    agent_max_exec_time = getattr(_agent_obj, "max_execution_time", "unknown")
    agent_max_retry = getattr(_agent_obj, "max_retry_limit", "unknown")
    task_desc = str(getattr(getattr(context, "task", None), "description", "unknown"))
    iteration = getattr(context, "iterations", "unknown")
    messages = getattr(context, "messages", [])

    llm_logger.info("=" * 70)
    llm_logger.info(f"[LLM CALL #{n}]")
    llm_logger.info(f"Agent          : {agent_role}")
    llm_logger.info(f"max_iter       : {agent_max_iter}  |  max_execution_time: {agent_max_exec_time}s  |  max_retry_limit: {agent_max_retry}")
    llm_logger.info(f"Task           : {task_desc[:120]}")
    llm_logger.info(f"Iteration      : {iteration}")
    llm_logger.info(f"Messages       : {len(messages)}")
    if messages:
        last_msg = messages[-1]
        llm_logger.info(f"Last msg role    : {last_msg.get('role')}")
        llm_logger.info(f"Last msg preview : {str(last_msg.get('content', ''))[:300]}")
    _flush_llm_logger()

    return None  # Allow the LLM call to proceed unchanged


@after_llm_call
def _on_after_llm_call(context):
    """Hook: log the LLM response preview after each call completes.

    Flushes to disk immediately so the file is readable while the pipeline
    is still running.
    """
    llm_logger = logging.getLogger("llm_calls")
    if not llm_logger.handlers:
        return None

    response = getattr(context, "response", None)
    llm_logger.info("[LLM RESPONSE PREVIEW]")
    llm_logger.info(str(response)[:500] if response else "No response found on context.")
    llm_logger.info("=" * 70)
    _flush_llm_logger()

    return None  # Keep the original response unchanged


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


#: Maps each pipeline task_key to the input placeholder name the next stage expects.
#: Used when chaining agents: each stage receives the previous stage's output under this key.
#: Example: alignment_task receives extraction output as ``extracted_structured_information``.
PIPELINE_INPUT_KEY_MAP = {
    "extraction_task": "input_text",
    "alignment_task": "extracted_structured_information",
    "judge_task": "aligned_structured_information",
    "humanfeedback_task": "judged_structured_information_with_human_feedback",
}

#: Maps task_key to the container key in that task's output (for merge and provenance).
#: NER uses extracted_terms, aligned_ner_terms, judge_ner_terms; resource uses resources, aligned_resources, judge_resource.
TASK_KEY_TO_CONTAINER_KEY = {
    "extraction_task": "resources",  # resource; NER uses extracted_terms (detected at runtime)
    "alignment_task": "aligned_resources",
    "judge_task": "judge_resource",
    "humanfeedback_task": "judge_resource",
}
#: All known container keys used when merging or validating NER/resource pipeline results.
DOWNSTREAM_CONTAINER_KEYS = (
    "extracted_terms",
    "aligned_ner_terms",
    "judge_ner_terms",  # NER
    "extracted_resources",
    "aligned_resources",
    "judge_resource",
    "resources",  # resource / generic
)


class StructSenseFlow:
    """Workflow for structured information extraction, alignment, and judgment using CrewAI.

    Single entry point for all extraction workflows. Supports chunked parallel
    extraction, multi-stage pipeline (extraction → alignment → judge → humanfeedback),
    context management, and token-window handling for downstream agents.

    Attributes
    ----------
    source_text : str
        Input text used when no override is passed to kickoff/information_extraction_task.
    agent_config : dict
        Loaded agent configuration (role, goal, llm, etc.).
    task_config : dict
        Loaded task configuration (description, agent_id, etc.).
    enable_chunking : bool
        Whether to split long text into chunks and process in parallel.
    token_limit : int
        Max tokens for downstream agent context (e.g. 100000 for 128k models).

    See Also
    --------
    information_extraction_task : Run the full pipeline (recommended).
    kickoff : Run a single agent-task pair.
    run_agent_task : Low-level single agent-task with full control.
    """

    def __init__(
            self,
            agent_config: Union[str, dict],
            task_config: Union[str, dict],
            embedder_config: Union[str, dict],
            source_text: Optional[str] = None,
            source: Optional[Union[str, dict]] = None,
            enable_human_feedback: bool = False,
            enable_chunking: bool = False,
            knowledge_config: Optional[Union[str, dict]] = None,
            agent_feedback_config: Optional[Dict[str, bool]] = None,
            env_file: Optional[str] = None,
            api_key: Optional[str] = None,
            chunk_size: Optional[int] = None,
            max_workers: Optional[int] = None,
            downstream_max_input_chars: Optional[int] = None,
            max_extraction_chunk_chars: Optional[int] = None,
            return_full_pipeline_details: bool = False,
            stage_output_dir: Optional[str] = os.getcwd(),
            downstream_chunk_size: Optional[int] = None,
            skip_alignment_llm: Optional[bool] = None,
            skip_judge_llm: Optional[bool] = None,
            direct_judge_api: Optional[bool] = True,
            direct_humanfeedback_api: Optional[bool] = True,
            model_context_window: Optional[int] = None,
            skip_stages: Optional[List[str]] = None,
            agent_max_iter: Optional[int] = None,
            agent_max_execution_time: Optional[int] = None,
            agent_max_retry_limit: Optional[int] = None,
    ):
        """Initialize StructSenseFlow with config paths and input.

        Parameters
        ----------
        agent_config : str or dict
            Path to agent YAML/JSON or dict (role, goal, llm per agent).
        task_config : str or dict
            Path to task YAML/JSON or dict (description, agent_id per task).
        embedder_config : str or dict
            Path to embedder config or dict (used for memory/embedding).
        source : str, optional
            Path to a file to process (PDF, CSV, or TXT). Mutually exclusive
            with ``source_text``. Processed internally via :func:`utils.utils.process_file`.
        source_text : str, optional
            Raw text to process directly. Mutually exclusive with ``source``.
        enable_human_feedback : bool, optional
            Whether to run the humanfeedback stage. Default False.
        enable_chunking : bool, optional
            If True, split text into chunks and run extraction in parallel.
        knowledge_config : str or dict, optional
            Optional knowledge-base config.
        agent_feedback_config : dict, optional
            Per-agent flags for feedback (e.g. which agents accept feedback).
        env_file : str, optional
            Path to .env file; loaded with override=True if set.
        api_key : str, optional
            If set, stored in ``OPENROUTER_API_KEY`` for LLM calls.
        chunk_size : int, optional
            Max characters per chunk when chunking is enabled.
        max_workers : int, optional
            Max parallel workers for chunked extraction.
        downstream_max_input_chars : int, optional
            Cap on input size for downstream agents (before token management).
        max_extraction_chunk_chars : int, optional
            Cap on chunk size for extraction agent context.
        return_full_pipeline_details : bool, optional
            If True, result includes pipeline_stages, token_usage, context_management.
        stage_output_dir : str, optional
            Directory path where each stage's output is written to disk as JSON
            immediately after that stage completes.  Useful for long-running pipelines
            (extraction 10 min + alignment 60 min + judge 40 min) so that a crash does
            not lose all prior work.  Files are named
            ``<stage_index>_<agent_key>_<task_key>.json`` and overwritten on re-run.
            Defaults to the current working directory.  Set to None to disable.
        skip_judge_llm : bool or None, optional
            When True, the judge LLM is bypassed: all entities receive a default
            ``judge_score=1.0`` and ``remarks="auto-approved"`` without an LLM call,
            and the pipeline continues immediately.  Useful for fast runs where quality
            scoring is not needed, or when the alignment output is already trusted.
            None (default): never auto-skip (judge always runs unless the stage is absent
            from the config or listed in ``skip_stages``).
            Overridden by env var ``SKIP_JUDGE_LLM=true/false``.
        skip_stages : list of str, optional
            Task keys to omit from the pipeline entirely.  The previous stage's output
            is passed directly to the next stage that is *not* skipped.  Example::

                # Run only extraction + alignment; skip judge and humanfeedback
                skip_stages=["judge_task", "humanfeedback_task"]

                # Run only extraction (single-stage)
                skip_stages=["alignment_task", "judge_task", "humanfeedback_task"]

            Accepted keys: ``extraction_task``, ``alignment_task``, ``judge_task``,
            ``humanfeedback_task``.  Skipping ``extraction_task`` is not supported
            (use ``preloaded_stages`` for that instead).
            ``enable_human_feedback=False`` (the default) already suppresses
            ``humanfeedback_task`` so you don't need to list it here unless you have
            a custom task key.
        agent_max_iter : int, optional
            Maximum number of reasoning iterations each CrewAI agent is allowed per
            run before it is forced to return its best answer.  Applies to every
            agent in the pipeline (extractor, alignment, judge, humanfeedback).
            CrewAI's built-in default is 20.  Lower values (e.g. 3–5) reduce cost
            and latency on straightforward extraction tasks; higher values give
            agents more attempts to self-correct on complex inputs.
            Overridden by env var ``AGENT_MAX_ITER=<int>``.

        Raises
        ------
        ConfigError
            If both or neither of ``source`` / ``source_text`` are provided,
            or if the resulting text is empty.
        """
        super().__init__()

        # Setup environment first
        if env_file:
            load_dotenv(env_file, override=True)
            logger.info(f"Loaded environment variables from {env_file} (override=True)")
        else:
            load_dotenv()
            logger.info("Loaded environment variables from default .env")

        # Human feedback: off by default; may be set from YAML human_in_loop_config when loading a combined config, then overridden by ENABLE_HUMAN_FEEDBACK env if set (see after config load).

        # Set API key in environment if provided
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
            logger.info("Set OPENROUTER_API_KEY in environment")

        # Disable CrewAI tracing and telemetry to reduce log noise
        if "CREWAI_TRACING_ENABLED" not in os.environ:
            os.environ["CREWAI_TRACING_ENABLED"] = "false"
        if "CREWAI_DISABLE_TELEMETRY" not in os.environ:
            os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        if "OTEL_SDK_DISABLED" not in os.environ:
            os.environ["OTEL_SDK_DISABLED"] = "true"
        if "CREWAI_TELEMETRY_OPT_OUT" not in os.environ:
            os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

        if source and source_text:
            raise ConfigError("Provide either source or source_text, not both.")
        elif source:
            self.source_text = process_file(source)
        elif source_text:
            self.source_text = source_text
        else:
            raise ConfigError("Either source or source_text must be provided.")

        # Validate that we have text to process
        if not self.source_text or not isinstance(self.source_text, str):
            raise ConfigError("source_text must be a non-empty string")

        if len(self.source_text.strip()) == 0:
            raise ConfigError("Extracted text is empty")

        logger.info(f"Initializing StructSenseFlow with source text (length: {len(self.source_text)} chars)")
        self.enable_human_feedback = enable_human_feedback

        try:
            # Load configs if they are file paths, otherwise use as-is
            if isinstance(agent_config, str):
                loaded = load_config(agent_config, "agent_config")
                # Combined YAML may have top-level agent_config, task_config, human_in_loop_config
                if isinstance(loaded, dict) and "agent_config" in loaded:
                    if "human_in_loop_config" in loaded and "ENABLE_HUMAN_FEEDBACK" not in os.environ:
                        hil = loaded.get("human_in_loop_config") or {}
                        enable_human_feedback = str_to_bool(str(hil.get("humanfeedback_agent", enable_human_feedback)))
                        logger.info("Human feedback set from config human_in_loop_config.humanfeedback_agent -> %s", enable_human_feedback)
                    self.agent_config = loaded.get("agent_config", loaded)
                else:
                    self.agent_config = loaded
            else:
                self.agent_config = agent_config

            if isinstance(task_config, str):
                loaded = load_config(task_config, "task_config")
                if isinstance(loaded, dict) and "task_config" in loaded:
                    if "human_in_loop_config" in loaded and "ENABLE_HUMAN_FEEDBACK" not in os.environ:
                        hil = loaded.get("human_in_loop_config") or {}
                        enable_human_feedback = str_to_bool(str(hil.get("humanfeedback_agent", enable_human_feedback)))
                        logger.info("Human feedback set from config human_in_loop_config.humanfeedback_agent -> %s", enable_human_feedback)
                    self.task_config = loaded.get("task_config", loaded)
                else:
                    self.task_config = loaded
            else:
                self.task_config = task_config

            # Env override: ENABLE_HUMAN_FEEDBACK takes precedence over config file
            if "ENABLE_HUMAN_FEEDBACK" in os.environ:
                enable_human_feedback = str_to_bool(os.environ["ENABLE_HUMAN_FEEDBACK"])
                logger.info(
                    "Human feedback overridden by env ENABLE_HUMAN_FEEDBACK=%s -> %s",
                    os.environ["ENABLE_HUMAN_FEEDBACK"],
                    enable_human_feedback,
                )
            self.enable_human_feedback = enable_human_feedback

            if isinstance(embedder_config, str):
                self.embedder_config = load_config(embedder_config, "embedder_config")
            else:
                self.embedder_config = embedder_config

            if knowledge_config:
                if isinstance(knowledge_config, str):
                    self.knowledge_config = load_config(knowledge_config, "knowledge_config")
                else:
                    self.knowledge_config = knowledge_config
            else:
                self.knowledge_config = None

            # Replace API key in configs if provided
            if api_key:
                self.agent_config = replace_api_key(self.agent_config, api_key)
                self.embedder_config = replace_api_key(self.embedder_config, api_key)

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ConfigError(f"Failed to load configurations: {str(e)}")

        setup_monitoring()
        _, llm_call_log_file = setup_llm_call_logger()
        logger.info("LLM call tracking log: %s", llm_call_log_file)

        # Crew memory (long/short/entity) is off by default; not recommended with local models.
        # Set ENABLE_CREW_MEMORY=true to enable (requires embedder_config, e.g. Ollama).
        enable_crew_memory = str_to_bool(os.environ.get("ENABLE_CREW_MEMORY", "false"))
        self.long_term_memory = None
        self.short_term_memory = None
        self.entity_memory = None

        if enable_crew_memory and self.embedder_config:
            try:
                embedder_provider = None
                if isinstance(self.embedder_config, dict):
                    embedder_provider = self.embedder_config.get("provider")
                    if embedder_provider == "ollama":
                        if not check_ollama_health():
                            logger.warning("Ollama not available, disabling memory")
                        else:
                            self.long_term_memory, self.short_term_memory, self.entity_memory = initialize_memory(
                                embedder_config=self.embedder_config
                            )
                    else:
                        self.long_term_memory, self.short_term_memory, self.entity_memory = initialize_memory(
                            embedder_config=self.embedder_config
                        )
                else:
                    self.long_term_memory, self.short_term_memory, self.entity_memory = initialize_memory(
                        embedder_config=self.embedder_config
                    )
                if any([self.long_term_memory, self.short_term_memory, self.entity_memory]):
                    logger.info("Crew memory systems initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize memory: {e}. Continuing without memory.")
        elif not enable_crew_memory:
            logger.info("Crew memory disabled (ENABLE_CREW_MEMORY=false, default). Set ENABLE_CREW_MEMORY=true to enable.")

        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size or 2000  # Default chunk size
        self.max_workers = max_workers
        # Entities per chunk for downstream parallel chunking (alignment/judge/humanfeedback).
        # None = auto-calculate from max_workers and entity count.
        self.downstream_chunk_size = downstream_chunk_size
        # Skip alignment LLM: call concept mapping tool directly and inject results,
        # bypassing the alignment agent entirely.
        # None = auto (True when CONCEPT_MAPPING_BACKEND=local and task is NER/keyphrase/resource).
        # True = always skip. False = always run alignment LLM.
        # Env var SKIP_ALIGNMENT_LLM=true/false/auto overrides the constructor argument.
        if "SKIP_ALIGNMENT_LLM" in os.environ:
            _env_sal = os.environ["SKIP_ALIGNMENT_LLM"].strip().lower()
            if _env_sal == "auto":
                skip_alignment_llm = None
            else:
                skip_alignment_llm = str_to_bool(_env_sal)
            logger.info(
                "skip_alignment_llm overridden by env SKIP_ALIGNMENT_LLM=%s -> %s",
                os.environ["SKIP_ALIGNMENT_LLM"], skip_alignment_llm,
            )
        self.skip_alignment_llm = skip_alignment_llm

        # Skip judge LLM: inject default judge_score/remarks directly, bypassing the judge agent.
        # None = never auto-skip (judge always runs when present).
        # True = always skip. False = always run judge LLM.
        # Env var SKIP_JUDGE_LLM=true/false overrides the constructor argument.
        if "SKIP_JUDGE_LLM" in os.environ:
            skip_judge_llm = str_to_bool(os.environ["SKIP_JUDGE_LLM"].strip())
            logger.info(
                "skip_judge_llm overridden by env SKIP_JUDGE_LLM=%s -> %s",
                os.environ["SKIP_JUDGE_LLM"], skip_judge_llm,
            )
        self.skip_judge_llm = skip_judge_llm

        # Direct API judge: call the LLM directly (no CrewAI agent overhead) for the judge stage.
        # Enabled by default (True). Env var DIRECT_JUDGE_API=false disables it.
        # When False and skip_judge_llm is also False, falls back to the full CrewAI agent run.
        if "DIRECT_JUDGE_API" in os.environ:
            direct_judge_api = str_to_bool(os.environ["DIRECT_JUDGE_API"].strip())
            logger.info(
                "direct_judge_api overridden by env DIRECT_JUDGE_API=%s -> %s",
                os.environ["DIRECT_JUDGE_API"], direct_judge_api,
            )
        self.direct_judge_api = direct_judge_api if direct_judge_api is not None else True

        # Direct API humanfeedback: same as direct_judge_api but for the humanfeedback stage.
        # Enabled by default (True). Env var DIRECT_HUMANFEEDBACK_API=false disables it.
        if "DIRECT_HUMANFEEDBACK_API" in os.environ:
            direct_humanfeedback_api = str_to_bool(os.environ["DIRECT_HUMANFEEDBACK_API"].strip())
            logger.info(
                "direct_humanfeedback_api overridden by env DIRECT_HUMANFEEDBACK_API=%s -> %s",
                os.environ["DIRECT_HUMANFEEDBACK_API"], direct_humanfeedback_api,
            )
        self.direct_humanfeedback_api = direct_humanfeedback_api if direct_humanfeedback_api is not None else True

        # Task keys to omit entirely from the pipeline (e.g. ["judge_task", "humanfeedback_task"]).
        # Filtered out by _get_ordered_agent_task_pairs so they never run.
        # Env var SKIP_STAGES=judge_task,humanfeedback_task (comma-separated) overrides the
        # constructor argument. Useful to control partial pipeline runs from a .env file
        # without changing CLI flags.
        if "SKIP_STAGES" in os.environ:
            _env_ss = [s.strip() for s in os.environ["SKIP_STAGES"].split(",") if s.strip()]
            if _env_ss:
                skip_stages = _env_ss
                logger.info("skip_stages overridden by env SKIP_STAGES=%s", os.environ["SKIP_STAGES"])
        self.skip_stages: List[str] = list(skip_stages) if skip_stages else []

        # Max reasoning iterations per CrewAI agent run.
        # None = use CrewAI's built-in default (20).
        # Env var AGENT_MAX_ITER=<int> overrides the constructor argument.
        if "AGENT_MAX_ITER" in os.environ:
            try:
                agent_max_iter = int(os.environ["AGENT_MAX_ITER"])
                logger.info("agent_max_iter overridden by env AGENT_MAX_ITER=%d", agent_max_iter)
            except ValueError:
                logger.warning(
                    "AGENT_MAX_ITER env var is not a valid integer (%r) — ignoring",
                    os.environ["AGENT_MAX_ITER"],
                )
        self.agent_max_iter: Optional[int] = agent_max_iter

        # Max wall-clock seconds per agent run. Env var AGENT_MAX_EXECUTION_TIME=<int>.
        if "AGENT_MAX_EXECUTION_TIME" in os.environ:
            try:
                agent_max_execution_time = int(os.environ["AGENT_MAX_EXECUTION_TIME"])
                logger.info("agent_max_execution_time overridden by env AGENT_MAX_EXECUTION_TIME=%d", agent_max_execution_time)
            except ValueError:
                logger.warning("AGENT_MAX_EXECUTION_TIME env var is not a valid integer (%r) — ignoring", os.environ["AGENT_MAX_EXECUTION_TIME"])
        self.agent_max_execution_time: Optional[int] = agent_max_execution_time

        # Max agent-level retries on recoverable errors. Env var AGENT_MAX_RETRY_LIMIT=<int>.
        if "AGENT_MAX_RETRY_LIMIT" in os.environ:
            try:
                agent_max_retry_limit = int(os.environ["AGENT_MAX_RETRY_LIMIT"])
                logger.info("agent_max_retry_limit overridden by env AGENT_MAX_RETRY_LIMIT=%d", agent_max_retry_limit)
            except ValueError:
                logger.warning("AGENT_MAX_RETRY_LIMIT env var is not a valid integer (%r) — ignoring", os.environ["AGENT_MAX_RETRY_LIMIT"])
        self.agent_max_retry_limit: Optional[int] = agent_max_retry_limit

        # User-supplied model context window override (tokens). When set, overrides the
        # auto-detected value from model_context.py for all downstream chunk sizing.
        # Useful when the model is not in the built-in registry or behind a custom proxy.
        self.model_context_window = model_context_window
        # Max input size (chars) for downstream agents (alignment, judge, humanfeedback) to avoid context limit.
        # Default ~80k chars (~20k tokens); 128k tokens ≈ 512k chars.
        self.downstream_max_input_chars = downstream_max_input_chars if downstream_max_input_chars is not None else 80_000
        # Cap extraction chunk size so (chunk + prompt) stays under model context; token limits vary by model.
        # Default 25000 chars (~6k tokens) leaves room for task prompt on 128k models. None = no cap.
        self.max_extraction_chunk_chars = max_extraction_chunk_chars if max_extraction_chunk_chars is not None else 25_000
        self.return_full_pipeline_details = return_full_pipeline_details
        self.stage_output_dir = stage_output_dir  # defaults to cwd; set to None to disable stage file output
        self.agent_feedback_config = agent_feedback_config or {}
        # Human-in-the-loop for feedback before humanfeedback_agent (see humanloop.py)
        self.human_loop = HumanInTheLoop(
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=self.agent_feedback_config,
        )

        # Initialize enhanced context management (always enabled)
        # Token limit based on model context (128k tokens ≈ 512k chars, use ~100k tokens = ~400k chars)
        self.token_limit = 100000  # Conservative limit for 128k token models
        self.agent_context = AgentContext(max_tokens=self.token_limit)
        self.shared_memory = ThreadSafeMemory()
        self.context_manager = ContextWindowManager(
            max_tokens=self.token_limit,
            reserve_tokens=2000,  # Reserve for prompts
        )

        logger.debug("Context management initialized (token_limit=%d)", self.token_limit)

        # Cache for pipeline-level task type: detected once at extraction phase,
        # reused for all downstream stages within the same pipeline run.
        self._pipeline_task_type: Optional[str] = None





    async def run_agent_task(
        self,
        agent_key: str,
        task_key: str,
        text: Optional[str] = None,
        pydantic_output_class: Optional[Any] = None,
        chunk_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        post_process: Optional[Any] = None,
        input_key: str = "input_text",
        default_result: Optional[Dict[str, Any]] = None,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a specific agent-task combination directly with full control.

        Tools are selected per agent/task via _get_detected_task_type and get_tools_for_agent.
        Supports chunking and post-processing for extraction; optional extra_inputs for multi-placeholder tasks.

        Args:
            agent_key: Key for the agent in agent_config
            task_key: Key for the task in task_config
            text: Input text to process (uses self.source_text if None)
            pydantic_output_class: Optional Pydantic class for structured output
            chunk_size: Maximum chunk size (None = no chunking, uses self.chunk_size if enabled)
            max_workers: Maximum parallel workers (None = auto, uses self.max_workers if set)
            post_process: Optional post-processing function
            input_key: Key to use in crew inputs dict
            default_result: Default result structure if parsing fails
            extra_inputs: Optional extra key-value inputs for crew (e.g. modification_context, user_feedback_text)

        Returns:
            Dict with results, raw_results, and errors
        """
        start_time = time.time()
        logger.info(f"Running agent '{agent_key}' with task '{task_key}'")

        # Use provided text or fall back to instance source_text
        if text is None:
            text = self.source_text

        # Detect task type; attach tools only for stages that use them (e.g. extractor_agent)
        # and also pass agent and task configuration for tools e.g., for using the llm call ...
        task_type = self._get_detected_task_type(agent_key, task_key)
        tools = get_tools_for_agent(
            agent_key,
            task_type,
            agent_config=self.agent_config,
            task_config=self.task_config,
            task_key=task_key,
        )

        # Initialize agent and task with dynamic tools
        agent, task = self._initialize_agent_and_task(
            agent_key=agent_key,
            task_key=task_key,
            pydantic_output_class=pydantic_output_class,
            tools=tools,
        )

        if not agent or not task:
            logger.error(f"Failed to initialize {agent_key}/{task_key}")
            return {
                "results": [],
                "raw_results": [],
                "errors": [{"scope": "initialization", "index": None, "error": f"Failed to initialize {agent_key}/{task_key}"}],
            }

        # Create a minimal crew just for this agent-task pair
        has_memory = any([self.long_term_memory, self.short_term_memory, self.entity_memory])
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            tracing=False,  # Disable tracing to reduce log noise
            memory=has_memory,
            long_term_memory_config=self.long_term_memory if self.long_term_memory else None,
            short_term_memory=self.short_term_memory if self.short_term_memory else None,
            entity_memory=self.entity_memory if self.entity_memory else None,
        )

        # Determine chunk size and max_workers
        if chunk_size is None:
            if self.enable_chunking:
                chunk_size = self.chunk_size
            else:
                chunk_size = None  # No chunking

        if max_workers is None:
            max_workers = self.max_workers

        # Use async execution for better concurrency when chunking is enabled
        # For resource/structured_extraction alignment: extract all aligned_resources blobs and return as list
        # so app.py can combine them in postprocessing (merge_downstream_chunk_results_with_provenance).
        # NER, extraction (e.g. pdf2_reproschema), judge, humanfeedback: single parse only.
        pick_richest = task_key == "alignment_task" and task_type in ("resource", "structured_extraction")
        if self.enable_chunking and chunk_size:
            result = await run_crew_extraction_async(
                crew=crew,
                text=text,
                chunk_size=chunk_size,
                max_workers=max_workers,
                input_key=input_key,
                default_result=default_result or {},
                post_process=post_process,
                extra_inputs=extra_inputs,
                max_chunk_chars=self.max_extraction_chunk_chars,
                agent_context=self.agent_context,
                shared_memory=self.shared_memory,
                context_manager=self.context_manager,
                pick_richest_alignment=pick_richest,
            )
        else:
            # Use synchronous execution for single chunk or no chunking
            result = run_crew_extraction(
                crew=crew,
                text=text,
                chunk_size=None,
                max_workers=max_workers,
                input_key=input_key,
                default_result=default_result or {},
                post_process=post_process,
                extra_inputs=extra_inputs,
                max_chunk_chars=self.max_extraction_chunk_chars,
                agent_context=self.agent_context,
                shared_memory=self.shared_memory,
                context_manager=self.context_manager,
                pick_richest_alignment=pick_richest,
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Agent '{agent_key}' completed in {elapsed_time:.2f} seconds")

        result["elapsed_time"] = elapsed_time
        result["agent_key"] = agent_key
        result["task_key"] = task_key
        return result

    async def kickoff(
        self,
        agent_key: Optional[str] = None,
        task_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a SINGLE agent-task pair (not the full pipeline).

        This method runs only ONE agent-task pair from your config. For the full
        multi-agent pipeline (extraction → alignment → judge → humanfeedback),
        use information_extraction_task() instead.

        Args:
            agent_key: Optional specific agent key (auto-discovered if None - uses first pair)
            task_key: Optional specific task key (auto-discovered if None - uses first pair)

        Returns:
            Dict with results from the single agent execution

        Note:
            For full pipeline execution, use information_extraction_task() instead.
        """
        # Setup timing logger for this execution
        timing_logger, timing_log_file = setup_timing_logger()
        total_start_time = time.time()

        timing_logger.info("=" * 100)
        timing_logger.info("TIMING LOG - Agent Execution")
        timing_logger.info("=" * 100)
        timing_logger.info(f"Timing log file: {timing_log_file}")
        timing_logger.info(f"Chunking enabled: {self.enable_chunking}, Chunk size: {self.chunk_size}, Max workers: {self.max_workers}")
        timing_logger.info("-" * 100)

        try:
            logger.info("Starting StructSenseFlow extraction...")

            # Timing: Config loading (already done in __init__, but log it)
            config_start = time.time()
            config_time = time.time() - config_start
            timing_logger.info(f"Config loading: {config_time:.3f}s (already loaded in __init__)")

            # Timing: Agent/task initialization
            init_start = time.time()
            effective_chunk_size = self.chunk_size if self.enable_chunking else None

            # Find agent-task pairs from config if not provided
            if agent_key is None or task_key is None:
                agent_task_pairs = []
                for task_key_iter, task_data in self.task_config.items():
                    if isinstance(task_data, dict) and "agent_id" in task_data:
                        agent_key_iter = task_data["agent_id"]
                        if agent_key_iter in self.agent_config:
                            agent_task_pairs.append((agent_key_iter, task_key_iter))
                            logger.info(f"Found agent-task pair: {agent_key_iter} -> {task_key_iter}")

                if not agent_task_pairs:
                    logger.warning("No agent-task pairs found in config. Trying default extractor_agent/extraction_task")
                    if "extractor_agent" in self.agent_config and "extraction_task" in self.task_config:
                        agent_task_pairs = [("extractor_agent", "extraction_task")]
                    else:
                        return {"error": "No valid agent-task pairs found in config", "entities": [], "key_terms": []}

                # Use first pair if not specified
                if agent_key is None or task_key is None:
                    agent_key, task_key = agent_task_pairs[0]

            logger.info(f"Running agent '{agent_key}' with task '{task_key}'")

            # Dynamic task type detection; tools only for extractor stage (alignment/judge/human_feedback get none)
            task_type = self._get_detected_task_type(agent_key, task_key)
            tools = get_tools_for_agent(
                agent_key,
                task_type,
                agent_config=self.agent_config,
                task_config=self.task_config,
                task_key=task_key,
            )

            # Map taxonomy task_type to post-processor key (ner, resource, extraction)
            if task_type == "ner":
                post_process_key = "ner"
            elif task_type in ("resource", "structured_extraction"):
                post_process_key = "resource"
            else:
                post_process_key = "extraction"
            post_processor = get_post_processor(post_process_key)
            result_merger = get_result_merger(post_process_key)

            # Get default result based on task type
            default_result = self._get_default_result_for_task(task_type)

            # Initialize agent and task with dynamic tools
            agent, task = self._initialize_agent_and_task(
                agent_key=agent_key,
                task_key=task_key,
                pydantic_output_class=None,
                tools=tools,
            )

            if not agent or not task:
                logger.error(f"Failed to initialize {agent_key}/{task_key}")
                return {"error": f"Failed to initialize {agent_key}/{task_key}", "entities": [], "key_terms": []}

            init_time = time.time() - init_start
            timing_logger.info(f"Agent/task initialization: {init_time:.3f}s")

            # Timing: Memory initialization (already done in __init__, but log it)
            memory_start = time.time()
            has_memory = any([self.long_term_memory, self.short_term_memory, self.entity_memory])
            memory_time = time.time() - memory_start
            timing_logger.info(f"Memory initialization: {memory_time:.3f}s (already initialized in __init__, enabled: {has_memory})")

            # Timing: Crew creation
            crew_start = time.time()
            use_verbose = not (self.enable_chunking and effective_chunk_size)
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=use_verbose,
                tracing=False,
                memory=has_memory,
                long_term_memory_config=self.long_term_memory if self.long_term_memory else None,
                short_term_memory=self.short_term_memory if self.short_term_memory else None,
                entity_memory=self.entity_memory if self.entity_memory else None,
            )
            crew_time = time.time() - crew_start
            timing_logger.info(f"Crew creation: {crew_time:.3f}s")

            # Timing: Extraction execution
            extraction_start = time.time()
            timing_logger.info("-" * 100)
            timing_logger.info("Starting extraction execution...")

            # Use async execution for better concurrency when chunking is enabled
            if self.enable_chunking and effective_chunk_size:
                timing_logger.info("Using async execution (akickoff) for better concurrency...")
                result = await run_crew_extraction_async(
                    crew=crew,
                    text=self.source_text,
                    chunk_size=effective_chunk_size,
                    max_workers=self.max_workers,
                    input_key="input_text",
                    default_result=default_result,
                    post_process=post_processor,
                )
            else:
                # Use synchronous execution for single chunk or no chunking
                result = run_crew_extraction(
                    crew=crew,
                    text=self.source_text,
                    chunk_size=None,
                    max_workers=self.max_workers,
                    input_key="input_text",
                    default_result=default_result,
                    post_process=post_processor,
                )

            extraction_time = time.time() - extraction_start
            timing_logger.info(f"Extraction execution: {extraction_time:.3f}s")
            timing_logger.info(f"  - Chunks processed: {len(result.get('results', []))}")
            timing_logger.info(f"  - Errors: {len(result.get('errors', []))}")

            # Timing: Result merging
            merge_start = time.time()
            merged_results = result_merger(result["results"], self.source_text)
            merge_time = time.time() - merge_start
            timing_logger.info(f"Result merging: {merge_time:.3f}s")
            # Verifier: ensure entities, text, sentences present in source
            merged_results = verify_merged_result(merged_results, self.source_text, task_type)

            # Total time
            total_time = time.time() - total_start_time
            timing_logger.info("-" * 100)
            timing_logger.info("TIMING SUMMARY:")
            timing_logger.info(f"  Config loading:       {config_time:.3f}s ({config_time/total_time*100:.1f}%)")
            timing_logger.info(f"  Agent/task init:       {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
            timing_logger.info(f"  Memory initialization: {memory_time:.3f}s ({memory_time/total_time*100:.1f}%)")
            timing_logger.info(f"  Crew creation:         {crew_time:.3f}s ({crew_time/total_time*100:.1f}%)")
            timing_logger.info(f"  Extraction execution:  {extraction_time:.3f}s ({extraction_time/total_time*100:.1f}%)")
            timing_logger.info(f"  Result merging:       {merge_time:.3f}s ({merge_time/total_time*100:.1f}%)")
            timing_logger.info("-" * 100)
            timing_logger.info(f"TOTAL TIME: {total_time:.3f}s ({total_time/60:.2f} minutes)")
            timing_logger.info("=" * 100)
            timing_logger.info(f"Timing log saved to: {timing_log_file}")
            timing_logger.info("=" * 100)

            logger.info("#" * 100)
            logger.info(f"Extraction completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Timing details saved to: {timing_log_file}")
            logger.info("#" * 100)

            return {
                **merged_results,
                "errors": result["errors"],
                "task_type": task_type,
                "elapsed_time": total_time,
                "agent_key": agent_key,
                "task_key": task_key,
            }

        except Exception as e:
            logger.error(f"StructSenseFlow execution failed: {str(e)}")
            raise

    async def extraction(self) -> Dict[str, Any]:
        """
        Convenience method to run ONLY the extraction agent.

        This is a shortcut for kickoff("extractor_agent", "extraction_task").
        For the full multi-agent pipeline, use information_extraction_task() instead.

        Returns:
            Dict with extraction results only
        """
        return await self.kickoff(agent_key="extractor_agent", task_key="extraction_task")

    async def information_extraction_task(
        self,
        text: Optional[str] = None,
        modification_context: Optional[str] = None,
        user_feedback_text: Optional[str] = None,
        preloaded_stages: Optional[Dict[str, Any]] = None,
    ):
        """
        Run the FULL multi-agent pipeline with context management.

        Executes all agents in sequence: extraction → alignment → judge → humanfeedback
        Each stage receives context from previous stages with automatic token management.

        Args:
            text: Optional input text (uses self.source_text if None)
            modification_context: Optional context for modifications
            user_feedback_text: Optional user feedback for humanfeedback stage
            preloaded_stages: Optional dict mapping task_key → pre-loaded result dict.
                Stages present in this dict are skipped — their saved output is used
                directly as if the agent had just run.  Useful for resuming a pipeline
                after a crash or for re-running only a subset of stages, e.g.:

                    import json
                    with open("00_extractor_agent_extraction_task.json") as f:
                        extraction_result = json.load(f)

                    result = await flow.information_extraction_task(
                        preloaded_stages={"extraction_task": extraction_result}
                    )
                    # → alignment, judge, humanfeedback run normally;
                    #   extraction is skipped and the saved output is used instead.

        Returns:
            Dict with final results. By default (return_full_pipeline_details=False):
            - entities, resources, key_terms (final output only, no repetition)
            - errors, task_type, elapsed_time
            When return_full_pipeline_details=True also includes:
            - pipeline_stages, stage_timings, token_usage, context_management

        Note:
            This is the recommended method for production use. For single-agent
            execution, use kickoff() or extraction() instead.
        """
        if text is None:
            text = self.source_text

        start_time = time.time()
        logger.info("Starting structured information extraction (full pipeline)")

        # Reset cached task type so each pipeline run re-detects from the extraction stage.
        self._pipeline_task_type = None

        # Check Ollama health (optional - won't fail if unavailable)
        if not check_ollama_health():
            logger.warning("Ollama health check failed, continuing anyway")

        ordered_pairs = self._get_ordered_agent_task_pairs()
        if not ordered_pairs:
            logger.warning("No agent-task pairs found in config")
            return {
                "errors": [{"scope": "pipeline", "index": None, "error": "No agent-task pairs in config"}],
                "task_type": "extraction",
                "elapsed_time": time.time() - start_time,
            }

        # First-stage task type drives post-processor and merger (extraction only)
        first_agent_key, first_task_key = ordered_pairs[0]
        task_type = self._get_detected_task_type(first_agent_key, first_task_key)
        logger.info(f"Detected task type (first stage): {task_type}")

        if task_type == "ner":
            post_process_key = "ner"
        elif task_type == "resource":
            post_process_key = "resource"
        else:
            post_process_key = "extraction"
        post_processor = get_post_processor(post_process_key)
        result_merger = get_result_merger(post_process_key)
        default_result = self._get_default_result_for_task(task_type)

        all_errors = []
        pipeline_stages = {}
        stage_timings = {}  # Track execution time for each stage
        prev_output = None
        humanfeedback_aborted_by_user = False  # Option 4: set when user aborts feedback step (save judge result)
        human_approved_skip_concept_mapping = False  # Option 1: when user approves, save result as-is without concept mapping

        for idx, (agent_key, task_key) in enumerate(ordered_pairs):
            stage_start_time = time.time()  # Start timing this stage
            is_first_stage = idx == 0
            input_key = PIPELINE_INPUT_KEY_MAP.get(task_key, "input_text")
            extra_inputs = None  # Initialize for each stage
            humanfeedback_approved_skip_run = False  # Option 1 or 4: skip running humanfeedback agent

            logger.info("=" * 80)
            logger.info(f"STAGE {idx + 1}/{len(ordered_pairs)}: {agent_key} / {task_key}")
            logger.info("=" * 80)

            # ------------------------------------------------------------------
            # PRELOADED STAGE — skip the agent run and use saved output directly.
            # The preloaded result is treated exactly as if the agent had just run:
            # it is stored in pipeline_stages, set as prev_output, and the loop
            # continues to the next stage.
            # ------------------------------------------------------------------
            if preloaded_stages and task_key in preloaded_stages:
                preloaded = preloaded_stages[task_key]
                pipeline_stages[task_key] = preloaded
                prev_output = preloaded
                entity_count = len(preloaded.get("entities") or []) if isinstance(preloaded, dict) else "n/a"
                logger.info(
                    "[preloaded] Skipping %s/%s — using saved output "
                    "(entities=%s, keys=%s)",
                    agent_key, task_key, entity_count,
                    list(preloaded.keys()) if isinstance(preloaded, dict) else "?",
                )
                stage_timings[task_key] = 0.0
                continue

            # Clear alignment-stage tool outputs so we only capture this stage's concept mapping (for extraction)
            if task_key == "alignment_task":
                clear_alignment_tool_outputs()

            # ------------------------------------------------------------------
            # Fast alignment bypass: skip the alignment LLM entirely.
            #
            # When enabled, Python calls the concept mapping tool directly with
            # all entity texts in one batch (the local service supports 4000
            # concepts/request), injects the results into the extraction output,
            # and stores that as the alignment result — no LLM call needed.
            #
            # This is correct because the alignment LLM's only real job is
            # ontology mapping, which the pre-compute + injection already does.
            # The LLM typically only maps 3–10 terms per call anyway and adds no
            # other value for NER tasks.
            #
            # Auto-enable when:
            #   - task_key == "alignment_task"
            #   - task_type is "ner" or "keyphrase_extraction"
            #   - CONCEPT_MAPPING_BACKEND == "local" (supports batch of 4000)
            #
            # Override with skip_alignment_llm=True  (force skip)
            #                 skip_alignment_llm=False (force LLM)
            # ------------------------------------------------------------------
            if task_key == "alignment_task" and prev_output is not None:
                _cm_backend_now = os.getenv("CONCEPT_MAPPING_BACKEND", "local").strip().lower()
                # Auto-skip also applies to resource and structured_extraction task types.
                # Resource alignment (like NER alignment) only needs ontology mapping — the
                # LLM adds no value over a direct batch tool call when the local backend is used.
                _auto_skip = (
                    task_type in ("ner", "keyphrase_extraction", "resource", "structured_extraction")
                    and _cm_backend_now == "local"
                )
                _do_skip = (
                    self.skip_alignment_llm is True
                    or (self.skip_alignment_llm is None and _auto_skip)
                )
                if _do_skip:
                    import copy as _copy
                    stage_start_time = time.time()
                    logger.info(
                        "[alignment_task] Fast-alignment bypass active — "
                        "calling concept mapping tool directly, skipping alignment LLM"
                    )

                    # Collect texts for concept mapping.
                    # For NER/keyphrase: entity text from entities, extracted_terms, key_terms.
                    # For resource/structured_extraction: resource name from resources list.
                    # Both paths deduplicate before calling the tool.
                    _fa_texts: list = []
                    def _fa_collect(ent):
                        if isinstance(ent, dict):
                            t = ent.get("entity") or ent.get("text") or ent.get("name")
                            if isinstance(t, str) and t.strip():
                                _fa_texts.append(t.strip())

                    extraction_output_fa = prev_output
                    if task_type in ("ner", "keyphrase_extraction"):
                        # NER/keyphrase: collect entity texts
                        for _e in extraction_output_fa.get("entities") or []:
                            _fa_collect(_e)
                        for _raw_key in ("extracted_terms", "aligned_ner_terms"):
                            for _e in _flatten_container_to_list(extraction_output_fa.get(_raw_key) or []):
                                _fa_collect(_e)
                        for _kt in extraction_output_fa.get("key_terms") or []:
                            if isinstance(_kt, str) and _kt.strip():
                                _fa_texts.append(_kt.strip())
                            elif isinstance(_kt, dict):
                                _t = _kt.get("term") or _kt.get("text") or _kt.get("entity")
                                if isinstance(_t, str) and _t.strip():
                                    _fa_texts.append(_t.strip())
                    else:
                        # Resource/structured_extraction: collect resource names from all resource
                        # list keys — the extraction stage may store them under different keys
                        # before alignment normalises them.
                        for _rkey in ("resources", "aligned_resources", "extracted_resources"):
                            for _r in extraction_output_fa.get(_rkey) or []:
                                if isinstance(_r, dict):
                                    _rname = _r.get("name") or _r.get("resource_name")
                                    if isinstance(_rname, str) and _rname.strip():
                                        _fa_texts.append(_rname.strip())

                    # Deduplicate
                    _fa_seen: set = set()
                    _fa_unique = [t for t in _fa_texts if not (t in _fa_seen or _fa_seen.add(t))]
                    logger.info("[alignment_task] Fast-alignment: %d unique terms to map", len(_fa_unique))

                    # Call concept mapping tool directly
                    if _fa_unique:
                        try:
                            from utils.conceptmappinglocal import ConceptMappingLocalTool as _CMToolFA
                            _CMToolFA()._run(text=_fa_unique)
                        except Exception as _fa_exc:
                            logger.warning("[alignment_task] Fast-alignment tool call failed: %s", _fa_exc)

                    # Build synthetic alignment result: deep copy extraction output,
                    # inject concept mapping results, add provenance
                    _aligned = _copy.deepcopy(extraction_output_fa)
                    _aligned["alignment_method"] = "direct_tool_call"
                    _aligned["alignment_llm_skipped"] = True

                    _fa_session = get_alignment_tool_outputs()
                    if _fa_session:
                        if task_type in ("ner", "keyphrase_extraction"):
                            # NER/keyphrase: inject ontology fields into entity dicts
                            _fa_entities = _aligned.get("entities") or []
                            if _fa_entities:
                                _fa_enriched = inject_alignment_concept_mapping_into_ner_entities(
                                    _fa_entities, _fa_session
                                )
                                logger.info(
                                    "[alignment_task] Fast-alignment enriched %d entities with ontology fields",
                                    _fa_enriched,
                                )
                                for _ent in _fa_entities:
                                    if isinstance(_ent, dict) and _ent.get("ontology_id"):
                                        _ent.setdefault("concept_mapping_provenance", "tool")
                        else:
                            # Resource/structured_extraction: inject ontology fields into resource dicts.
                            # Resources are normalised to the "resources" key by promote_stage_output_to_canonical
                            # later in this block, but we inject into whichever keys are present now so
                            # the fields survive the promote call.
                            _fa_res_enriched = 0
                            for _rkey in ("resources", "aligned_resources", "extracted_resources"):
                                _fa_resources = _aligned.get(_rkey) or []
                                if _fa_resources:
                                    _fa_res_enriched += inject_alignment_concept_mapping_into_resources(
                                        _fa_resources, _fa_session
                                    )
                            if _fa_res_enriched:
                                logger.info(
                                    "[alignment_task] Fast-alignment enriched %d resources with ontology fields",
                                    _fa_res_enriched,
                                )
                            for _rkey in ("resources", "aligned_resources", "extracted_resources"):
                                for _res in _aligned.get(_rkey) or []:
                                    if isinstance(_res, dict) and _res.get("ontology_id"):
                                        _res.setdefault("concept_mapping_provenance", "tool")

                    promote_stage_output_to_canonical(_aligned, task_type)

                    prev_output = _aligned
                    pipeline_stages[task_key] = _aligned
                    stage_elapsed = time.time() - stage_start_time
                    stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
                    logger.info(
                        "[alignment_task] Fast-alignment completed in %.2fs (LLM call skipped)",
                        stage_elapsed,
                    )

                    # Save stage output to disk
                    if self.stage_output_dir and pipeline_stages.get(task_key) is not None:
                        try:
                            os.makedirs(self.stage_output_dir, exist_ok=True)
                            _stage_fname = f"{idx:02d}_{agent_key}_{task_key}.json"
                            _stage_path = os.path.join(self.stage_output_dir, _stage_fname)
                            with open(_stage_path, "w") as _sf:
                                json.dump(pipeline_stages[task_key], _sf, indent=2, default=str)
                            logger.info("[alignment_task] Stage output saved to %s", _stage_path)
                        except Exception as _save_exc:
                            logger.warning("[alignment_task] Failed to save stage output: %s", _save_exc)

                    continue  # skip the alignment LLM run entirely

            # ------------------------------------------------------------------
            # Fast judge bypass: skip the judge LLM entirely.
            #
            # When enabled, Python deep-copies the alignment output and injects
            # default judge_score=1.0 and remarks="auto-approved" into every entity
            # dict — no LLM call is made.  The result is stored as the judge stage
            # output and the pipeline continues.
            #
            # Use this when:
            #   - you trust the alignment output and do not need quality scoring, or
            #   - you want the fastest possible run (extraction + alignment only, but
            #     humanfeedback still needs a judge stage output to work from)
            #
            # Enable with skip_judge_llm=True (constructor / Python API)
            #             SKIP_JUDGE_LLM=true (.env or environment)
            #             --skip_judge_llm true (CLI)
            # ------------------------------------------------------------------
            if task_key == "judge_task" and prev_output is not None and self.skip_judge_llm:
                import copy as _jcopy
                stage_start_time = time.time()
                logger.info(
                    "[judge_task] Fast-judge bypass active — "
                    "injecting default judge_score/remarks, skipping judge LLM"
                )
                _judged = _jcopy.deepcopy(prev_output)
                _judged["judge_method"] = "auto_approved"
                _judged["judge_llm_skipped"] = True

                # Inject default judge_score and remarks into every entity dict
                _jfa_count = 0
                for _jent in _judged.get("entities") or []:
                    if isinstance(_jent, dict):
                        _jent.setdefault("judge_score", 1.0)
                        _jent.setdefault("remarks", "auto-approved: judge LLM skipped")
                        _jfa_count += 1
                logger.info(
                    "[judge_task] Fast-judge: injected default scores into %d entities", _jfa_count
                )

                promote_stage_output_to_canonical(_judged, task_type)

                prev_output = _judged
                pipeline_stages[task_key] = _judged
                stage_elapsed = time.time() - stage_start_time
                stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
                logger.info(
                    "[judge_task] Fast-judge completed in %.2fs (LLM call skipped)", stage_elapsed
                )

                if self.stage_output_dir and pipeline_stages.get(task_key) is not None:
                    try:
                        os.makedirs(self.stage_output_dir, exist_ok=True)
                        _stage_fname = f"{idx:02d}_{agent_key}_{task_key}.json"
                        _stage_path = os.path.join(self.stage_output_dir, _stage_fname)
                        with open(_stage_path, "w") as _sf:
                            json.dump(pipeline_stages[task_key], _sf, indent=2, default=str)
                        logger.info("[judge_task] Stage output saved to %s", _stage_path)
                    except Exception as _save_exc:
                        logger.warning("[judge_task] Failed to save stage output: %s", _save_exc)

                continue  # skip the judge LLM run entirely

            # ------------------------------------------------------------------
            # DIRECT-API JUDGE BYPASS
            # ------------------------------------------------------------------
            # Replace the CrewAI judge agent with a direct OpenAI-compatible API
            # call.  The CrewAI path adds significant overhead: one LLM call for
            # the agent iteration, a second "forced final answer" call when
            # max_iter=1, and internal retry scaffolding.  For large payloads
            # that get split into N chunks this overhead multiplies by N and can
            # turn a simple scoring step into a 10+ minute bottleneck.
            #
            # The direct-API path sends each batch of entities to the LLM once,
            # parses the JSON response, and injects judge_score / remarks.  There
            # is no agent loop, no tool scaffolding, and no forced-final-answer.
            #
            # Enable (default): direct_judge_api=True  /  DIRECT_JUDGE_API=true
            # Disable (fallback to CrewAI): direct_judge_api=False / DIRECT_JUDGE_API=false
            # ------------------------------------------------------------------
            if task_key == "judge_task" and prev_output is not None and self.direct_judge_api:
                import copy as _djcopy
                stage_start_time = time.time()
                logger.info(
                    "[judge_task] Direct-API judge active — bypassing CrewAI agent"
                )

                # Resolve LLM config for the judge agent from its agent_config entry
                _djllm = self.agent_config.get(agent_key, {}).get("llm", {})
                if isinstance(_djllm, dict) and _djllm.get("model"):
                    _dj_model = _djllm["model"]
                    _dj_base_url = _djllm.get("base_url") or "https://openrouter.ai/api/v1"
                    logger.info("[judge_task] Direct-API: model=%s (from agent_config[%s])", _dj_model, agent_key)
                else:
                    _dj_model = "openai/gpt-4o-mini"
                    _dj_base_url = "https://openrouter.ai/api/v1"
                    logger.warning(
                        "[judge_task] Direct-API: no LLM config found for agent_key=%s — using fallback model=%s",
                        agent_key, _dj_model,
                    )
                # OpenRouter expects model ID without the "openrouter/" prefix
                if "openrouter" in _dj_base_url.lower() and _dj_model.startswith("openrouter/"):
                    _dj_model = _dj_model.replace("openrouter/", "", 1)

                _dj_api_key = (
                    os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or ""
                )
                _djudged = _djcopy.deepcopy(prev_output)
                _dj_entities = _djudged.get("entities") or []

                # Determine primary work-item list and its key (entities for NER/keyphrase,
                # resources for resource/structured_extraction tasks)
                _dj_items_key = (
                    "entities"
                    if _djudged.get("entities")
                    else "resources"
                    if _djudged.get("resources")
                    else "entities"
                )
                _dj_entities = _djudged.get(_dj_items_key) or []

                # Token-aware batch size — reuse the same compute_downstream_chunk_size
                # logic used by the CrewAI chunking path so sizing is consistent.
                _dj_workers = self.max_workers or 4
                _dj_extraction_chunk_count = None
                for _pk in pipeline_stages:
                    _ps = pipeline_stages.get(_pk)
                    if isinstance(_ps, dict) and "_extraction_chunk_count" in _ps:
                        _dj_extraction_chunk_count = _ps["_extraction_chunk_count"]
                        break
                _dj_task_cfg = (
                    self.task_config.get(task_key, {})
                    if isinstance(self.task_config, dict)
                    else {}
                )
                _dj_prompt_overhead = estimate_agent_prompt_tokens(
                    agent_config=self.agent_config.get(agent_key, {}),
                    task_config=_dj_task_cfg,
                )
                _dj_batch, _dj_should_batch = compute_downstream_chunk_size(
                    payload=_djudged,
                    model_str=_dj_model,
                    max_workers=_dj_workers,
                    extraction_chunk_count=_dj_extraction_chunk_count,
                    explicit_chunk_size=self.downstream_chunk_size,
                    context_window_override=self.model_context_window,
                    prompt_overhead_tokens=_dj_prompt_overhead,
                )
                # If the whole payload fits in one call, use a single batch
                if not _dj_should_batch:
                    _dj_batch = len(_dj_entities) or 1

                # ── Direct-API output-size guard ─────────────────────────────
                # compute_downstream_chunk_size only checks INPUT token fit.
                # For judge/humanfeedback the OUTPUT is the same size as the
                # input (all items echoed back + small added fields), so the
                # model's output token limit can be hit even when the input fits.
                #
                # Two caps applied (take the stricter of the two):
                #
                #  1. Output token cap: estimate tokens/item from payload size;
                #     cap batch so expected output ≤ DIRECT_API_MAX_OUTPUT_TOKENS.
                #     Default 32 768 — safe for models with 32 k output limit.
                #
                #  2. Min-chunks: always split into at least DIRECT_API_MIN_CHUNKS
                #     parallel batches regardless of token math (default 8).
                #     This avoids a single giant call even on very large models.
                _dj_out_cap = int(os.environ.get("DIRECT_API_MAX_OUTPUT_TOKENS", "32768"))
                _dj_min_chunks = int(os.environ.get("DIRECT_API_MIN_CHUNKS", "8"))
                _dj_n_items = len(_dj_entities) or 1
                # tokens/item ≈ chars/item (1 char ≈ 1 token for structured JSON)
                _dj_tok_per_item = max(1, len(json.dumps(_djudged, ensure_ascii=False)) / _dj_n_items)
                _dj_out_capped = max(1, int(_dj_out_cap / _dj_tok_per_item))
                _dj_min_chunk_batch = max(1, math.ceil(_dj_n_items / _dj_min_chunks))
                _dj_effective = min(_dj_batch, _dj_out_capped, _dj_min_chunk_batch)
                if _dj_effective < _dj_batch:
                    logger.info(
                        "[judge_task] Direct-API output guard: output_cap=%d tok "
                        "(%.1f tok/item → max %d items), min_chunks=%d (→ max %d items) "
                        "→ batch %d → %d items",
                        _dj_out_cap, _dj_tok_per_item, _dj_out_capped,
                        _dj_min_chunks, _dj_min_chunk_batch,
                        _dj_batch, _dj_effective,
                    )
                _dj_batch = _dj_effective

                logger.info(
                    "[judge_task] Direct-API batch size: %d items/batch "
                    "(should_batch=%s, model=%s)",
                    _dj_batch, _dj_should_batch, _dj_model,
                )

                async def _judge_entities_direct(_ents, _model, _base_url, _key, _batch_idx, _items_key, _attempt=0):
                    """Score a flat list of item dicts via a single direct LLM call.

                    Retry policy:
                    - JSONDecodeError (truncated output): split batch in half and recurse
                      so each half is a smaller independent call.  Recurses down to a
                      batch of 1 before giving up.
                    - Other API / network errors: retry up to 3 times with exponential
                      back-off (1 s, 2 s, 4 s).
                    """
                    _MAX_RETRIES = 3
                    from openai import AsyncOpenAI as _AsyncOpenAI
                    _c = _AsyncOpenAI(base_url=_base_url, api_key=_key)
                    _sys = (
                        "You are a neuroscience NER quality judge. "
                        f"For EVERY item in the input \"{_items_key}\" list add exactly two fields: "
                        "\"judge_score\" (float 0.0-1.0, where 1.0=perfect alignment) and "
                        "\"remarks\" (string, brief explanation). "
                        "Preserve ALL existing fields unchanged. "
                        f"Return ONLY valid JSON: {{\"{_items_key}\": [...]}}. No markdown, no prose."
                    )
                    _payload = json.dumps({_items_key: _ents}, ensure_ascii=False)

                    # Log this call into the llm_calls log (same counter as CrewAI calls)
                    with _llm_call_lock:
                        _llm_call_count["n"] += 1
                        _call_n = _llm_call_count["n"]
                    _dj_llm_log = logging.getLogger("llm_calls")
                    if _dj_llm_log.handlers:
                        _dj_llm_log.info("=" * 70)
                        _dj_llm_log.info(f"[LLM CALL #{_call_n}]  (direct API — no CrewAI agent)")
                        _dj_llm_log.info(f"Agent          : {agent_key} / judge_task (direct)")
                        _dj_llm_log.info(f"Model          : {_dj_model}")
                        _dj_llm_log.info(f"Batch          : {_batch_idx + 1}  ({len(_ents)} entities, attempt {_attempt + 1})")
                        _dj_llm_log.info(f"Payload preview: {_payload[:300]}")
                        _dj_llm_log.info("=" * 70)
                        _flush_llm_logger()

                    try:
                        _r = await _c.chat.completions.create(
                            model=_model,
                            messages=[
                                {"role": "system", "content": _sys},
                                {"role": "user", "content": _payload},
                            ],
                        )
                        _raw = (_r.choices[0].message.content or "").strip()

                        # Log the response
                        if _dj_llm_log.handlers:
                            _dj_llm_log.info(f"[LLM RESPONSE PREVIEW]  (call #{_call_n})")
                            _dj_llm_log.info(_raw[:500])
                            _dj_llm_log.info("=" * 70)
                            _flush_llm_logger()

                        # Strip markdown fences if present
                        if _raw.startswith("```"):
                            _parts = _raw.split("```")
                            _raw = _parts[1] if len(_parts) > 1 else _raw
                            if _raw.startswith("json"):
                                _raw = _raw[4:]
                        _stripped = _raw.strip()
                        try:
                            _parsed = json.loads(_stripped)
                        except json.JSONDecodeError as _parse_ex:
                            if "Extra data" in str(_parse_ex):
                                # LLM appended trailing content after valid JSON — extract first object only
                                _parsed, _ = json.JSONDecoder().raw_decode(_stripped)
                                logger.debug(
                                    "[judge_task] Direct API batch %d: recovered from 'Extra data' via raw_decode",
                                    _batch_idx + 1,
                                )
                            else:
                                raise  # truncated JSON — handled by outer except below
                        return _parsed.get(_items_key) or _ents
                    except json.JSONDecodeError as _ex:
                        # Truncated JSON (Unterminated string, Expecting value, etc.) — split batch in half
                        logger.warning(
                            "[judge_task] Direct API batch %d JSON parse error (attempt %d/%d, %d items): %s — splitting batch",
                            _batch_idx + 1, _attempt + 1, _MAX_RETRIES, len(_ents), _ex,
                        )
                        if len(_ents) > 1:
                            _mid = len(_ents) // 2
                            _l_res, _r_res = await asyncio.gather(
                                _judge_entities_direct(_ents[:_mid], _model, _base_url, _key, _batch_idx, _items_key, 0),
                                _judge_entities_direct(_ents[_mid:], _model, _base_url, _key, _batch_idx, _items_key, 0),
                            )
                            return _l_res + _r_res
                        logger.warning("[judge_task] Direct API single-item batch still failed — returning original item")
                        return _ents
                    except Exception as _ex:
                        if _attempt + 1 < _MAX_RETRIES:
                            _wait = 2 ** _attempt
                            logger.warning(
                                "[judge_task] Direct API batch %d error (attempt %d/%d, retry in %ds): %s",
                                _batch_idx + 1, _attempt + 1, _MAX_RETRIES, _wait, _ex,
                            )
                            await asyncio.sleep(_wait)
                            return await _judge_entities_direct(_ents, _model, _base_url, _key, _batch_idx, _items_key, _attempt + 1)
                        logger.warning("[judge_task] Direct API batch %d failed after %d attempts: %s", _batch_idx + 1, _MAX_RETRIES, _ex)
                        return _ents

                if _dj_entities and _dj_api_key:
                    _begin_stage_llm_tracking(task_key, agent_key)
                    try:
                        # Split into batches and run all in parallel
                        _dj_batches = [
                            _dj_entities[_dj_i: _dj_i + _dj_batch]
                            for _dj_i in range(0, len(_dj_entities), _dj_batch)
                        ]
                        _dj_n_batches = len(_dj_batches)
                        logger.info(
                            "[judge_task] Direct-API judge: %d items → %d batches × %d, running in parallel",
                            len(_dj_entities), _dj_n_batches, _dj_batch,
                        )
                        _dj_batch_results = await asyncio.gather(*[
                            _judge_entities_direct(
                                _dj_batches[_bi], _dj_model, _dj_base_url, _dj_api_key,
                                _bi, _dj_items_key,
                            )
                            for _bi in range(_dj_n_batches)
                        ])
                        _dj_scored: list = []
                        for _dj_result in _dj_batch_results:
                            for _e in _dj_result:
                                if isinstance(_e, dict):
                                    _e.setdefault("judge_score", 0.8)
                                    _e.setdefault("remarks", "direct-api: default score")
                            _dj_scored.extend(_dj_result)
                        _djudged[_dj_items_key] = _dj_scored
                        _djudged["judge_method"] = "direct_api"
                        _end_stage_llm_tracking(task_key, _dj_n_batches, time.time() - stage_start_time)
                        logger.info(
                            "[judge_task] Direct-API judge scored %d %s in %.2fs",
                            len(_dj_scored), _dj_items_key, time.time() - stage_start_time,
                        )
                    except Exception as _dj_exc:
                        logger.warning(
                            "[judge_task] Direct-API judge failed (%s); injecting defaults", _dj_exc
                        )
                        for _e in _dj_entities:
                            if isinstance(_e, dict):
                                _e.setdefault("judge_score", 0.8)
                                _e.setdefault("remarks", "direct-api: error, default score")
                        _djudged["judge_method"] = "direct_api_fallback"
                else:
                    # No API key or no items: inject default scores
                    for _e in _dj_entities:
                        if isinstance(_e, dict):
                            _e.setdefault("judge_score", 1.0)
                            _e.setdefault("remarks", "auto-approved: no items or API key")
                    _djudged["judge_method"] = "auto_approved"
                    logger.info(
                        "[judge_task] Direct-API judge: no items/key, injected defaults into %d %s",
                        len(_dj_entities), _dj_items_key,
                    )

                # --- post-processing: same checks as the CrewAI chunked path ---

                # 1. Data-loss guard: if the LLM dropped items, recover from the
                #    best available prior stage rather than silently losing data.
                _dj_pre_count = len(_dj_entities)
                _dj_post_count = len(_djudged.get(_dj_items_key) or [])
                if _dj_post_count < _dj_pre_count:
                    logger.warning(
                        "[judge_task] Direct-API: item count dropped %d → %d; "
                        "recovering from best prior stage.",
                        _dj_pre_count, _dj_post_count,
                    )
                    _dj_fallback = None
                    for _fkey in list(pipeline_stages.keys())[::-1]:
                        _fs = pipeline_stages.get(_fkey)
                        if isinstance(_fs, dict) and len(_fs.get(_dj_items_key) or []) >= _dj_pre_count:
                            _dj_fallback = _fs
                            logger.info(
                                "[judge_task] Direct-API: recovered %d %s from stage '%s'",
                                len(_fs.get(_dj_items_key, [])), _dj_items_key, _fkey,
                            )
                            break
                    if _dj_fallback is not None:
                        from utils.downstream_agent_helper import _extend_previous_stage
                        _djudged = _extend_previous_stage(_dj_fallback, _djudged)
                        _djudged[_dj_items_key] = (
                            _dj_fallback.get(_dj_items_key) or _djudged.get(_dj_items_key) or []
                        )

                # 2. Ontology consistency pass: unify ontology IDs that may differ
                #    across parallel batches for the same entity text.
                _dj_final_items = _djudged.get(_dj_items_key)
                if _dj_final_items and _dj_items_key == "entities":
                    _djudged["entities"] = unify_ontology_across_entities(_dj_final_items)
                    logger.info(
                        "[judge_task] Direct-API: ontology consistency pass on %d entities",
                        len(_dj_final_items),
                    )

                # 3. Provenance tagging
                _dj_ckey = self._detect_container_key(_djudged)
                if _dj_ckey:
                    add_provenance_to_result(_djudged, _dj_ckey, agent_key)

                promote_stage_output_to_canonical(_djudged, task_type)
                prev_output = _djudged
                pipeline_stages[task_key] = _djudged
                stage_elapsed = time.time() - stage_start_time
                stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
                logger.info(
                    "[judge_task] Direct-API judge completed in %.2fs", stage_elapsed
                )

                if self.stage_output_dir and pipeline_stages.get(task_key) is not None:
                    try:
                        os.makedirs(self.stage_output_dir, exist_ok=True)
                        _dj_fname = f"{idx:02d}_{agent_key}_{task_key}.json"
                        _dj_path = os.path.join(self.stage_output_dir, _dj_fname)
                        with open(_dj_path, "w") as _djf:
                            json.dump(pipeline_stages[task_key], _djf, indent=2, default=str)
                        logger.info("[judge_task] Stage output saved to %s", _dj_path)
                    except Exception as _dj_save_exc:
                        logger.warning("[judge_task] Failed to save stage output: %s", _dj_save_exc)

                continue  # skip the CrewAI judge agent run entirely

            # Begin per-stage LLM call tracking (only for stages that run an LLM).
            # Bypassed / preloaded stages already `continue`d above so we never reach here for them.
            _begin_stage_llm_tracking(task_key, agent_key)
            _stage_n_chunks = 1  # updated to len(chunks) if the stage is split

            if is_first_stage:
                # First stage: source text, chunking, post-processing, merger
                stage_text = text
                stage_chunk_size = self.chunk_size if self.enable_chunking else None
                stage_post_process = post_processor
                stage_default_result = default_result
            else:
                # Downstream stages: prepare token-managed input for alignment, judge, or humanfeedback
                if prev_output is None:
                    logger.warning(f"Skipping {agent_key}/{task_key}: no previous output")
                    continue

                # Store previous agent result in context
                prev_agent_key, prev_task_key = ordered_pairs[idx - 1]
                self.agent_context.add_agent_result(
                    agent_key=prev_agent_key,
                    task_key=prev_task_key,
                    result=prev_output,
                    confidence=prev_output.get("confidence", 0.0) if isinstance(prev_output, dict) else 0.0,
                    metadata={"stage_index": idx - 1},
                )

                # Each agent always takes the previous agent's output (no exception).
                # Alignment <- extractor; Judge <- alignment; Human feedback <- judge + human input.
                if task_key == "alignment_task":
                    # Alignment always takes previous agent (extractor) output
                    extraction_output = pipeline_stages.get(ordered_pairs[0][1]) if idx >= 1 else prev_output
                    if extraction_output is None:
                        extraction_output = prev_output
                    if not isinstance(extraction_output, dict):
                        extraction_output = prev_output if isinstance(prev_output, dict) else {}

                    # -----------------------------------------------------------------------
                    # LAYER 1 OF 3 — Pre-compute concept mapping before the alignment LLM runs
                    # -----------------------------------------------------------------------
                    # WHY THIS EXISTS:
                    #   The alignment agent is an LLM whose system prompt instructs it to call
                    #   ConceptMappingLocalTool once with ALL entity texts.  In practice LLMs
                    #   (gpt-4o-mini, Gemini flash, etc.) ignore this and either:
                    #     - call the tool with only 5–10 terms and stop, or
                    #     - call it multiple times with small subsets.
                    #   Out of ~87 entities only 8 might get mapped — not because the tool
                    #   failed, but because the LLM simply chose not to call it for the rest.
                    #
                    # WHAT WE DO:
                    #   Before the alignment LLM runs, Python collects every entity text
                    #   programmatically and calls ConceptMappingLocalTool._run() directly
                    #   with the full deduplicated batch.  Results land in _ALIGNMENT_TOOL_OUTPUTS.
                    #   The alignment LLM then runs normally; any additional calls it makes also
                    #   append to _ALIGNMENT_TOOL_OUTPUTS.  After the stage finishes,
                    #   inject_alignment_concept_mapping_into_ner_entities reads the accumulated
                    #   outputs and stamps ontology_id / ontology_label / ontology onto every
                    #   entity dict whose text matches a captured result.
                    #
                    # WHY WE COLLECT FROM MULTIPLE KEYS (not just "entities"):
                    #   The extractor LLM sometimes writes BOTH:
                    #     entities: [2 items]          ← small partial list
                    #     extracted_terms: {"1": [...75 items...]}  ← the full output
                    #   promote_stage_output_to_canonical sees entities is non-empty and
                    #   skips promoting extracted_terms → the pre-compute previously only
                    #   found those 2 items.  We now walk all four possible locations:
                    #     1. entities          – canonical promoted list (may be partial)
                    #     2. extracted_terms   – raw stage key; dict-of-lists or list-of-dicts
                    #     3. aligned_ner_terms – present in re-run scenarios
                    #     4. key_terms         – string list or list-of-dicts
                    #   Log line shows coverage per source, e.g.:
                    #     [alignment_task] Pre-computing: 87 terms (entities=2, key_terms=10, raw_extracted=75)
                    _cm_backend = os.getenv("CONCEPT_MAPPING_BACKEND", "local").strip().lower()
                    if _cm_backend == "local" and task_type in (
                        "ner", "extraction", "keyphrase_extraction",
                        "resource", "structured_extraction",
                    ):
                        try:
                            from utils.conceptmappinglocal import ConceptMappingLocalTool as _CMTool
                            _pre_texts: list = []

                            def _collect_entity_text(ent):
                                if isinstance(ent, dict):
                                    t = ent.get("entity") or ent.get("text") or ent.get("name")
                                    if isinstance(t, str) and t.strip():
                                        _pre_texts.append(t.strip())

                            if task_type in ("ner", "extraction", "keyphrase_extraction"):
                                # 1. Canonical entities list
                                for _e in extraction_output.get("entities", []):
                                    _collect_entity_text(_e)

                                # 2 & 3. Raw NER stage keys (extracted_terms, aligned_ner_terms) —
                                #        these may contain more entities than the promoted list when
                                #        promote_stage_output_to_canonical found a non-empty entities
                                #        key and skipped the stage-specific keys.
                                for _raw_key in ("extracted_terms", "aligned_ner_terms"):
                                    _raw_container = extraction_output.get(_raw_key)
                                    if _raw_container:
                                        for _e in _flatten_container_to_list(_raw_container):
                                            _collect_entity_text(_e)

                                # 4. key_terms (strings or dicts)
                                for _kt in extraction_output.get("key_terms", []):
                                    if isinstance(_kt, str) and _kt.strip():
                                        _pre_texts.append(_kt.strip())
                                    elif isinstance(_kt, dict):
                                        _kt_t = _kt.get("term") or _kt.get("text") or _kt.get("entity")
                                        if isinstance(_kt_t, str) and _kt_t.strip():
                                            _pre_texts.append(_kt_t.strip())
                            else:
                                # Resource/structured_extraction: collect resource names
                                for _rkey in ("resources", "aligned_resources", "extracted_resources"):
                                    for _r in extraction_output.get(_rkey) or []:
                                        if isinstance(_r, dict):
                                            _rname = _r.get("name") or _r.get("resource_name")
                                            if isinstance(_rname, str) and _rname.strip():
                                                _pre_texts.append(_rname.strip())

                            # Deduplicate preserving order
                            _pre_seen: set = set()
                            _pre_unique = [_t for _t in _pre_texts if not (_t in _pre_seen or _pre_seen.add(_t))]
                            logger.info(
                                "[alignment_task] Pre-computing concept mapping: %d unique term(s) "
                                "(task_type=%s, entities=%d, resources=%d, key_terms=%d, raw_extracted=%d)",
                                len(_pre_unique),
                                task_type,
                                len(extraction_output.get("entities") or []),
                                len(extraction_output.get("resources") or []),
                                len(extraction_output.get("key_terms") or []),
                                len(_flatten_container_to_list(extraction_output.get("extracted_terms") or [])),
                            )
                            if _pre_unique:
                                print(
                                    f"[PRE-COMPUTE] Calling ConceptMappingLocalTool with {len(_pre_unique)} terms",
                                    flush=True,
                                )
                                _CMTool()._run(text=_pre_unique)
                            else:
                                print(
                                    "[PRE-COMPUTE] _pre_unique is empty — skipping concept mapping pre-compute",
                                    flush=True,
                                )
                        except Exception as _pre_exc:
                            import traceback
                            print(
                                f"[PRE-COMPUTE ERROR] Concept mapping pre-compute failed: {_pre_exc}\n"
                                f"{traceback.format_exc()}",
                                flush=True,
                            )
                            logger.warning("[alignment_task] Pre-compute concept mapping skipped: %s", _pre_exc)

                    logger.info(f"[{agent_key}] Preparing token-managed input for alignment agent")
                    managed_input = prepare_alignment_agent_input(
                        extraction_results=extraction_output,
                        original_text=text,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                    )
                    extra_inputs = managed_input
                    stage_text = None
                elif task_key == "judge_task":
                    # Judge always takes previous agent (alignment) output
                    alignment_output = pipeline_stages.get("alignment_task") if idx >= 2 else prev_output
                    if alignment_output is None:
                        alignment_output = prev_output
                    if not isinstance(alignment_output, dict):
                        alignment_output = prev_output if isinstance(prev_output, dict) else {}
                    logger.info(f"[{agent_key}] Preparing token-managed input for judge agent")
                    extraction_results = None
                    if idx >= 2:
                        extraction_agent_key, _ = ordered_pairs[0]
                        extraction_result = self.agent_context.get_latest_result(extraction_agent_key)
                        if extraction_result:
                            extraction_results = extraction_result.result
                    managed_input = prepare_judge_agent_input(
                        alignment_results=alignment_output,
                        extraction_results=extraction_results,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                    )
                    extra_inputs = managed_input
                    stage_text = None
                else:
                    # For other downstream stages, pass the full previous output as JSON string
                    stage_text = json.dumps(prev_output, indent=2) if isinstance(prev_output, dict) else str(prev_output)

                stage_chunk_size = None
                stage_post_process = None
                # Reuse the task_type detected at the extraction stage (cached in self._pipeline_task_type)
                stage_default_result = self._get_default_result_for_task(task_type)

            # Human feedback receives judge output: prev_output at this point is the judge stage result
            if task_key == "humanfeedback_task" and prev_output is not None:
                # Early NER fallback: if the judge returned the wrong schema (e.g. resource keys
                # instead of judge_ner_terms), promote_stage_output_to_canonical leaves
                # entities=[] in prev_output.  The post-loop fallback at line ~1451 fixes this
                # for the final result, but human feedback and prepare_humanfeedback_agent_input
                # both run INSIDE the loop and would receive empty data.
                # Resolve the best available stage now so the human sees real entities and the
                # humanfeedback agent gets the correct judge_output below.
                if task_type == "ner" and isinstance(prev_output, dict) and not prev_output.get("entities"):
                    for _fk in ("judge_task", "alignment_task", "extraction_task"):
                        _fs = pipeline_stages.get(_fk)
                        if isinstance(_fs, dict) and _fs.get("entities"):
                            prev_output = _fs
                            # Keep pipeline_stages["judge_task"] consistent so the
                            # judge_output = pipeline_stages.get("judge_task") line below
                            # also receives the recovered data.
                            pipeline_stages["judge_task"] = prev_output
                            logger.info(
                                "[humanfeedback_task] prev_output had empty entities after judge stage "
                                "(wrong-schema / default-fallback case); recovered %d entities from '%s'.",
                                len(_fs["entities"]), _fk,
                            )
                            break

                # Collect user feedback (1=Approve, 2=View, 3=Modify, 4=Abort)
                feedback_text = user_feedback_text
                if feedback_text is None and self.human_loop.is_feedback_enabled_for_agent("humanfeedback_agent"):
                    feedback_result = self.human_loop.request_feedback(
                        prev_output,
                        step_name="human_feedback_processing",
                        agent_name="humanfeedback_agent",
                    )
                    if isinstance(feedback_result, dict):
                        if feedback_result.get("_human_abort_feedback"):
                            # Option 4: skip humanfeedback agent, save judge output as final (not full pipeline abort)
                            pipeline_stages[task_key] = prev_output
                            humanfeedback_approved_skip_run = True
                            humanfeedback_aborted_by_user = True
                            logger.info("[humanfeedback_agent] Skipping LLM run (user aborted feedback step). Judge result will be saved.")
                        elif feedback_result.get("_human_approved_skip_agent"):
                            # Option 1: keep judge output as final; do not overwrite prev_output (it already holds judge output)
                            pipeline_stages[task_key] = prev_output
                            humanfeedback_approved_skip_run = True
                            human_approved_skip_concept_mapping = True
                            logger.info("[humanfeedback_agent] Skipping LLM run (user approved judge output).")
                        else:
                            feedback_text = feedback_result.get("user_feedback_text", "")
                            if feedback_result.get("user_feedback_json") is not None:
                                mod_context = json.dumps(feedback_result["user_feedback_json"], indent=2)
                                feedback_text = f"{feedback_text}\n\nModification Context:\n{mod_context}"

                if not humanfeedback_approved_skip_run:
                    if not feedback_text:
                        feedback_text = modification_context or "No specific feedback provided."

                    # Human feedback always takes previous agent (judge) output + human input
                    judge_output = pipeline_stages.get("judge_task") if len(ordered_pairs) >= 3 else prev_output
                    if judge_output is None:
                        judge_output = prev_output
                    if not isinstance(judge_output, dict):
                        judge_output = prev_output if isinstance(prev_output, dict) else {}
                    alignment_for_human = pipeline_stages.get("alignment_task")  # for helper's merge when needed
                    # Extraction output — gives humanfeedback agent access to entities that
                    # survived extraction but were dropped during alignment/judge.
                    _hf_extraction_out = pipeline_stages.get("extraction_task")
                    logger.info(f"[{agent_key}] Preparing token-managed input for humanfeedback agent")
                    managed_input = prepare_humanfeedback_agent_input(
                        judge_results=judge_output,
                        user_feedback=feedback_text,
                        alignment_results=alignment_for_human,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                        original_text=text,
                        extraction_results=_hf_extraction_out,
                    )
                    extra_inputs = managed_input
                    stage_text = None
            elif task_key != "humanfeedback_task" and not is_first_stage and extra_inputs is None:
                # For other downstream stages, extra_inputs was set above
                pass

            # ------------------------------------------------------------------
            # DIRECT-API HUMANFEEDBACK BYPASS
            # ------------------------------------------------------------------
            # Replace the CrewAI humanfeedback agent with direct AsyncOpenAI
            # calls — same pattern as direct_judge_api.
            #
            # Only fires when:
            #   - task_key == "humanfeedback_task"
            #   - humanfeedback_approved_skip_run is False (user did not already
            #     approve/abort — those paths are handled above and set continue)
            #   - extra_inputs has been prepared (feedback_text + judge output)
            #   - self.direct_humanfeedback_api is True (default)
            #
            # Enable (default): direct_humanfeedback_api=True / DIRECT_HUMANFEEDBACK_API=true
            # Disable (fallback to CrewAI): direct_humanfeedback_api=False / DIRECT_HUMANFEEDBACK_API=false
            # ------------------------------------------------------------------
            if (
                task_key == "humanfeedback_task"
                and not humanfeedback_approved_skip_run
                and extra_inputs is not None
                and prev_output is not None
                and self.direct_humanfeedback_api
            ):
                import copy as _hfcopy
                stage_start_time = time.time()
                logger.info(
                    "[humanfeedback_task] Direct-API humanfeedback active — bypassing CrewAI agent"
                )

                # Resolve LLM config for the humanfeedback agent from its agent_config entry
                _hf_llm = self.agent_config.get(agent_key, {}).get("llm", {})
                if isinstance(_hf_llm, dict) and _hf_llm.get("model"):
                    _hf_model = _hf_llm["model"]
                    _hf_base_url = _hf_llm.get("base_url") or "https://openrouter.ai/api/v1"
                    logger.info("[humanfeedback_task] Direct-API: model=%s (from agent_config[%s])", _hf_model, agent_key)
                else:
                    _hf_model = "openai/gpt-4o-mini"
                    _hf_base_url = "https://openrouter.ai/api/v1"
                    logger.warning(
                        "[humanfeedback_task] Direct-API: no LLM config found for agent_key=%s — using fallback model=%s",
                        agent_key, _hf_model,
                    )
                if "openrouter" in _hf_base_url.lower() and _hf_model.startswith("openrouter/"):
                    _hf_model = _hf_model.replace("openrouter/", "", 1)

                _hf_api_key = (
                    os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or ""
                )

                # Extract the full judged payload and feedback text from extra_inputs
                _hf_payload_key = "judged_structured_information_with_human_feedback"
                _hf_judged = _hfcopy.deepcopy(
                    extra_inputs.get(_hf_payload_key) or prev_output
                )
                _hf_feedback = (
                    extra_inputs.get("user_feedback_text")
                    or extra_inputs.get("modification_context")
                    or "No specific feedback provided."
                )
                # Source text and extraction output — passed so the LLM can find entities
                # missed by earlier stages when the user reports a low entity count.
                _hf_source_text = extra_inputs.get("source_text") or ""
                _hf_extraction_out = extra_inputs.get("extraction_results")

                # Determine primary work-item list and key
                _hf_items_key = (
                    "entities" if _hf_judged.get("entities")
                    else "resources" if _hf_judged.get("resources")
                    else "entities"
                )
                _hf_items = _hf_judged.get(_hf_items_key) or []

                # Token-aware batch sizing — same logic as judge and CrewAI chunked path
                _hf_workers = self.max_workers or 4
                _hf_extraction_chunk_count = None
                for _pk in pipeline_stages:
                    _ps = pipeline_stages.get(_pk)
                    if isinstance(_ps, dict) and "_extraction_chunk_count" in _ps:
                        _hf_extraction_chunk_count = _ps["_extraction_chunk_count"]
                        break
                _hf_task_cfg = (
                    self.task_config.get(task_key, {})
                    if isinstance(self.task_config, dict)
                    else {}
                )
                _hf_prompt_overhead = estimate_agent_prompt_tokens(
                    agent_config=self.agent_config.get(agent_key, {}),
                    task_config=_hf_task_cfg,
                )
                _hf_batch, _hf_should_batch = compute_downstream_chunk_size(
                    payload=_hf_judged,
                    model_str=_hf_model,
                    max_workers=_hf_workers,
                    extraction_chunk_count=_hf_extraction_chunk_count,
                    explicit_chunk_size=self.downstream_chunk_size,
                    context_window_override=self.model_context_window,
                    prompt_overhead_tokens=_hf_prompt_overhead,
                )
                if not _hf_should_batch:
                    _hf_batch = len(_hf_items) or 1

                # ── Direct-API output-size guard (same logic as judge) ───────
                _hf_out_cap = int(os.environ.get("DIRECT_API_MAX_OUTPUT_TOKENS", "32768"))
                _hf_min_chunks = int(os.environ.get("DIRECT_API_MIN_CHUNKS", "8"))
                _hf_n_items = len(_hf_items) or 1
                _hf_tok_per_item = max(1, len(json.dumps(_hf_judged, ensure_ascii=False)) / _hf_n_items)
                _hf_out_capped = max(1, int(_hf_out_cap / _hf_tok_per_item))
                _hf_min_chunk_batch = max(1, math.ceil(_hf_n_items / _hf_min_chunks))
                _hf_effective = min(_hf_batch, _hf_out_capped, _hf_min_chunk_batch)
                if _hf_effective < _hf_batch:
                    logger.info(
                        "[humanfeedback_task] Direct-API output guard: output_cap=%d tok "
                        "(%.1f tok/item → max %d items), min_chunks=%d (→ max %d items) "
                        "→ batch %d → %d items",
                        _hf_out_cap, _hf_tok_per_item, _hf_out_capped,
                        _hf_min_chunks, _hf_min_chunk_batch,
                        _hf_batch, _hf_effective,
                    )
                _hf_batch = _hf_effective

                logger.info(
                    "[humanfeedback_task] Direct-API batch size: %d items/batch "
                    "(should_batch=%s, model=%s)",
                    _hf_batch, _hf_should_batch, _hf_model,
                )

                async def _humanfeedback_items_direct(
                    _ents, _model, _base_url, _key, _batch_idx, _items_key, _feedback,
                    _source_text="", _extraction_out=None, _attempt=0
                ):
                    """Apply human feedback to a batch of items via a direct LLM call.

                    Also receives the original source text and raw extraction output so
                    the LLM can find entities missed by earlier stages when the user
                    reports a low entity count.

                    Retry policy:
                    - JSONDecodeError (truncated output): split batch in half and recurse
                      so each half is a smaller independent call.  Recurses down to a
                      batch of 1 before giving up.
                    - Other API / network errors: retry up to 3 times with exponential
                      back-off (1 s, 2 s, 4 s).
                    """
                    _MAX_RETRIES = 3
                    from openai import AsyncOpenAI as _AsyncOpenAI
                    _c = _AsyncOpenAI(base_url=_base_url, api_key=_key)
                    # Build task-aware system prompt so any natural-language feedback works
                    # for NER, resource extraction, generic extraction, or any other task type.
                    _hf_task = (task_type or "extraction").strip().lower()
                    if _hf_task == "ner":
                        _task_desc = "named entity recognition (NER)"
                        _item_desc = "entities — fix wrong labels, update ontology mappings, revise judge remarks, add missing entities"
                    elif _hf_task in ("resource", "structured_extraction"):
                        _task_desc = "resource / structured extraction"
                        _item_desc = "resources — fix wrong types, categories, ontology mappings, add missing resources"
                    elif _hf_task == "keyphrase_extraction":
                        _task_desc = "keyphrase extraction"
                        _item_desc = "key terms — correct labels, ontology mappings, or add missing terms"
                    else:
                        _task_desc = _hf_task.replace("_", " ")
                        _item_desc = f"{_items_key} items — apply any corrections or additions requested"
                    _sys = (
                        f"You are a human feedback integration agent for a {_task_desc} pipeline. "
                        f"Apply the human feedback below to every item in the \"{_items_key}\" list. "
                        f"Specifically: {_item_desc}. "
                        "If the feedback indicates items are missing, search the source text and "
                        "raw extraction output (provided below) to find and add them. "
                        "Preserve ALL existing fields that are not being corrected. "
                        f"Return ONLY valid JSON: {{\"{_items_key}\": [...]}}. No markdown, no prose."
                    )
                    # Build user message: feedback + optional source context + current items
                    _user_parts = [f"Human feedback:\n{_feedback}"]
                    if _source_text:
                        _user_parts.append(f"Source text (for finding missing entities):\n{_source_text[:20_000]}")
                    if _extraction_out and isinstance(_extraction_out, dict):
                        _ext_items = _extraction_out.get(_items_key) or []
                        if _ext_items:
                            _user_parts.append(
                                f"Raw extraction output (may contain entities dropped by later stages):\n"
                                f"{json.dumps({_items_key: _ext_items[:100]}, ensure_ascii=False)}"
                            )
                    _user_parts.append(f"Current data to revise:\n{json.dumps({_items_key: _ents}, ensure_ascii=False)}")
                    _user_msg = "\n\n".join(_user_parts)

                    # Log into llm_calls log (same counter as CrewAI calls)
                    with _llm_call_lock:
                        _llm_call_count["n"] += 1
                        _call_n = _llm_call_count["n"]
                    _hf_llm_log = logging.getLogger("llm_calls")
                    if _hf_llm_log.handlers:
                        _hf_llm_log.info("=" * 70)
                        _hf_llm_log.info(f"[LLM CALL #{_call_n}]  (direct API — no CrewAI agent)")
                        _hf_llm_log.info(f"Agent          : {agent_key} / humanfeedback_task (direct)")
                        _hf_llm_log.info(f"Model          : {_hf_model}")
                        _hf_llm_log.info(f"Batch          : {_batch_idx + 1}  ({len(_ents)} items, attempt {_attempt + 1})")
                        _hf_llm_log.info(f"Feedback       : {_feedback[:200]}")
                        _hf_llm_log.info(f"Payload preview: {json.dumps({_items_key: _ents}, ensure_ascii=False)[:300]}")
                        _hf_llm_log.info("=" * 70)
                        _flush_llm_logger()

                    try:
                        _r = await _c.chat.completions.create(
                            model=_model,
                            messages=[
                                {"role": "system", "content": _sys},
                                {"role": "user", "content": _user_msg},
                            ],
                        )
                        _raw = (_r.choices[0].message.content or "").strip()

                        if _hf_llm_log.handlers:
                            _hf_llm_log.info(f"[LLM RESPONSE PREVIEW]  (call #{_call_n})")
                            _hf_llm_log.info(_raw[:500])
                            _hf_llm_log.info("=" * 70)
                            _flush_llm_logger()

                        if _raw.startswith("```"):
                            _parts = _raw.split("```")
                            _raw = _parts[1] if len(_parts) > 1 else _raw
                            if _raw.startswith("json"):
                                _raw = _raw[4:]
                        _stripped = _raw.strip()
                        try:
                            _parsed = json.loads(_stripped)
                        except json.JSONDecodeError as _parse_ex:
                            if "Extra data" in str(_parse_ex):
                                # LLM appended trailing content after valid JSON — extract first object only
                                _parsed, _ = json.JSONDecoder().raw_decode(_stripped)
                                logger.debug(
                                    "[humanfeedback_task] Direct API batch %d: recovered from 'Extra data' via raw_decode",
                                    _batch_idx + 1,
                                )
                            else:
                                raise  # truncated JSON — handled by outer except below
                        return _parsed.get(_items_key) or _ents
                    except json.JSONDecodeError as _ex:
                        # Truncated JSON (Unterminated string, Expecting value, etc.) — split batch in half
                        logger.warning(
                            "[humanfeedback_task] Direct API batch %d JSON parse error (attempt %d/%d, %d items): %s — splitting batch",
                            _batch_idx + 1, _attempt + 1, _MAX_RETRIES, len(_ents), _ex,
                        )
                        if len(_ents) > 1:
                            _mid = len(_ents) // 2
                            _l_res, _r_res = await asyncio.gather(
                                _humanfeedback_items_direct(_ents[:_mid], _model, _base_url, _key, _batch_idx, _items_key, _feedback, _source_text, _extraction_out, 0),
                                _humanfeedback_items_direct(_ents[_mid:], _model, _base_url, _key, _batch_idx, _items_key, _feedback, _source_text, _extraction_out, 0),
                            )
                            return _l_res + _r_res
                        logger.warning("[humanfeedback_task] Direct API single-item batch still failed — returning original item")
                        return _ents
                    except Exception as _ex:
                        if _attempt + 1 < _MAX_RETRIES:
                            _wait = 2 ** _attempt
                            logger.warning(
                                "[humanfeedback_task] Direct API batch %d error (attempt %d/%d, retry in %ds): %s",
                                _batch_idx + 1, _attempt + 1, _MAX_RETRIES, _wait, _ex,
                            )
                            await asyncio.sleep(_wait)
                            return await _humanfeedback_items_direct(_ents, _model, _base_url, _key, _batch_idx, _items_key, _feedback, _source_text, _extraction_out, _attempt + 1)
                        logger.warning("[humanfeedback_task] Direct API batch %d failed after %d attempts: %s", _batch_idx + 1, _MAX_RETRIES, _ex)
                        return _ents

                if _hf_items and _hf_api_key:
                    _begin_stage_llm_tracking(task_key, agent_key)
                    try:
                        _hf_batches = [
                            _hf_items[_hf_i: _hf_i + _hf_batch]
                            for _hf_i in range(0, len(_hf_items), _hf_batch)
                        ]
                        _hf_n_batches = len(_hf_batches)
                        logger.info(
                            "[humanfeedback_task] Direct-API: %d items → %d batches × %d, running in parallel",
                            len(_hf_items), _hf_n_batches, _hf_batch,
                        )
                        _hf_batch_results = await asyncio.gather(*[
                            _humanfeedback_items_direct(
                                _hf_batches[_bi], _hf_model, _hf_base_url, _hf_api_key,
                                _bi, _hf_items_key, _hf_feedback,
                                _hf_source_text, _hf_extraction_out,
                            )
                            for _bi in range(_hf_n_batches)
                        ])
                        _hf_revised: list = []
                        for _hf_result in _hf_batch_results:
                            _hf_revised.extend(_hf_result)
                        _hf_judged[_hf_items_key] = _hf_revised
                        _hf_judged["humanfeedback_method"] = "direct_api"
                        _end_stage_llm_tracking(task_key, _hf_n_batches, time.time() - stage_start_time)
                        logger.info(
                            "[humanfeedback_task] Direct-API revised %d %s in %.2fs",
                            len(_hf_revised), _hf_items_key, time.time() - stage_start_time,
                        )
                    except Exception as _hf_exc:
                        logger.warning(
                            "[humanfeedback_task] Direct-API failed (%s); keeping judge output unchanged", _hf_exc
                        )
                        _hf_judged["humanfeedback_method"] = "direct_api_fallback"
                else:
                    # No API key or no items: pass judge output through unchanged
                    _hf_judged["humanfeedback_method"] = "direct_api_passthrough"
                    logger.info(
                        "[humanfeedback_task] Direct-API: no items/key — passing judge output through (%d %s)",
                        len(_hf_items), _hf_items_key,
                    )

                # --- post-processing: same checks as the CrewAI chunked path ---

                # 1. Data-loss guard
                _hf_pre_count = len(_hf_items)
                _hf_post_count = len(_hf_judged.get(_hf_items_key) or [])
                if _hf_post_count < _hf_pre_count:
                    logger.warning(
                        "[humanfeedback_task] Direct-API: item count dropped %d → %d; "
                        "recovering from best prior stage.",
                        _hf_pre_count, _hf_post_count,
                    )
                    _hf_fallback = None
                    for _fkey in list(pipeline_stages.keys())[::-1]:
                        _fs = pipeline_stages.get(_fkey)
                        if isinstance(_fs, dict) and len(_fs.get(_hf_items_key) or []) >= _hf_pre_count:
                            _hf_fallback = _fs
                            logger.info(
                                "[humanfeedback_task] Direct-API: recovered %d %s from stage '%s'",
                                len(_fs.get(_hf_items_key, [])), _hf_items_key, _fkey,
                            )
                            break
                    if _hf_fallback is not None:
                        from utils.downstream_agent_helper import _extend_previous_stage
                        _hf_judged = _extend_previous_stage(_hf_fallback, _hf_judged)
                        _hf_judged[_hf_items_key] = (
                            _hf_fallback.get(_hf_items_key) or _hf_judged.get(_hf_items_key) or []
                        )

                # 2. Ontology consistency pass
                _hf_final_items = _hf_judged.get(_hf_items_key)
                if _hf_final_items and _hf_items_key == "entities":
                    _hf_judged["entities"] = unify_ontology_across_entities(_hf_final_items)
                    logger.info(
                        "[humanfeedback_task] Direct-API: ontology consistency pass on %d entities",
                        len(_hf_final_items),
                    )

                # 3. Provenance tagging
                _hf_ckey = self._detect_container_key(_hf_judged)
                if _hf_ckey:
                    add_provenance_to_result(_hf_judged, _hf_ckey, agent_key)

                promote_stage_output_to_canonical(_hf_judged, task_type)
                prev_output = _hf_judged
                pipeline_stages[task_key] = _hf_judged
                stage_elapsed = time.time() - stage_start_time
                stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
                logger.info(
                    "[humanfeedback_task] Direct-API completed in %.2fs", stage_elapsed
                )

                if self.stage_output_dir and pipeline_stages.get(task_key) is not None:
                    try:
                        os.makedirs(self.stage_output_dir, exist_ok=True)
                        _hf_fname = f"{idx:02d}_{agent_key}_{task_key}.json"
                        _hf_path = os.path.join(self.stage_output_dir, _hf_fname)
                        with open(_hf_path, "w") as _hff:
                            json.dump(pipeline_stages[task_key], _hff, indent=2, default=str)
                        logger.info("[humanfeedback_task] Stage output saved to %s", _hf_path)
                    except Exception as _hf_save_exc:
                        logger.warning("[humanfeedback_task] Failed to save stage output: %s", _hf_save_exc)

                continue  # skip the CrewAI humanfeedback agent run entirely

            # Downstream stages: chunk merged result → process chunks in parallel → merge (avoids context-length errors)
            # Only when stage_text is used (not token-managed alignment/judge/humanfeedback) and over char limit
            if (
                not is_first_stage
                and stage_text is not None
                and isinstance(prev_output, dict)
                and len(stage_text) > self.downstream_max_input_chars
            ):
                chunks = self._split_downstream_payload(prev_output, self.downstream_max_input_chars)
                logger.info(
                    f"Downstream {task_key}: splitting into {len(chunks)} chunks (input {len(stage_text)} chars > {self.downstream_max_input_chars})"
                )

                # Process each chunk in parallel, then merge into one clean result for the next stage
                async def run_one_chunk(chunk_payload: Dict[str, Any]) -> Dict[str, Any]:
                    chunk_text = json.dumps(chunk_payload, indent=2)
                    return await self.run_agent_task(
                        agent_key=agent_key,
                        task_key=task_key,
                        text=chunk_text,
                        pydantic_output_class=None,
                        chunk_size=None,
                        max_workers=1,
                        post_process=None,
                        input_key=input_key,
                        default_result=stage_default_result,
                        extra_inputs=extra_inputs,
                    )

                chunk_results = await asyncio.gather(*[run_one_chunk(c) for c in chunks])
                downstream_results = []
                for result in chunk_results:
                    all_errors.extend(result.get("errors", []))
                    if result.get("results"):
                        downstream_results.append(result["results"][0])
                if downstream_results:
                    # FIX (empty-entities root cause): always detect the container key from
                    # the actual result dict instead of using the hardcoded TASK_KEY_TO_CONTAINER_KEY
                    # lookup.  The old code did:
                    #   TASK_KEY_TO_CONTAINER_KEY.get(task_key) or self._detect_container_key(...)
                    # For NER tasks TASK_KEY_TO_CONTAINER_KEY["judge_task"] == "judge_resource" which
                    # is truthy, so _detect_container_key was never called.  The merge was then
                    # executed with the wrong key ("judge_resource") on an NER result that contains
                    # "judge_ner_terms", producing an empty list every time.
                    # _detect_container_key inspects the actual keys present in the result, so it
                    # correctly returns "judge_ner_terms" for NER and "judge_resource" for resources.
                    container_key = self._detect_container_key(downstream_results[0])
                    if container_key:
                        prev_output = merge_downstream_chunk_results_with_provenance(downstream_results, container_key, agent_key)
                    else:
                        prev_output = self._merge_downstream_chunk_results(downstream_results)
                pipeline_stages[task_key] = prev_output
            elif humanfeedback_approved_skip_run:
                # Option 1 (Approve): prev_output and pipeline_stages already set; no agent run
                pass
            else:
                # Parallel chunking for alignment/judge/humanfeedback.
                #
                # Token-aware sizing (compute_downstream_chunk_size) decides
                # whether to split and how many chunks to use:
                #   - payload fits in model's usable context → 1 call, no split
                #   - payload too large → minimum chunks needed, ≤ max_workers
                #
                # entities_per_chunk = downstream_chunk_size  (explicit override)
                #                    = token-aware auto  (default)
                #                    = 70  (fallback when payload is empty/tiny)
                use_chunked = not is_first_stage and extra_inputs and task_key in ("alignment_task", "judge_task", "humanfeedback_task")
                payload_key = (
                    "extracted_structured_information"
                    if task_key == "alignment_task"
                    else "aligned_structured_information"
                    if task_key == "judge_task"
                    else "judged_structured_information_with_human_feedback"
                )
                payload = extra_inputs.get(payload_key) if use_chunked else None

                # Determine entities-per-chunk using token-aware sizing.
                #
                # Priority order:
                #   1. downstream_chunk_size (explicit override)
                #   2. Token-aware auto-calculation via compute_downstream_chunk_size:
                #      - payload fits within model's usable context → single call, no split
                #      - otherwise → minimum chunks needed, capped at max_workers
                #      - usable context = (model_context_window − 10k) × 0.70
                #
                # The model string is read from the current agent's llm config so that
                # context-window limits are per-model accurate (e.g. Gemini 1M vs DeepSeek 128k).
                _workers = self.max_workers or 4
                _n_items = 0
                if isinstance(payload, dict):
                    _n_items = len(payload.get("entities") or payload.get("resources") or [])

                # Look up how many extraction chunks ran (stored by extractor stage above)
                _extraction_chunk_count = None
                for _pk in pipeline_stages:
                    _ps = pipeline_stages.get(_pk)
                    if isinstance(_ps, dict) and "_extraction_chunk_count" in _ps:
                        _extraction_chunk_count = _ps["_extraction_chunk_count"]
                        break

                # Resolve model string for the current downstream agent
                _agent_llm_cfg = self.agent_config.get(agent_key, {}).get("llm", {})
                _model_str = (
                    _agent_llm_cfg.get("model", "")
                    if isinstance(_agent_llm_cfg, dict)
                    else str(_agent_llm_cfg)
                )

                # Probe OpenRouter for the real context window if the model is
                # served through OpenRouter and we have an API key.  The result
                # is cached in _openrouter_context_cache so the probe runs at
                # most once per model per process lifetime.
                if not self.model_context_window and _model_str:
                    _or_api_key = os.environ.get("OPENROUTER_API_KEY", "")
                    _or_base_url = (
                        (_agent_llm_cfg.get("base_url") or "")
                        if isinstance(_agent_llm_cfg, dict)
                        else ""
                    )
                    if _or_api_key and (
                        "openrouter" in (_model_str or "").lower()
                        or "openrouter" in (_or_base_url or "").lower()
                    ):
                        _probe_base = _or_base_url or "https://openrouter.ai/api/v1"
                        probe_openrouter_context_window(
                            model_str=_model_str,
                            api_key=_or_api_key,
                            base_url=_probe_base,
                        )

                # Adaptively estimate the agent's prompt overhead from the
                # *actual* config content (role + goal + backstory + task
                # description template + CrewAI framework boilerplate).
                # This avoids the fixed 10 k fallback which was 100× too small
                # for detailed configs with many entity-type examples.
                _task_cfg = (
                    self.task_config.get(task_key, {})
                    if isinstance(self.task_config, dict)
                    else {}
                )
                _prompt_overhead = estimate_agent_prompt_tokens(
                    agent_config=self.agent_config.get(agent_key, {}),
                    task_config=_task_cfg,
                )
                logger.debug(
                    "[chunk_size] Adaptive prompt overhead for '%s/%s': %d tokens",
                    agent_key, task_key, _prompt_overhead,
                )

                _ecs, _token_should_chunk = compute_downstream_chunk_size(
                    payload=payload if isinstance(payload, dict) else {},
                    model_str=_model_str,
                    max_workers=_workers,
                    extraction_chunk_count=_extraction_chunk_count,
                    explicit_chunk_size=self.downstream_chunk_size,
                    context_window_override=self.model_context_window,
                    prompt_overhead_tokens=_prompt_overhead,
                )

                # Downstream chunking is AUTOMATIC and independent of --enable_chunking.
                #
                # --enable_chunking controls text extraction (splitting a long PDF into
                # parallel extraction chunks).  Downstream agent chunking (alignment,
                # judge, humanfeedback) is purely token-driven: if the payload is too
                # large for the model's context window it MUST be split regardless of
                # whether the user passed --enable_chunking.
                #
                # Without this separation, running without --enable_chunking on a large
                # extraction result sent the full 1.6 M-token payload in a single call
                # to a 1 M-token model, triggering repeated 400 context-overflow errors
                # (each retried by LiteLLM, producing 13+ failed calls per stage).
                #
                # Rule:
                #   should_chunk = True  iff  token-aware sizing says the payload
                #                             exceeds the model's usable budget.
                #   enable_chunking      only gates whether the *extraction* stage
                #                        itself is split; it does NOT gate downstream.
                should_chunk = (
                    use_chunked
                    and isinstance(payload, dict)
                    and _token_should_chunk
                )

                if use_chunked and should_chunk:
                    # Record entity count going IN so we can verify nothing is lost after merge.
                    _pre_split_entities = len(payload.get("entities") or []) if isinstance(payload, dict) else 0
                    _pre_split_resources = len(payload.get("resources") or []) if isinstance(payload, dict) else 0
                    _chunk_size_source = (
                        "explicit downstream_chunk_size" if self.downstream_chunk_size
                        else f"token-aware/extraction_chunks={_extraction_chunk_count}" if _extraction_chunk_count
                        else f"token-aware/max_workers={_workers}"
                    )
                    logger.info(
                        "[%s] Splitting payload for parallel processing: %d entities, %d resources "
                        "→ ~%d per chunk (model=%s, source=%s)",
                        task_key, _pre_split_entities, _pre_split_resources, _ecs,
                        _model_str or "unknown", _chunk_size_source,
                    )

                    chunks = split_structured_payload(
                        payload,
                        max_entities_per_chunk=_ecs,
                        max_key_terms_per_chunk=max(10, _ecs // 3),
                        max_resources_per_chunk=max(5, _ecs // 10),
                    )
                    _stage_n_chunks = len(chunks)

                    async def run_one_structured_chunk(chunk_payload: Dict[str, Any]) -> Dict[str, Any]:
                        chunk_inputs = {**extra_inputs, payload_key: chunk_payload}
                        return await self.run_agent_task(
                            agent_key=agent_key,
                            task_key=task_key,
                            text=None,
                            pydantic_output_class=None,
                            chunk_size=None,
                            max_workers=1,
                            post_process=None,
                            input_key=PIPELINE_INPUT_KEY_MAP.get(task_key, "input_text"),
                            default_result=stage_default_result,
                            extra_inputs=chunk_inputs,
                        )

                    if len(chunks) > 1:
                        logger.info(
                            "[%s] Running %d chunks in parallel",
                            task_key, len(chunks),
                        )
                        chunk_results = await asyncio.gather(*[run_one_structured_chunk(c) for c in chunks])
                    else:
                        chunk_results = [await run_one_structured_chunk(chunks[0])]

                    for r in chunk_results:
                        all_errors.extend(r.get("errors", []))
                    raw_list = [r["results"][0] for r in chunk_results if r.get("results") and isinstance(r["results"][0], dict)]
                    if raw_list:
                        # FIX (empty-entities root cause): detect container key from the actual
                        # result rather than the hardcoded TASK_KEY_TO_CONTAINER_KEY map.
                        # TASK_KEY_TO_CONTAINER_KEY["judge_task"] == "judge_resource" caused NER
                        # chunks to be merged under the wrong key, yielding empty entities.
                        # See the same fix comment at the extraction-stage call site above.
                        ckey = self._detect_container_key(raw_list[0])
                        # For alignment/judge with multiple chunks: use resource-aware merge so tool-backed concepts (real IDs) beat N/A
                        if ckey and task_key in ("alignment_task", "judge_task") and len(raw_list) > 1:
                            prev_output = merge_downstream_chunk_results_with_provenance(raw_list, ckey, agent_key)
                        else:
                            prev_output = merge_structured_chunk_results(raw_list)
                            if isinstance(prev_output, dict) and ckey:
                                add_provenance_to_result(prev_output, ckey, agent_key)

                        # Data-loss guard: verify entity/resource count after merge matches what
                        # was sent in.  Each entity goes to exactly one chunk (slice distribution),
                        # so the merged count must equal the pre-split count.  A lower count means
                        # the LLM dropped entities; fall back to the best previous stage so no
                        # data is silently lost.
                        if isinstance(prev_output, dict):
                            _post_merge_entities = len(prev_output.get("entities") or [])
                            _post_merge_resources = len(prev_output.get("resources") or [])
                            if _post_merge_entities < _pre_split_entities:
                                logger.warning(
                                    "[%s] Entity count dropped after parallel merge: %d → %d. "
                                    "LLM may have omitted entities from some chunks. "
                                    "Falling back to best available prior stage to prevent data loss.",
                                    task_key, _pre_split_entities, _post_merge_entities,
                                )
                                # Recover from best available stage (extraction → alignment for judge, etc.)
                                _fallback = None
                                for _fkey in list(pipeline_stages.keys())[::-1]:
                                    _fs = pipeline_stages.get(_fkey)
                                    if isinstance(_fs, dict) and len(_fs.get("entities") or []) >= _pre_split_entities:
                                        _fallback = _fs
                                        logger.info("[%s] Recovered %d entities from stage '%s'", task_key, len(_fs.get("entities", [])), _fkey)
                                        break
                                if _fallback is not None:
                                    # Merge: keep richer ontology/judge fields from current output,
                                    # but restore the full entity list from fallback
                                    from utils.downstream_agent_helper import _extend_previous_stage
                                    prev_output = _extend_previous_stage(_fallback, prev_output)
                                    prev_output["entities"] = _fallback.get("entities") or prev_output.get("entities") or []

                        # Ontology consistency pass: parallel LLM calls may assign different
                        # ontology IDs to the same entity text in different chunks.  Unify by
                        # applying the best mapping (tool-backed > llm_knowledge, real IRI > N/A)
                        # to every occurrence across all merged chunks.
                        if isinstance(prev_output, dict):
                            ents = prev_output.get("entities")
                            if ents:
                                prev_output["entities"] = unify_ontology_across_entities(ents)
                                logger.info(
                                    "[%s] Ontology consistency pass: %d entities unified across %d chunks",
                                    task_key, len(ents), len(raw_list),
                                )

                        pipeline_stages[task_key] = prev_output
                    else:
                        result = chunk_results[0] if chunk_results else {}
                        prev_output = result.get("results", [{}])[0] if result.get("results") else prev_output
                        pipeline_stages[task_key] = prev_output
                else:
                    if task_key == "humanfeedback_task":
                        logger.info(
                            "[humanfeedback_agent] Running agent with your feedback (Modify path). " "Input keys: %s",
                            list(extra_inputs.keys()) if extra_inputs else [],
                        )
                    result = await self.run_agent_task(
                        agent_key=agent_key,
                        task_key=task_key,
                        text=stage_text,
                        pydantic_output_class=None,
                        chunk_size=stage_chunk_size,
                        max_workers=self.max_workers,
                        post_process=stage_post_process,
                        input_key=input_key,
                        default_result=stage_default_result,
                        extra_inputs=extra_inputs,
                    )

                    all_errors.extend(result.get("errors", []))

                    if is_first_stage:
                        merged = result_merger(result["results"], text)
                        merged = verify_merged_result(merged, text, task_type)
                        prev_output = merged
                        # Tag extraction output with the number of chunks that ran so
                        # downstream stages (alignment/judge) can target the same parallelism.
                        merged["_extraction_chunk_count"] = len(result.get("results") or [1])
                        pipeline_stages[task_key] = merged
                        # Add provenance for extraction stage by default (NER and resource)
                        ext_container = self._detect_container_key(merged)
                        if ext_container:
                            add_provenance_to_result(merged, ext_container, "extractor_agent")
                    else:
                        results_list = result.get("results") or []
                        # Combine multiple blobs (e.g. alignment Final Answer blocks) via postprocessing merge
                        if len(results_list) > 1 and task_key in ("alignment_task", "judge_task"):
                            # FIX (empty-entities root cause): detect container key from the actual
                            # result.  Previously TASK_KEY_TO_CONTAINER_KEY.get(task_key) was used
                            # first; for NER it returned "judge_resource" (wrong key) so the merge
                            # silently produced empty entities.  Now we always inspect the result.
                            # Example — NER judge returns {"judge_ner_terms": {"1": [...], "2": [...]}};
                            # _detect_container_key finds "judge_ner_terms" and merge proceeds correctly.
                            ckey = self._detect_container_key(results_list[0])
                            if ckey:
                                try:
                                    prev_output = merge_downstream_chunk_results_with_provenance(
                                        results_list, ckey, agent_key
                                    )
                                    pipeline_stages[task_key] = prev_output
                                except (AttributeError, TypeError, KeyError, ValueError) as e:
                                    logger.warning(
                                        "Merge of downstream chunk results failed (%s); using first result. %s",
                                        task_key,
                                        e,
                                        exc_info=True,
                                    )
                                    prev_output = results_list[0] if results_list else prev_output
                                    pipeline_stages[task_key] = prev_output
                            else:
                                prev_output = results_list[0] if results_list else prev_output
                                pipeline_stages[task_key] = prev_output
                        else:
                            raw = results_list[0] if results_list else prev_output
                            if isinstance(raw, dict):
                                # FIX (empty-entities root cause): always detect container key from
                                # the actual result dict.  The old guard
                                #   TASK_KEY_TO_CONTAINER_KEY.get(task_key) or _detect_container_key(raw)
                                # short-circuited for NER because TASK_KEY_TO_CONTAINER_KEY["judge_task"]
                                # == "judge_resource" is truthy, so the real key "judge_ner_terms"
                                # was never detected and provenance/merge logic ran on the wrong key.
                                # Example: NER single-result judge → raw = {"judge_ner_terms": {...}};
                                # _detect_container_key returns "judge_ner_terms" so provenance is
                                # stamped on the correct container.
                                container_key = self._detect_container_key(raw)
                                if container_key:
                                    add_provenance_to_result(raw, container_key, agent_key)
                            prev_output = raw
                            pipeline_stages[task_key] = raw

            # KEY ROBUSTNESS FIX: normalize stage output to canonical keys immediately
            # after every stage completes, before passing to the next stage.
            #
            # Root cause of the empty-entities bug:
            #   Each pipeline stage uses a stage-specific output key instead of the canonical
            #   "entities" / "resources" key.  Examples:
            #     - NER extractor  → {"extracted_terms":  {"1": [{entity...}], ...}}
            #     - NER alignment  → {"aligned_ner_terms": {"1": [{entity...}], ...}}
            #     - NER judge      → {"judge_ner_terms":   {"1": [{entity...}], ...}}
            #     - Resource judge → {"judge_resource":    [{resource...}, ...]}
            #   Without this call, verify_ner_result / normalize_final_result_for_output
            #   look for "entities" / "resources" at the top level, find nothing, and the
            #   final output is always empty even though the LLM produced valid data.
            #
            # What promote_stage_output_to_canonical does (see postprocessing.py):
            #   1. Unwrap any pipeline placeholder wrapper key
            #      (e.g. "judged_structured_information_with_human_feedback": {"entities": [...]})
            #   2. Detect the highest-priority stage key present in priority order
            #      NER:      judge_ner_terms > aligned_ner_terms > extracted_terms
            #      resource: judge_resource  > aligned_resources > extracted_resources
            #   3. Flatten dict-of-lists ({"1": [...], "2": [...]}) → flat list
            #   4. Write the result to the canonical key ("entities" or "resources")
            #
            # This single call covers all paths (chunked, token-split, normal) because
            # every path converges to prev_output before this line.
            # See promote_stage_output_to_canonical() in postprocessing.py for full details.
            if isinstance(prev_output, dict):
                prev_output = promote_stage_output_to_canonical(prev_output, task_type)
                pipeline_stages[task_key] = prev_output

            # Preserve alignment agent's concept mapping tool output after the alignment stage.
            #
            # Background:
            #   During the alignment stage the agent calls ConceptMappingLocalTool (or
            #   ConceptMappingTool for BioPortal).  Each call records its input/output in
            #   the module-level _ALIGNMENT_TOOL_OUTPUTS list (see conceptmappingtool.py /
            #   conceptmappinglocal.py).  Without the code below those results are captured
            #   but never written back to the pipeline output — callers would see no
            #   ontology information on entities even though the alignment agent resolved them.
            #
            # How we surface them per task type:
            # - extraction: store as a top-level "concept_mapping" list in prev_output so
            #     downstream stages (judge, humanfeedback) and the final result carry it.
            #     Example entry: {"term": "SNOMED", "ontology_id": "...", "ontology_label": "..."}
            #
            # - NER (FIX — previously missing entirely):
            #     Inject class_uri / ontology_label / ontology_id directly into each entity
            #     dict so the caller gets ontology info on the entity objects themselves.
            #     Example — entity before injection:
            #       {"entity": "scRNA-seq", "label": "Technique", "sentence": "..."}
            #     Entity after injection:
            #       {"entity": "scRNA-seq", "label": "Technique", "sentence": "...",
            #        "class_uri": "http://purl.obolibrary.org/obo/OBI_0002631",
            #        "ontology_label": "single cell RNA sequencing assay",
            #        "ontology_id": "OBI:0002631"}
            #     Only non-None values are written so no existing field is overwritten with None.
            #     Entities are guaranteed to be at the top level because
            #     promote_stage_output_to_canonical already ran just above.
            #
            # - resource: no injection here — resource alignment concept data lives inside
            #     the judge_resource / aligned_resources structures and is handled by the
            #     end-of-pipeline apply_concept_mapping_to_result call.
            # -------------------------------------------------------------------
            # LAYER 2 OF 3 — Post-alignment injection of concept mapping results
            # -------------------------------------------------------------------
            # WHY THIS EXISTS:
            #   The alignment LLM calls ConceptMappingLocalTool during its run.
            #   Each call appends {"input": term, "output": {ontology fields}} to
            #   _ALIGNMENT_TOOL_OUTPUTS (module-level, accumulated across the run).
            #   Without this block, those results are captured but never written
            #   back to the entity dicts — callers would see no ontology fields
            #   even though the alignment agent successfully resolved them.
            #
            # HOW IT WORKS:
            #   inject_alignment_concept_mapping_into_ner_entities builds a
            #   case-insensitive term → mapping lookup from _ALIGNMENT_TOOL_OUTPUTS
            #   and writes ontology_id / ontology_label / ontology into each entity
            #   dict in-place.  Only non-None values are written so no existing
            #   field is overwritten.
            #
            # NOTE — WHY A SECOND INJECTION IS STILL NEEDED (see Layer 3 below):
            #   This injection enriches alignment-stage entity dicts.  The judge
            #   stage then outputs brand-new entity dicts that do not carry these
            #   extra fields.  The final re-injection (after the pipeline loop)
            #   handles that case so ontology fields always reach the caller.
            if task_key == "alignment_task" and isinstance(prev_output, dict):
                session_outputs = get_alignment_tool_outputs()
                if session_outputs:
                    if task_type == "extraction":
                        concept_mapping_list = format_alignment_tool_outputs_as_concept_mapping(session_outputs)
                        if concept_mapping_list:
                            prev_output["concept_mapping"] = concept_mapping_list
                            logger.info("Preserved %d concept mappings from alignment agent tool output", len(concept_mapping_list))
                    elif task_type == "ner":
                        # Entities are at top level after promote_stage_output_to_canonical above.
                        entities = prev_output.get("entities") or []
                        if entities:
                            enriched = inject_alignment_concept_mapping_into_ner_entities(entities, session_outputs)
                            if enriched:
                                logger.info(
                                    "[alignment_task] Enriched %d NER entities with concept mapping "
                                    "(ontology_id/ontology_label/ontology)",
                                    enriched,
                                )
                    elif task_type in ("resource", "structured_extraction"):
                        # Inject ontology fields into all resource list keys.
                        # Mirrors the NER injection but uses resource "name" as the lookup key.
                        _l2_enriched = 0
                        for _rkey in ("resources", "aligned_resources", "extracted_resources"):
                            _l2_resources = prev_output.get(_rkey) or []
                            if _l2_resources:
                                _l2_enriched += inject_alignment_concept_mapping_into_resources(
                                    _l2_resources, session_outputs
                                )
                        if _l2_enriched:
                            logger.info(
                                "[alignment_task] Enriched %d resources with concept mapping "
                                "(ontology_id/ontology_label/ontology)",
                                _l2_enriched,
                            )

            # Record stage timing
            stage_elapsed = time.time() - stage_start_time
            stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
            logger.info(f"[{agent_key}] Completed in {stage_elapsed:.2f}s")
            _end_stage_llm_tracking(task_key, _stage_n_chunks, stage_elapsed)

            # -----------------------------------------------------------------------
            # Persist stage output to disk immediately after completion.
            #
            # WHY THIS EXISTS:
            #   Extraction takes ~10 min, alignment ~60 min, judge ~40 min.
            #   If the process crashes during the judge stage, extraction and alignment
            #   results are lost because they only live in `pipeline_stages` (in-memory).
            #   Writing each stage to disk right after it finishes means the data is
            #   safe.  Files are small (JSON), overwrites are atomic on most OS, and
            #   the cost is negligible vs. the LLM call time.
            #
            # FILE NAMING:
            #   <stage_index>_<agent_key>_<task_key>.json
            #   e.g. 00_extractor_agent_extraction_task.json
            #        01_alignment_agent_alignment_task.json
            #        02_judge_agent_judge_task.json
            #   Leading zero-padded index keeps files in pipeline order in file explorers.
            # -----------------------------------------------------------------------
            if self.stage_output_dir and pipeline_stages.get(task_key) is not None:
                try:
                    os.makedirs(self.stage_output_dir, exist_ok=True)
                    _stage_fname = f"{idx:02d}_{agent_key}_{task_key}.json"
                    _stage_path = os.path.join(self.stage_output_dir, _stage_fname)
                    with open(_stage_path, "w", encoding="utf-8") as _sf:
                        json.dump(pipeline_stages[task_key], _sf, indent=2, ensure_ascii=False, default=str)
                    logger.info("[stage_output] Wrote %s (%.1f KB)", _stage_path,
                                os.path.getsize(_stage_path) / 1024)
                except Exception as _se:
                    logger.warning("[stage_output] Failed to write stage file: %s", _se)

        elapsed_time = time.time() - start_time

        # Log total LLM call count for this run
        with _llm_call_lock:
            total_llm_calls = _llm_call_count["n"]
        llm_call_logger = logging.getLogger("llm_calls")
        if llm_call_logger.handlers:
            llm_call_logger.info("=" * 70)
            llm_call_logger.info("PIPELINE COMPLETE")
            llm_call_logger.info(f"  Total LLM calls : {total_llm_calls}")
            llm_call_logger.info(f"  Total elapsed   : {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
            if _llm_stage_tracker:
                llm_call_logger.info("")
                llm_call_logger.info(f"  {'Stage':<30} {'Chunks':>6}  {'Calls':>5}  {'Calls/chunk':>11}  {'Elapsed':>8}")
                llm_call_logger.info(f"  {'-'*30}  {'-'*6}  {'-'*5}  {'-'*11}  {'-'*8}")
                for _sk, _sv in _llm_stage_tracker.items():
                    if "calls" in _sv:
                        llm_call_logger.info(
                            f"  {_sk:<30} {_sv['n_chunks']:>6}  {_sv['calls']:>5}  "
                            f"{_sv['calls_per_chunk']:>10.1f}x  {_sv['elapsed']:>7.1f}s"
                        )
                llm_call_logger.info(f"  {'TOTAL':<30} {'':>6}  {total_llm_calls:>5}")
            llm_call_logger.info("=" * 70)
            _flush_llm_logger()
        logger.info("Total LLM calls this run: %d", total_llm_calls)

        # Log token usage statistics
        total_tokens = self.agent_context.get_total_tokens()
        utilization_pct = (total_tokens / self.token_limit) * 100 if self.token_limit > 0 else 0

        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Stages: {len(ordered_pairs)}")
        logger.info("")
        logger.info("Stage Timings:")
        for stage_name, stage_time in stage_timings.items():
            pct = (stage_time / elapsed_time * 100) if elapsed_time > 0 else 0
            logger.info(f"  {stage_name:50s} {stage_time:8.2f}s ({pct:5.1f}%)")
        logger.info("")
        logger.info(f"Token usage: {total_tokens:,} / {self.token_limit:,} ({utilization_pct:.1f}%)")
        logger.info(f"Context management: {len(self.agent_context.agent_results)} agents tracked")
        logger.info(f"Shared memory: {self.shared_memory.get_stats()['total_keys']} keys stored")
        logger.info("=" * 80)

        # NER ENTITY FALLBACK: if the last stage (judge / humanfeedback) returned no entity
        # data at all, fall back to the best earlier stage that has entities.
        #
        # Root cause this handles:
        #   The judge_task uses a pydantic output model that defines resource-type fields
        #   (resources, aligned_resources, judge_resource).  When the same task config is
        #   reused for NER, the LLM populates those fields (with empty lists) but never
        #   writes judge_ner_terms / entities.  After promote_stage_output_to_canonical,
        #   entities stays [] even though 19 entities existed after the alignment stage.
        #
        # Fallback priority: humanfeedback → judge → alignment → extraction.
        # We only fall back when:
        #   1. task_type is "ner" (resource tasks use a different set of keys)
        #   2. prev_output has entities == [] (empty after full promotion)
        #   3. None of the NER stage keys (judge_ner_terms / aligned_ner_terms /
        #      extracted_terms) are present in prev_output either — meaning the LLM
        #      returned the wrong structure entirely, not just an intentionally empty list.
        # If the judge intentionally returned entities=[] (all entities rejected), all
        # NER stage keys would also be absent, but that is indistinguishable from the
        # wrong-schema case.  We accept the occasional over-recovery in favour of not
        # silently losing valid extraction results.
        if task_type == "ner" and isinstance(prev_output, dict):
            _ner_keys = ("judge_ner_terms", "aligned_ner_terms", "extracted_terms")
            _no_ner_output = (
                not prev_output.get("entities")
                and not prev_output.get("key_terms")
                and not any(prev_output.get(k) for k in _ner_keys)
            )
            if _no_ner_output:
                # Walk pipeline stages from most-refined to least-refined looking for entities.
                _fallback_order = ("humanfeedback_task", "judge_task", "alignment_task", "extraction_task")
                for _fkey in _fallback_order:
                    _fstage = pipeline_stages.get(_fkey)
                    if isinstance(_fstage, dict) and _fstage.get("entities"):
                        logger.warning(
                            "[NER fallback] Last stage (%s) produced no entity data "
                            "(returned resource-type keys instead of judge_ner_terms). "
                            "Falling back to stage '%s' which has %d entities.",
                            list(ordered_pairs)[-1][1] if ordered_pairs else "unknown",
                            _fkey,
                            len(_fstage["entities"]),
                        )
                        prev_output = _fstage
                        break

        # Build result: last stage output (human feedback if enabled, else judge). Each stage extends the previous.
        # Unwrap if LLM returned output nested under pipeline placeholder key (e.g. judged_structured_information_with_human_feedback)
        final = dict(prev_output) if isinstance(prev_output, dict) else {}
        if humanfeedback_aborted_by_user:
            final["human_feedback_skipped"] = True  # User chose Abort; judge result saved without feedback step
        unwrap_keys = (
            "judged_structured_information_with_human_feedback",
            "aligned_structured_information",
            "extracted_structured_information",
        )
        if isinstance(final, dict) and len(final) >= 1:
            for key in unwrap_keys:
                inner = final.get(key)
                if inner is None:
                    continue
                if isinstance(inner, dict) and (
                    "entities" in inner or "key_terms" in inner or "resources" in inner or "judge_resource" in inner
                ):
                    # Use inner as base so entities/key_terms are at top level; keep other top-level keys
                    rest = {k: v for k, v in final.items() if k != key}
                    final = {**inner, **rest}
                    logger.debug("Unwrapped final result from key %s", key)
                    break
                # Judge/alignment may return a list of entities (e.g. after chunked merge) or a dict-of-lists
                if isinstance(inner, (list, dict)):
                    flattened = _flatten_container_to_list(inner)
                    if flattened and all(isinstance(x, dict) for x in flattened):
                        final["entities"] = flattened
                        final["key_terms"] = final.get("key_terms") or (
                            inner.get("key_terms") if isinstance(inner, dict) else []
                        )
                        logger.debug("Unwrapped final result from key %s (list/container, %d entities)", key, len(flattened))
                    break
        final["errors"] = all_errors
        final["task_type"] = task_type
        final["elapsed_time"] = elapsed_time

        # For extraction: carry over concept_mapping from alignment stage (judge/humanfeedback overwrite prev_output)
        if task_type == "extraction" and pipeline_stages.get("alignment_task") and isinstance(pipeline_stages["alignment_task"], dict):
            alignment_cm = pipeline_stages["alignment_task"].get("concept_mapping")
            if isinstance(alignment_cm, list) and alignment_cm:
                final["concept_mapping"] = alignment_cm

        # -------------------------------------------------------------------
        # LAYER 3 OF 3 — Final re-injection of concept mapping into definitive entities
        # -------------------------------------------------------------------
        # WHY THIS EXISTS:
        #   Layer 2 (post-alignment injection) enriches the alignment-stage entity dicts
        #   in-place.  But the pipeline does not stop there:
        #
        #     alignment stage → entities enriched with ontology fields ✓
        #     judge stage     → LLM outputs BRAND-NEW entity dicts from scratch
        #                       → ontology fields are NOT carried through ✗
        #     final assembly  → picks up judge entities → no ontology fields ✗
        #
        #   The judge LLM receives the aligned entities (with ontology fields) as context,
        #   but its Pydantic output model only defines the core entity schema (entity,
        #   label, sentence, etc.) — extra fields like ontology_id are silently dropped
        #   when the model is instantiated from the LLM's JSON output.
        #
        # WHAT WE DO:
        #   After all stages have run and final["entities"] holds the authoritative list,
        #   we re-run inject_alignment_concept_mapping_into_ner_entities against it.
        #   _ALIGNMENT_TOOL_OUTPUTS is module-level and accumulates results from both
        #   the programmatic pre-compute (Layer 1) and any calls the alignment LLM made
        #   itself, so this covers every term that was mapped during the entire run.
        #
        #   This is the last write to entity dicts before the result is returned to the
        #   caller, so ontology fields are guaranteed to be present regardless of which
        #   stage (judge, humanfeedback, fallback) produced the final entity list.
        if task_type == "ner":
            _final_entities = final.get("entities") or []
            if _final_entities:
                _final_session = get_alignment_tool_outputs()
                if _final_session:
                    _final_enriched = inject_alignment_concept_mapping_into_ner_entities(
                        _final_entities, _final_session
                    )
                    if _final_enriched:
                        logger.info(
                            "[final] Re-injected concept mapping into %d / %d NER entities",
                            _final_enriched,
                            len(_final_entities),
                        )

        # Include pipeline details only when requested (avoids repeating entities and heavy metadata)
        if self.return_full_pipeline_details:
            final["stage_timings"] = stage_timings
            final["pipeline_stages"] = pipeline_stages
            final["token_usage"] = {
                "total": total_tokens,
                "limit": self.token_limit,
                "utilization_pct": utilization_pct,
                "by_agent": {
                    agent_key: sum(r.token_count for r in results) for agent_key, results in self.agent_context.agent_results.items()
                },
            }
            final["context_management"] = {
                "agents_tracked": len(self.agent_context.agent_results),
                "shared_memory_keys": self.shared_memory.get_stats()["total_keys"],
                "agent_context": self.agent_context.to_dict(),
            }

        # End-of-pipeline verifier: ensure all text, sentences, entities present in source
        final = verify_merged_result(final, text, task_type)

        # Resource task: promote judge/aligned output into final["resources"] so concept mapping and output use refined list (with mapped_* from alignment/judge)
        if task_type in ("resource", "structured_extraction"):
            # Inject alignment stage so promote can prefer alignment's rich mapped_* when judge output is thin (e.g. after approve)
            if pipeline_stages.get("alignment_task") and isinstance(pipeline_stages["alignment_task"], dict):
                al = pipeline_stages["alignment_task"].get("aligned_resources")
                if al is not None:
                    final["aligned_resources"] = al
            promote_canonical_resources_for_resource_task(final)

        # Concept mapping at end: run only when user modified (option 3). Do not run when user approved (1) or aborted (4).
        # Skip flags are only set when human feedback was shown and user chose 1 or 4; NER/extraction/resource
        # behave unchanged when human feedback is disabled or when user chose Modify (3).
        skip_end_concept_mapping = human_approved_skip_concept_mapping or humanfeedback_aborted_by_user
        if skip_end_concept_mapping:
            logger.info("Saving result without end concept mapping (user approved or aborted; no modification).")
            # Ensure resource mapped concepts have provenance (tool vs llm_knowledge) when we did not run end concept mapping
            if task_type in ("resource", "structured_extraction") and isinstance(final.get("resources"), list) and final["resources"]:
                ensure_resource_mapped_concepts_provenance(final["resources"])
        else:
            extraction_has_alignment_concept_mapping = (
                task_type == "extraction" and isinstance(final.get("concept_mapping"), list) and len(final.get("concept_mapping", [])) > 0
            )
            if extraction_has_alignment_concept_mapping:
                logger.info(
                    "Using concept_mapping from alignment agent tool output (skipping end-of-pipeline concept mapping for extraction)."
                )
            else:
                logger.info("Applying concept mapping (task_type=%s, parallel, max_workers=%s)", task_type, self.max_workers or 8)
                final = apply_concept_mapping_to_result(final, task_type=task_type, max_workers=self.max_workers or 8)

        # Keep only task-specific canonical output (no separate aligned/judge keys), like NER.
        final = normalize_final_result_for_output(final, task_type)
        return final

    def _get_default_result_for_task(self, task_type: str) -> Dict[str, Any]:
        """Get default result structure based on task type."""
        defaults = {
            "ner": {"entities": [], "key_terms": []},
            "extraction": {},
            "resource": {"resources": []},
            "structured_extraction": {"resources": []},
        }
        return defaults.get(task_type, {})

    def _detect_container_key(self, result_dict: Dict[str, Any]) -> Optional[str]:
        """Return the first known container key present in result_dict, or None."""
        if not isinstance(result_dict, dict):
            return None
        for key in DOWNSTREAM_CONTAINER_KEYS:
            if key in result_dict and result_dict[key] is not None:
                return key
        return None

    def _get_ordered_agent_task_pairs(self) -> list:
        """Return list of (agent_key, task_key) in config order for pipeline execution.

        Stages are excluded when:
        - humanfeedback_task: enable_human_feedback is False (default)
        - any task_key in self.skip_stages (user-supplied list passed to __init__ or CLI)

        When a stage is skipped the previous stage's output is forwarded directly
        to the next non-skipped stage via the existing prev_output chain.
        """
        pairs = []
        for task_key_iter, task_data in self.task_config.items():
            if isinstance(task_data, dict) and "agent_id" in task_data:
                agent_key_iter = task_data["agent_id"]
                if agent_key_iter in self.agent_config:
                    pairs.append((agent_key_iter, task_key_iter))
        if not pairs and "extractor_agent" in self.agent_config and "extraction_task" in self.task_config:
            pairs = [("extractor_agent", "extraction_task")]
        # Exclude humanfeedback stage when human feedback is disabled
        if not self.enable_human_feedback:
            pairs = [p for p in pairs if p != ("humanfeedback_agent", "humanfeedback_task")]
        # Exclude any stage explicitly listed in skip_stages
        if self.skip_stages:
            skipped = set(self.skip_stages)
            if "extraction_task" in skipped:
                logger.warning(
                    "skip_stages contains 'extraction_task' — extraction is the first stage and "
                    "cannot be skipped. Use preloaded_stages={'extraction_task': ...} instead. "
                    "Ignoring 'extraction_task' in skip_stages."
                )
                skipped.discard("extraction_task")
            removed = {p[1] for p in pairs if p[1] in skipped}
            pairs = [p for p in pairs if p[1] not in skipped]
            if removed:
                logger.info("Skipping pipeline stage(s): %s", sorted(removed))
        return pairs

    def _split_downstream_payload(self, payload: Dict[str, Any], max_chars: int) -> list:
        """Split a downstream payload into chunks so each chunk's JSON size is <= max_chars.

        Looks for a container key (extracted_resources, aligned_resources, judge_resource, resources)
        whose value is a dict (id -> list) or a list; splits that into batches and returns a list of
        payload dicts (same structure, subset of items per chunk).
        """
        if not isinstance(payload, dict):
            return [payload] if payload is not None else []
        serialized = json.dumps(payload, indent=2)
        if len(serialized) <= max_chars:
            return [payload]

        # Prefer known container keys in order
        container_keys = (
            "extracted_resources",
            "aligned_resources",
            "judge_resource",
            "resources",
            "extracted_structured_information",
            "aligned_structured_information",
            "judged_structured_information_with_human_feedback",
        )
        container_key = None
        container = None
        for key in container_keys:
            if key in payload and payload[key] is not None:
                val = payload[key]
                if isinstance(val, dict) or isinstance(val, list):
                    container_key = key
                    container = val
                    break
        if container_key is None:
            # Fallback: pick first key that is dict or list
            for key, val in payload.items():
                if isinstance(val, dict) and val or isinstance(val, list) and val:
                    container_key = key
                    container = val
                    break
        if container_key is None:
            return [payload]

        if isinstance(container, list):
            # Split list into batches by size
            batches = []
            current = []
            current_size = 2  # "[]"
            for item in container:
                item_str = json.dumps(item)
                if current_size + len(item_str) + 2 > max_chars and current:
                    batches.append(current)
                    current = []
                    current_size = 2
                current.append(item)
                current_size += len(item_str) + 2
            if current:
                batches.append(current)
            return [{**{k: v for k, v in payload.items() if k != container_key}, container_key: batch} for batch in batches]

        # container is dict (id -> list or value)
        keys_order = list(container.keys())
        batches = []
        current_keys = []
        current_payload = {k: v for k, v in payload.items() if k != container_key}
        current_sub = {}
        current_size = len(json.dumps(current_payload)) + 20
        for k in keys_order:
            v = container[k]
            v_str = json.dumps(v)
            if current_size + len(k) + len(v_str) + 4 > max_chars and current_sub:
                batches.append({**current_payload, container_key: dict(current_sub)})
                current_sub = {}
                current_size = len(json.dumps(current_payload)) + 20
            current_sub[k] = v
            current_size += len(k) + len(v_str) + 4
        if current_sub:
            batches.append({**current_payload, container_key: dict(current_sub)})
        return batches if batches else [payload]

    def _merge_downstream_chunk_results(self, chunk_results: list) -> Dict[str, Any]:
        """Merge results from multiple downstream chunk runs into one payload.

        Each chunk result is a dict (e.g. aligned_resources, judge_resource). We merge by
        updating the same container key from each result.
        """
        if not chunk_results:
            return {}
        if len(chunk_results) == 1:
            return dict(chunk_results[0]) if isinstance(chunk_results[0], dict) else {}

        merged = {}
        container_keys = ("extracted_resources", "aligned_resources", "judge_resource", "resources")
        for result in chunk_results:
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                if key in container_keys and value is not None:
                    if key not in merged:
                        merged[key] = {} if isinstance(value, dict) else []
                    if isinstance(value, dict) and isinstance(merged[key], dict):
                        merged[key].update(value)
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        merged[key].extend(value)
                elif key not in merged:
                    merged[key] = value
        return merged

    def _get_detected_task_type(self, agent_key: str, task_key: str) -> str:
        """Detect task type for tool selection and merging (taxonomy-aligned).

        Uses LLM-based detect_task_type when API key and LLM config are available;
        otherwise falls back to a heuristic so resource/structured extraction get
        no NER tool and correct post-processor/merger.

        Once detected for a pipeline run, the result is cached in
        ``self._pipeline_task_type`` and returned directly on all subsequent
        calls within the same run (avoids multiple LLM API calls that could
        return inconsistent results across stages).
        """
        if self._pipeline_task_type is not None:
            logger.debug(f"Reusing cached task type '{self._pipeline_task_type}' for {agent_key}/{task_key}")
            return self._pipeline_task_type

        task_data = self.task_config.get(task_key) or {}
        description = task_data.get("description", "") or ""
        if not description and isinstance(task_data, dict):
            description = str(task_data)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        agent_id = task_data.get("agent_id", agent_key)
        llm_config = self.agent_config.get(agent_id, {}).get("llm", {})
        if api_key and llm_config and (llm_config.get("model") or llm_config.get("base_url")):
            try:
                result = detect_task_type(
                    taskconfig=description,
                    api_key=api_key,
                    llm_config=llm_config,
                )
                logger.info(f"Detected task type '{result.task_type}' (confidence={result.confidence:.2f})")
                self._pipeline_task_type = result.task_type
                return self._pipeline_task_type
            except Exception as e:
                logger.warning(f"Task detection failed: {e}, using heuristic")
        # Heuristic: NER vs resource vs extraction
        text = (description or str(self.task_config)).lower()
        if "ner" in text or "named entity" in text or "entity" in text:
            detected = "ner"
        elif "resource" in text and ("extract" in text or "dataset" in text or "tool" in text or "model" in text):
            detected = "resource"
        elif "structured extraction" in text or "structured_extraction" in text:
            detected = "structured_extraction"
        else:
            detected = "extraction"
        self._pipeline_task_type = detected
        return self._pipeline_task_type

    def _initialize_agent_and_task(
        self,
        agent_key: str,
        task_key: str,
        pydantic_output_class: Optional[Any] = None,
        tools: Optional[list] = None,
    ) -> Tuple[Optional[object], Optional[object]]:
        """Initialize an agent and its associated task.

        Uses the robust initialization from crew_utils. Tools are chosen
        dynamically from the detected task type (only NER/keyphrase get tools).
        """
        _extra_agent_kwargs: Dict[str, Any] = {}
        if self.agent_max_execution_time is not None:
            _extra_agent_kwargs["max_execution_time"] = self.agent_max_execution_time
        if self.agent_max_retry_limit is not None:
            _extra_agent_kwargs["max_retry_limit"] = self.agent_max_retry_limit
        return initialize_agent_and_task(
            agent_config=self.agent_config,
            task_config=self.task_config,
            agent_key=agent_key,
            task_key=task_key,
            embedder_config=self.embedder_config,
            tools=tools if tools is not None else [],
            pydantic_output=pydantic_output_class,
            max_iter=self.agent_max_iter,
            **_extra_agent_kwargs,
        )


async def kickoff(
    agentconfig: Union[str, dict],
    taskconfig: Union[str, dict],
    embedderconfig: Union[str, dict],
    source: Optional[str] = None,
    source_text: Optional[str] = None,
    knowledgeconfig: Optional[Union[str, dict]] = None,
    enable_human_feedback: bool = True,
    agent_feedback_config: Optional[Dict[str, bool]] = None,
    env_file: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_chunking: bool = False,
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    downstream_max_input_chars: Optional[int] = None,
    max_extraction_chunk_chars: Optional[int] = None,
) -> Union[Dict[str, Any], str]:
    """
    Standalone kickoff function for backward compatibility.

    This function now uses StructSenseFlow internally as the single entry point.
    All functionality is preserved.
    """
    # Use StructSenseFlow as the single entry point
    try:
        flow = StructSenseFlow(
            agent_config=agentconfig,
            task_config=taskconfig,
            embedder_config=embedderconfig,
            source=source,
            source_text=source_text,
            knowledge_config=knowledgeconfig,
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=agent_feedback_config,
            env_file=env_file,
            api_key=api_key,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            max_workers=max_workers,
            downstream_max_input_chars=downstream_max_input_chars,
            max_extraction_chunk_chars=max_extraction_chunk_chars,
        )

        # Run the kickoff method
        return await flow.kickoff()

    except Exception as e:
        logger.error(f"Kickoff execution failed: {str(e)}")
        raise
