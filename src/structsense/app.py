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
from pathlib import Path
from datetime import datetime

# Filter warnings at the beginning
import warnings
from typing import Any, Dict, Optional, Tuple, Union

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
        source: Optional[str] = None,
        source_text: Optional[str] = None,
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
        # Max input size (chars) for downstream agents (alignment, judge, humanfeedback) to avoid context limit.
        # Default ~80k chars (~20k tokens); 128k tokens ≈ 512k chars.
        self.downstream_max_input_chars = downstream_max_input_chars if downstream_max_input_chars is not None else 80_000
        # Cap extraction chunk size so (chunk + prompt) stays under model context; token limits vary by model.
        # Default 25000 chars (~6k tokens) leaves room for task prompt on 128k models. None = no cap.
        self.max_extraction_chunk_chars = max_extraction_chunk_chars if max_extraction_chunk_chars is not None else 25_000
        self.return_full_pipeline_details = return_full_pipeline_details
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

        logger.info("Enhanced context management initialized:")
        logger.info(f"  - Token limit: {self.token_limit}")
        logger.info(f"  - Available tokens: {self.context_manager.available_tokens}")
        logger.info(f"  - Thread-safe memory: enabled")
        logger.info(f"  - Context passing: enabled")

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
    ):
        """
        Run the FULL multi-agent pipeline with context management.

        Executes all agents in sequence: extraction → alignment → judge → humanfeedback
        Each stage receives context from previous stages with automatic token management.

        Args:
            text: Optional input text (uses self.source_text if None)
            modification_context: Optional context for modifications
            user_feedback_text: Optional user feedback for humanfeedback stage

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
        elif task_type in ("resource", "structured_extraction"):
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

            # Clear alignment-stage tool outputs so we only capture this stage's concept mapping (for extraction)
            if task_key == "alignment_task":
                clear_alignment_tool_outputs()

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

                # Prepare token-managed input based on agent type
                if task_key == "alignment_task":
                    logger.info(f"[{agent_key}] Preparing token-managed input for alignment agent")
                    managed_input = prepare_alignment_agent_input(
                        extraction_results=prev_output,
                        original_text=text,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                    )
                    # Use managed input as extra_inputs
                    extra_inputs = managed_input
                    stage_text = None  # Will use extra_inputs instead
                elif task_key == "judge_task":
                    logger.info(f"[{agent_key}] Preparing token-managed input for judge agent")
                    # Get extraction results from context if available
                    extraction_results = None
                    if idx >= 2:  # There was an extraction stage before alignment
                        extraction_agent_key, extraction_task_key = ordered_pairs[0]
                        extraction_result = self.agent_context.get_latest_result(extraction_agent_key)
                        if extraction_result:
                            extraction_results = extraction_result.result

                    managed_input = prepare_judge_agent_input(
                        alignment_results=prev_output,
                        extraction_results=extraction_results,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                    )
                    extra_inputs = managed_input
                    stage_text = None  # Will use extra_inputs instead
                else:
                    # For other downstream stages or if not alignment/judge, use JSON string
                    stage_text = json.dumps(prev_output, indent=2) if isinstance(prev_output, dict) else str(prev_output)

                    # Check token limit and compress if needed
                    current_tokens = self.context_manager.count_tokens(stage_text)
                    if current_tokens > self.token_limit:
                        logger.warning(
                            f"[{agent_key}] Input exceeds token limit ({current_tokens}/{self.token_limit}). " "Applying compression..."
                        )
                        compressed = self.context_manager.prepare_for_downstream_agent(
                            results=prev_output if isinstance(prev_output, dict) else {"output": prev_output},
                            agent_key=agent_key,
                            max_tokens=self.token_limit,
                        )
                        stage_text = json.dumps(compressed, indent=2)
                        final_tokens = self.context_manager.count_tokens(stage_text)
                        logger.info(f"[{agent_key}] Compressed: {current_tokens} -> {final_tokens} tokens")

                stage_chunk_size = None
                stage_post_process = None
                stage_default_result = self._get_default_result_for_task(self._get_detected_task_type(agent_key, task_key))

            # Human feedback receives judge output: prev_output at this point is the judge stage result
            if task_key == "humanfeedback_task" and prev_output is not None:
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
                    # If still no feedback, use a default (only when we will run the agent)
                    if not feedback_text:
                        feedback_text = modification_context or "No specific feedback provided."

                    # Prepare token-managed input for humanfeedback agent (only when running agent)
                    alignment_for_human = None
                    alignment_ctx = self.agent_context.get_latest_result("alignment_agent") if self.agent_context else None
                    if alignment_ctx:
                        alignment_for_human = alignment_ctx.result
                    logger.info(f"[{agent_key}] Preparing token-managed input for humanfeedback agent")
                    managed_input = prepare_humanfeedback_agent_input(
                        judge_results=prev_output,
                        user_feedback=feedback_text,
                        alignment_results=alignment_for_human,
                        agent_context=self.agent_context,
                        context_manager=self.context_manager,
                        max_tokens=self.token_limit,
                    )
                    extra_inputs = managed_input
                    stage_text = None  # Will use extra_inputs instead
            elif task_key != "humanfeedback_task" and not is_first_stage and extra_inputs is None:
                # For other downstream stages, extra_inputs was set above
                pass

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
                    container_key = TASK_KEY_TO_CONTAINER_KEY.get(task_key) or self._detect_container_key(downstream_results[0])
                    if container_key:
                        prev_output = merge_downstream_chunk_results_with_provenance(downstream_results, container_key, agent_key)
                    else:
                        prev_output = self._merge_downstream_chunk_results(downstream_results)
                pipeline_stages[task_key] = prev_output
            elif humanfeedback_approved_skip_run:
                # Option 1 (Approve): prev_output and pipeline_stages already set; no agent run
                pass
            else:
                # Token-based chunking for alignment/judge/humanfeedback when payload exceeds context limit
                use_chunked = not is_first_stage and extra_inputs and task_key in ("alignment_task", "judge_task", "humanfeedback_task")
                payload_key = (
                    "extracted_structured_information"
                    if task_key == "alignment_task"
                    else "aligned_structured_information"
                    if task_key == "judge_task"
                    else "judged_structured_information_with_human_feedback"
                )
                payload = extra_inputs.get(payload_key) if use_chunked else None
                token_budget = int(self.token_limit * 0.6)
                over_limit = isinstance(payload, dict) and self.context_manager.estimate_tokens(payload) > token_budget
                if use_chunked and over_limit:
                    chunks = split_structured_payload(
                        payload,
                        self.context_manager,
                        token_budget,
                        max_entities_per_chunk=70,
                        max_key_terms_per_chunk=25,
                        max_resources_per_chunk=15,
                    )

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
                            "[%s] Running %s chunks in parallel (payload over token limit)",
                            task_key,
                            len(chunks),
                        )
                        chunk_results = await asyncio.gather(*[run_one_structured_chunk(c) for c in chunks])
                    else:
                        chunk_results = [await run_one_structured_chunk(chunks[0])]

                    for r in chunk_results:
                        all_errors.extend(r.get("errors", []))
                    raw_list = [r["results"][0] for r in chunk_results if r.get("results") and isinstance(r["results"][0], dict)]
                    if raw_list:
                        ckey = TASK_KEY_TO_CONTAINER_KEY.get(task_key) or self._detect_container_key(raw_list[0])
                        # For alignment/judge with multiple chunks: use resource-aware merge so tool-backed concepts (real IDs) beat N/A
                        if ckey and task_key in ("alignment_task", "judge_task") and len(raw_list) > 1:
                            prev_output = merge_downstream_chunk_results_with_provenance(raw_list, ckey, agent_key)
                        else:
                            prev_output = merge_structured_chunk_results(raw_list)
                            if isinstance(prev_output, dict) and ckey:
                                add_provenance_to_result(prev_output, ckey, agent_key)
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
                        pipeline_stages[task_key] = merged
                        # Add provenance for extraction stage by default (NER and resource)
                        ext_container = self._detect_container_key(merged)
                        if ext_container:
                            add_provenance_to_result(merged, ext_container, "extractor_agent")
                    else:
                        results_list = result.get("results") or []
                        # Combine multiple blobs (e.g. alignment Final Answer blocks) via postprocessing merge
                        if len(results_list) > 1 and task_key in ("alignment_task", "judge_task"):
                            ckey = TASK_KEY_TO_CONTAINER_KEY.get(task_key) or self._detect_container_key(results_list[0])
                            if ckey:
                                prev_output = merge_downstream_chunk_results_with_provenance(results_list, ckey, agent_key)
                                pipeline_stages[task_key] = prev_output
                            else:
                                prev_output = results_list[0] if results_list else prev_output
                                pipeline_stages[task_key] = prev_output
                        else:
                            raw = results_list[0] if results_list else prev_output
                            if isinstance(raw, dict):
                                container_key = TASK_KEY_TO_CONTAINER_KEY.get(task_key) or self._detect_container_key(raw)
                                if container_key:
                                    add_provenance_to_result(raw, container_key, agent_key)
                            prev_output = raw
                            pipeline_stages[task_key] = raw

            # Preserve alignment agent's concept mapping tool output only for extraction (pdf2_reproschema).
            # For NER and resource we do not inject concept_mapping into prev_output so judge input shape is unchanged.
            if task_key == "alignment_task" and task_type == "extraction" and isinstance(prev_output, dict):
                session_outputs = get_alignment_tool_outputs()
                if session_outputs:
                    concept_mapping_list = format_alignment_tool_outputs_as_concept_mapping(session_outputs)
                    if concept_mapping_list:
                        prev_output["concept_mapping"] = concept_mapping_list
                        logger.info("Preserved %d concept mappings from alignment agent tool output", len(concept_mapping_list))

            # Record stage timing
            stage_elapsed = time.time() - stage_start_time
            stage_timings[f"{agent_key}_{task_key}"] = stage_elapsed
            logger.info(f"[{agent_key}] Completed in {stage_elapsed:.2f}s")

        elapsed_time = time.time() - start_time

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
                if isinstance(inner, dict) and (
                    "entities" in inner or "key_terms" in inner or "resources" in inner or "judge_resource" in inner
                ):
                    # Use inner as base so entities/key_terms are at top level; keep other top-level keys
                    rest = {k: v for k, v in final.items() if k != key}
                    final = {**inner, **rest}
                    logger.debug("Unwrapped final result from key %s", key)
                    break
        final["errors"] = all_errors
        final["task_type"] = task_type
        final["elapsed_time"] = elapsed_time

        # For extraction: carry over concept_mapping from alignment stage (judge/humanfeedback overwrite prev_output)
        if task_type == "extraction" and pipeline_stages.get("alignment_task") and isinstance(pipeline_stages["alignment_task"], dict):
            alignment_cm = pipeline_stages["alignment_task"].get("concept_mapping")
            if isinstance(alignment_cm, list) and alignment_cm:
                final["concept_mapping"] = alignment_cm

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
        Human-feedback stage is included only when enable_human_feedback is True.
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
        """
        task_data = self.task_config.get(task_key, {})
        task_type = self.task_config.get(task_key, {}).get("task_type")
        if task_type in DEFAULT_TAXONOMY:
            logger.info(f"Using task type from agent config for agent '{agent_key}': {task_type}")
            return task_type
        elif task_type:
            logger.warning(
                f"Task config for '{task_key}' specifies task type '{task_type}' which is not in the default taxonomy list. Falling back to detection."
            )
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
                return result.task_type
            except Exception as e:
                logger.warning(f"Task detection failed: {e}, using heuristic")
        # Heuristic: NER vs resource vs extraction
        text = (description or str(self.task_config)).lower()
        if "ner" in text or "named entity" in text or "entity" in text:
            return "ner"
        if "resource" in text and ("extract" in text or "dataset" in text or "tool" in text or "model" in text):
            return "resource"
        if "structured extraction" in text or "structured_extraction" in text:
            return "structured_extraction"
        return "extraction"

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
        return initialize_agent_and_task(
            agent_config=self.agent_config,
            task_config=self.task_config,
            agent_key=agent_key,
            task_key=task_key,
            embedder_config=self.embedder_config,
            tools=tools if tools is not None else [],
            pydantic_output=pydantic_output_class,
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
