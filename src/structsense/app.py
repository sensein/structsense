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
    process_input_data,
    replace_api_key,
    str_to_bool,
    check_ollama_health,
)
from utils.task_detection import detect_task_type
from utils.task_tools import get_tools_for_agent
from utils.crew_utils import initialize_memory
from utils.mlops import setup_monitoring

from utils.text_chunking import (
    _chunk_doc_by_sentences,
    _get_sentence_info_for_span,
    _validate_text_presence,
    _globalize_entities,
)
from utils.crew_utils import run_crew_extraction, run_crew_extraction_async, initialize_agent_and_task
from utils.postprocessing import (
    get_post_processor,
    get_result_merger,
)
from .humanloop import HumanInTheLoop


# Start memory tracking
tracemalloc.start()

# Configure logging - filter out warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s"
)
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


# Mapping from task_key to the input placeholder name expected by that task in the pipeline.
# Used when chaining agents: each stage receives the previous stage's output under this key.
PIPELINE_INPUT_KEY_MAP = {
    "extraction_task": "input_text",
    "alignment_task": "extracted_structured_information",
    "judge_task": "aligned_structured_information",
    "humanfeedback_task": "judged_structured_information_with_human_feedback",
}


class StructSenseFlow:
    """A workflow for structured information extraction, alignment, and judgment using CrewAI.
    Includes improved crew communication and shared memory.
    
    This is the single entry point for all extraction workflows.
    """

    def __init__(
            self,
            agent_config: Union[str, dict],
            task_config: Union[str, dict],
            embedder_config: Union[str, dict],
            source_text: Optional[str] = None,
            input_source: Optional[Union[str, dict]] = None,
            enable_human_feedback: bool = False,
            enable_chunking: bool = False,
            knowledge_config: Optional[Union[str, dict]] = None,
            agent_feedback_config: Optional[Dict[str, bool]] = None,
            env_file: Optional[str] = None,
            api_key: Optional[str] = None,
            chunk_size: Optional[int] = None,
            max_workers: Optional[int] = None,
    ):
        super().__init__()
        
        # Setup environment first
        if env_file:
            load_dotenv(env_file, override=True)
            logger.info(f"Loaded environment variables from {env_file} (override=True)")
        else:
            load_dotenv()
            logger.info("Loaded environment variables from default .env")

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

        # Process input source if provided (takes precedence over source_text)
        if input_source is not None:
            processed_data = process_input_data(input_source)
            # Convert processed data to string
            if isinstance(processed_data, dict):
                if "sections" in processed_data:
                    text_parts = []
                    for section in processed_data.get("sections", []):
                        if isinstance(section, dict):
                            heading = section.get("heading", "")
                            content = section.get("content", "")
                            if heading:
                                text_parts.append(f"{heading}\n{content}")
                            else:
                                text_parts.append(content)
                    self.source_text = "\n\n".join(text_parts)
                elif "text" in processed_data:
                    title = processed_data.get("title", "")
                    text = processed_data.get("text", "")
                    if title and text:
                        self.source_text = f"{title}\n\n{text}"
                    elif text:
                        self.source_text = text
                    else:
                        self.source_text = str(processed_data)
                elif "error" in processed_data:
                    error_msg = processed_data.get("error", "Unknown error processing input")
                    raise ConfigError(f"Input processing failed: {error_msg}")
                else:
                    # Unknown dict format, try to extract text-like values
                    text_parts = []
                    for key, value in processed_data.items():
                        if isinstance(value, str) and len(value) > 10:
                            text_parts.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    text_parts.append(item)
                                elif isinstance(item, dict) and "content" in item:
                                    text_parts.append(item.get("content", ""))
                    if text_parts:
                        self.source_text = "\n\n".join(text_parts)
                    else:
                        logger.warning(f"Unknown dict format from process_input_data: {list(processed_data.keys())}")
                        self.source_text = str(processed_data)
            elif isinstance(processed_data, list):
                logger.warning("List input detected, converting to string representation")
                self.source_text = "\n".join(str(item) for item in processed_data)
            else:
                self.source_text = processed_data
        elif source_text is not None:
            self.source_text = source_text
        else:
            raise ConfigError("Either source_text or input_source must be provided")

        # Validate that we have text to process
        if not self.source_text or not isinstance(self.source_text, str):
            raise ConfigError("No valid text content could be extracted from input source")
        
        if len(self.source_text.strip()) == 0:
            raise ConfigError("Extracted text is empty")

        logger.info(f"Initializing StructSenseFlow with source text (length: {len(self.source_text)} chars)")
        self.enable_human_feedback = enable_human_feedback

        try:
            # Load configs if they are file paths, otherwise use as-is
            if isinstance(agent_config, str):
                self.agent_config = load_config(agent_config, "agent_config")
            else:
                self.agent_config = agent_config
                
            if isinstance(task_config, str):
                self.task_config = load_config(task_config, "task_config")
            else:
                self.task_config = task_config
                
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
        
        # Initialize memory
        self.long_term_memory = None
        self.short_term_memory = None
        self.entity_memory = None
        
        if self.embedder_config:
            try:
                # Check if using Ollama and if it's available
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
                        # Non-Ollama embedder (e.g., OpenAI)
                        self.long_term_memory, self.short_term_memory, self.entity_memory = initialize_memory(
                            embedder_config=self.embedder_config
                        )
                else:
                    self.long_term_memory, self.short_term_memory, self.entity_memory = initialize_memory(
                        embedder_config=self.embedder_config
                    )
                
                if any([self.long_term_memory, self.short_term_memory, self.entity_memory]):
                    logger.info("Memory systems initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize memory: {e}. Continuing without memory.")
        
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size or 2000  # Default chunk size
        self.max_workers = max_workers
        self.agent_feedback_config = agent_feedback_config or {}
        # Human-in-the-loop for feedback before humanfeedback_agent (see humanloop.py)
        self.human_loop = HumanInTheLoop(
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=self.agent_feedback_config,
        )





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
        task_type = self._get_detected_task_type(agent_key, task_key)
        tools = get_tools_for_agent(agent_key, task_type)

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
        Main entry point for extraction workflow with full timing and logging.
        
        This method automatically discovers agent-task pairs from config or uses
        the provided agent_key/task_key. It includes comprehensive timing logs
        and handles all the extraction workflow.
        
        Args:
            agent_key: Optional specific agent key (auto-discovered if None)
            task_key: Optional specific task key (auto-discovered if None)
            
        Returns:
            Dict with merged results, errors, task_type, elapsed_time, etc.
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
            tools = get_tools_for_agent(agent_key, task_type)

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

            logger.info("#"*100)
            logger.info(f"Extraction completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Timing details saved to: {timing_log_file}")
            logger.info("#"*100)
            
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

    async def information_extraction_task(
        self,
        text: Optional[str] = None,
        modification_context: Optional[str] = None,
        user_feedback_text: Optional[str] = None,
    ):
        """Run the full pipeline: all agent-task pairs from config in order, passing each result to the next.
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
        prev_output = None

        for idx, (agent_key, task_key) in enumerate(ordered_pairs):
            is_first_stage = idx == 0
            input_key = PIPELINE_INPUT_KEY_MAP.get(task_key, "input_text")

            if is_first_stage:
                # First stage: source text, chunking, post-processing, merger
                stage_text = text
                stage_chunk_size = self.chunk_size if self.enable_chunking else None
                stage_post_process = post_processor
                stage_default_result = default_result
            else:
                # Downstream stages: previous stage output as JSON, no chunking
                if prev_output is None:
                    logger.warning(f"Skipping {agent_key}/{task_key}: no previous output")
                    continue
                stage_text = json.dumps(prev_output, indent=2) if isinstance(prev_output, dict) else str(prev_output)
                stage_chunk_size = None
                stage_post_process = None
                stage_default_result = self._get_default_result_for_task(
                    self._get_detected_task_type(agent_key, task_key)
                )

            # Optional extra inputs for humanfeedback_task (from humanloop or explicit args)
            extra_inputs = None
            if task_key == "humanfeedback_task":
                extra_inputs = {}
                # Use humanloop to request feedback when enabled and no explicit feedback passed
                if modification_context is None and user_feedback_text is None and self.human_loop.is_feedback_enabled_for_agent("humanfeedback_agent"):
                    feedback_result = self.human_loop.request_feedback(
                        prev_output,
                        step_name="human_feedback_processing",
                        agent_name="humanfeedback_agent",
                    )
                    if isinstance(feedback_result, dict):
                        if feedback_result.get("user_feedback_text"):
                            extra_inputs["user_feedback_text"] = feedback_result["user_feedback_text"]
                        if feedback_result.get("user_feedback_json") is not None:
                            extra_inputs["modification_context"] = json.dumps(feedback_result["user_feedback_json"], indent=2)
                if modification_context is not None:
                    extra_inputs["modification_context"] = modification_context
                if user_feedback_text is not None:
                    extra_inputs["user_feedback_text"] = user_feedback_text
                if not extra_inputs:
                    extra_inputs = None

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
                prev_output = merged
                pipeline_stages[task_key] = merged
            else:
                raw = result["results"][0] if result.get("results") else prev_output
                prev_output = raw
                pipeline_stages[task_key] = raw

        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds ({len(ordered_pairs)} stages)")

        # Build result: last stage output at top level, plus errors/task_type/elapsed_time/pipeline_stages
        final = dict(prev_output) if isinstance(prev_output, dict) else {}
        final["errors"] = all_errors
        final["task_type"] = task_type
        final["elapsed_time"] = elapsed_time
        final["pipeline_stages"] = pipeline_stages
        # Backward compat: if only extraction ran, top-level already has merged result; if full pipeline, keep last stage at top and use pipeline_stages for per-stage outputs
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

    def _get_ordered_agent_task_pairs(self) -> list:
        """Return list of (agent_key, task_key) in config order for pipeline execution."""
        pairs = []
        for task_key_iter, task_data in self.task_config.items():
            if isinstance(task_data, dict) and "agent_id" in task_data:
                agent_key_iter = task_data["agent_id"]
                if agent_key_iter in self.agent_config:
                    pairs.append((agent_key_iter, task_key_iter))
        if not pairs and "extractor_agent" in self.agent_config and "extraction_task" in self.task_config:
            pairs = [("extractor_agent", "extraction_task")]
        return pairs

    def _get_detected_task_type(self, agent_key: str, task_key: str) -> str:
        """Detect task type for tool selection and merging (taxonomy-aligned).

        Uses LLM-based detect_task_type when API key and LLM config are available;
        otherwise falls back to a heuristic so resource/structured extraction get
        no NER tool and correct post-processor/merger.
        """
        task_data = self.task_config.get(task_key) or {}
        description = task_data.get("description", "") or ""
        if not description and isinstance(task_data, dict):
            description = str(task_data)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        agent_id = task_data.get("agent_id", agent_key)
        llm_config = (self.agent_config.get(agent_id) or {}).get("llm") or {}
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
        input_source: Union[str, dict],
        knowledgeconfig: Optional[Union[str, dict]] = None,
        enable_human_feedback: bool = True,
        agent_feedback_config: Optional[Dict[str, bool]] = None,
        env_file: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_chunking: bool = False,
        chunk_size: Optional[int] = None,
        max_workers: Optional[int] = None,
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
            input_source=input_source,
            knowledge_config=knowledgeconfig,
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=agent_feedback_config,
            env_file=env_file,
            api_key=api_key,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )
        
        # Run the kickoff method
        return await flow.kickoff()
        
    except Exception as e:
        logger.error(f"Kickoff execution failed: {str(e)}")
        raise
