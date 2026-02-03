# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------
 
# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : crew_utils.py
# @Software: PyCharm

import copy
import json
import os
import logging
import time
import asyncio
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*serialization.*")

# Disable CrewAI tracing BEFORE importing crewai
# This must be set before any crewai imports
# os.environ["CREWAI_TRACING_ENABLED"] = "false"
# # Disable CrewAI telemetry only
# os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
# # Disable all OpenTelemetry (including CrewAI)
# os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from crewai import Crew, Agent, Task, Process
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.paths import db_storage_path

# Import chunking functions and nlp model
from .text_chunking import _chunk_doc_by_sentences
from .tools import get_spacy_model
from .utils import check_ollama_health
from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask

# Import new context management modules
from .agent_context import AgentContext, ThreadSafeMemory
from .context_window_manager import ContextWindowManager

logger = logging.getLogger(__name__)

# Suppress warning logs
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("pydantic_core").setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", category=UserWarning)
# Suppress Pydantic serialization warnings specifically (using string patterns)
warnings.filterwarnings("ignore", message=".*Pydantic.*")
warnings.filterwarnings("ignore", message=".*serialization.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields.*")




def _run_crew_on_retry(
    crew: Crew,
    text: str,
    input_key: str = "input_text",
    default_result: Optional[Dict[str, Any]] = None,
    extra_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crew runner (generic) - Synchronous version.

    Args:
        crew: The Crew instance to run
        text: The input text to process
        input_key: The key to use in the inputs dict (default: "input_text")
        default_result: Default result dict if parsing fails (default: empty dict)
        extra_inputs: Optional extra key-value inputs merged into crew inputs (e.g. modification_context)

    Returns:
        Dict with parsed results. If both attempts fail, returns:
          {**default_result, "error": "..."}
    """
    if default_result is None:
        default_result = {}
    inputs = {input_key: text}
    if extra_inputs:
        inputs.update(extra_inputs)

    def attempt() -> Dict[str, Any]:
        try:
            res = crew.kickoff(inputs=inputs)

            # Different possible shapes:
            if isinstance(res, str):
                return json.loads(res)

            raw = getattr(res, "raw", res)
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    print("[WARN] Crew returned non-JSON string; returning default.")
                    return default_result.copy()

            if isinstance(raw, dict):
                return raw

            return default_result.copy()
        except Exception as e:
            result = default_result.copy()
            result["error"] = str(e)
            return result

    # First try
    r1 = attempt()
    if "error" not in r1:
        return r1

    print(f"[WARN] First attempt failed: {r1['error']}. Retrying once...")

    # Second try
    r2 = attempt()
    if "error" not in r2:
        print("[OK] Retry succeeded.")
        return r2

    print(f"[WARN] Retry failed: {r2['error']}. Returning default result with error.")
    result = default_result.copy()
    result["error"] = r2["error"]
    return result


async def _run_crew_on_retry_async(
    crew: Crew,
    text: str,
    input_key: str = "input_text",
    default_result: Optional[Dict[str, Any]] = None,
    extra_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    SAFE + RETRY-ONCE Crew runner (generic) - Async version.

    Tries to use native async akickoff() if available, otherwise falls back to
    kickoff_async() (thread-based) or synchronous kickoff() in a thread pool.
    See: https://docs.crewai.com/en/learn/kickoff-async

    Args:
        crew: The Crew instance to run
        text: The input text to process
        input_key: The key to use in the inputs dict (default: "input_text")
        default_result: Default result dict if parsing fails (default: empty dict)
        extra_inputs: Optional extra key-value inputs merged into crew inputs

    Returns:
        Dict with parsed results. If both attempts fail, returns:
          {**default_result, "error": "..."}
    """
    if default_result is None:
        default_result = {}
    inputs = {input_key: text}
    if extra_inputs:
        inputs.update(extra_inputs)

    # Check if akickoff is available (newer CrewAI versions)
    has_akickoff = hasattr(crew, 'akickoff')
    has_kickoff_async = hasattr(crew, 'kickoff_async')
    
    # Log which method we're using (only once, not per attempt)
    if not hasattr(_run_crew_on_retry_async, '_method_logged'):
        if has_akickoff:
            logger.debug("Using native async akickoff() for best performance")
        elif has_kickoff_async:
            logger.debug("Using thread-based kickoff_async() (akickoff not available)")
        else:
            logger.debug("Using sync kickoff() in thread pool (async methods not available)")
        _run_crew_on_retry_async._method_logged = True

    async def attempt() -> Dict[str, Any]:
        try:
            # Try native async akickoff() first (best performance)
            if has_akickoff:
                res = await crew.akickoff(inputs=inputs)
            # Fall back to thread-based async wrapper
            elif has_kickoff_async:
                res = await crew.kickoff_async(inputs=inputs)
            # Last resort: run sync kickoff in thread pool
            else:
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(None, lambda: crew.kickoff(inputs=inputs))

            # Different possible shapes:
            if isinstance(res, str):
                return json.loads(res)

            raw = getattr(res, "raw", res)
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    print("[WARN] Crew returned non-JSON string; returning default.")
                    return default_result.copy()

            if isinstance(raw, dict):
                return raw

            return default_result.copy()
        except Exception as e:
            result = default_result.copy()
            result["error"] = str(e)
            return result

    # First try
    r1 = await attempt()
    if "error" not in r1:
        return r1

    print(f"[WARN] First attempt failed: {r1['error']}. Retrying once...")

    # Second try
    r2 = await attempt()
    if "error" not in r2:
        print("[OK] Retry succeeded.")
        return r2

    print(f"[WARN] Retry failed: {r2['error']}. Returning default result with error.")
    result = default_result.copy()
    result["error"] = r2["error"]
    return result


# ============================================================
# GENERIC CREW RUNNER
# ============================================================
def run_crew_extraction(
    crew: Crew,
    text: str,
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    input_key: str = "input_text",
    default_result: Optional[Dict[str, Any]] = None,
    post_process: Optional[Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
    extra_inputs: Optional[Dict[str, Any]] = None,
    max_chunk_chars: Optional[int] = None,
    agent_context: Optional[AgentContext] = None,
    shared_memory: Optional[ThreadSafeMemory] = None,
    context_manager: Optional[ContextWindowManager] = None,
) -> Dict[str, Any]:
    """
    Generic robust crew extraction pipeline with enhanced context management.

    - If chunk_size is None or text is shorter than chunk_size:
        * Run once on full text.
    - Else:
        * Chunk text into sentence-aligned slices up to chunk_size chars.
        * Run extraction separately on each chunk (optionally in parallel).
        * Retry once per chunk if crew fails.
        * Optionally apply post-processing to results.

    This function NEVER raises due to crew/task failures.
    It returns best-effort results plus an 'errors' list.

    Args:
        crew: The Crew instance to run
        text: The input text to process
        chunk_size: Maximum chunk size in characters (None = no chunking)
        max_workers: Maximum parallel workers (None = auto)
        input_key: Key to use in crew inputs dict (default: "input_text")
        default_result: Default result structure if parsing fails
        post_process: Optional function to post-process results.
                     Signature: (full_text, full_doc, chunk, raw_result) -> processed_result
        extra_inputs: Optional extra key-value inputs merged into crew inputs (e.g. modification_context)
        max_chunk_chars: Optional cap on chunk_size (chars) so chunk + prompt stays under model context.
                        E.g. 25000 for 128k-token models. None = no cap.
        agent_context: Optional AgentContext for multi-agent context passing
        shared_memory: Optional ThreadSafeMemory for parallel execution
        context_manager: Optional ContextWindowManager for token limit management

    Returns:
        {
          "results": [...],  # List of processed results from all chunks
          "raw_results": [...],  # List of raw results before post-processing
          "errors": [
            {"scope": "chunk" | "full", "index": int | None, "error": str}, ...
          ]
        }
    """
    full_text = text
    errors: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    all_raw_results: List[Dict[str, Any]] = []

    if default_result is None:
        default_result = {}

    # Get timing logger if available
    timing_logger = logging.getLogger("timing")
    has_timing_logger = bool(timing_logger.handlers)
    
    # ---------------- NO CHUNKING PATH ----------------
    if not chunk_size or len(full_text) <= chunk_size:
        exec_start = time.time()
        raw = _run_crew_on_retry(crew, full_text, input_key, default_result, extra_inputs)
        exec_time = time.time() - exec_start
        if has_timing_logger:
            timing_logger.info(f"  Single execution (no chunking): {exec_time:.3f}s")

        if "error" in raw:
            errors.append(
                {"scope": "full", "index": None, "error": raw["error"]}
            )

        all_raw_results.append(raw)

        # Only add to all_results when there was no error, so merge uses only good results
        if "error" not in raw:
            if post_process:
                try:
                    nlp = get_spacy_model()
                    full_doc = nlp(full_text)
                    processed = post_process(full_text, full_doc, {"text": full_text, "start": 0}, raw)
                    all_results.append(processed)
                except Exception as e:
                    errors.append({
                        "scope": "full",
                        "index": None,
                        "error": f"Post-processing failed: {e}"
                    })
                    all_results.append(raw)
            else:
                all_results.append(raw)

        return {
            "results": all_results,
            "raw_results": all_raw_results,
            "errors": errors,
        }

    # ---------------- CHUNKED + (OPTIONAL) PARALLEL PATH ----------------
    # Cap chunk_size by max_chunk_chars so (chunk + prompt) stays under model context (token-safe)
    effective_chunk_size = chunk_size
    if max_chunk_chars is not None and (effective_chunk_size is None or effective_chunk_size > max_chunk_chars):
        effective_chunk_size = max_chunk_chars
        if chunk_size and chunk_size > max_chunk_chars and has_timing_logger:
            timing_logger.info(f"  Chunk size capped to {max_chunk_chars} chars (max_chunk_chars) for model context limit")
    chunking_start = time.time()
    nlp = get_spacy_model()
    full_doc = nlp(full_text)
    chunks = _chunk_doc_by_sentences(full_doc, max_chars=effective_chunk_size)
    chunking_time = time.time() - chunking_start
    if has_timing_logger:
        timing_logger.info(f"  Chunking: {chunking_time:.3f}s ({len(chunks)} chunks created)")

    if max_workers is None:
        max_workers = min(8, len(chunks))
    
    logger.info(f"Processing {len(chunks)} chunks with {max_workers} parallel workers")
    if has_timing_logger:
        timing_logger.info(f"  Parallel workers: {max_workers}")
    
    # Extract agent and task from the crew for creating new instances per thread
    # This is necessary because Crew instances are not thread-safe
    agents = crew.agents if hasattr(crew, 'agents') else []
    tasks = crew.tasks if hasattr(crew, 'tasks') else []
    memory_config = {
        'long_term_memory_config': getattr(crew, 'long_term_memory_config', None),
        'short_term_memory': getattr(crew, 'short_term_memory', None),
        'entity_memory': getattr(crew, 'entity_memory', None),
    }
    # Disable memory for parallel chunking to avoid SQLite database locking issues
    # Multiple Crew instances writing to the same database causes "database is locked" errors
    original_has_memory = any(memory_config.values())
    has_memory = False  # Disabled for parallel chunking
    if original_has_memory:
        logger.warning("Memory disabled for parallel chunking to avoid SQLite database locking issues. Memory will be available for non-chunked execution.")
    crew_process = crew.process if hasattr(crew, 'process') else Process.sequential
    crew_verbose = crew.verbose if hasattr(crew, 'verbose') else False

    def _copy_agents_tasks_for_chunk():
        """Return a copy of agents and tasks so each chunk's Crew has isolated state.
        Prevents context/token accumulation when the same Task/Agent objects are mutated during kickoff.

        NOTE: Recreates agents/tasks from scratch instead of deep-copying to avoid
        "cannot pickle '_thread.RLock' object" error with CrewAI's internal locks.
        """
        try:
            # Try deep copy first (faster if it works)
            return copy.deepcopy(agents), copy.deepcopy(tasks)
        except Exception as e:
            # Deep copy failed (likely due to thread locks)
            # Recreate agents/tasks from config instead
            logger.debug(
                f"Deep-copy failed ({e}). Recreating agents/tasks from config for chunk isolation."
            )

            # Recreate agents - get config from existing agent
            new_agents = []
            for agent in agents:
                try:
                    # Extract agent configuration
                    new_agent = Agent(
                        role=agent.role,
                        goal=agent.goal,
                        backstory=agent.backstory,
                        llm=agent.llm,  # LLM objects are usually safe to share
                        tools=agent.tools.copy() if hasattr(agent.tools, 'copy') else agent.tools,
                        allow_delegation=agent.allow_delegation if hasattr(agent, 'allow_delegation') else False,
                        verbose=agent.verbose if hasattr(agent, 'verbose') else False,
                    )
                    new_agents.append(new_agent)
                except Exception as agent_error:
                    logger.warning(f"Failed to recreate agent: {agent_error}. Using original.")
                    new_agents.append(agent)

            # Recreate tasks - get config from existing task
            new_tasks = []
            for i, task in enumerate(tasks):
                try:
                    # Extract task configuration
                    new_task = Task(
                        description=task.description,
                        expected_output=task.expected_output,
                        agent=new_agents[i] if i < len(new_agents) else task.agent,
                        output_pydantic=task.output_pydantic if hasattr(task, 'output_pydantic') else None,
                    )
                    new_tasks.append(new_task)
                except Exception as task_error:
                    logger.warning(f"Failed to recreate task: {task_error}. Using original.")
                    new_tasks.append(task)

            return new_agents, new_tasks

    def create_crew_for_chunk():
        """Create a new Crew instance for thread safety.
        
        Note: verbose is disabled for parallel execution to reduce overhead
        from CrewAI's flow management system.
        Memory is disabled for parallel chunking to avoid SQLite database locking.
        Each chunk gets its own copy of agents/tasks to avoid context accumulation (token limit errors).
        """
        crew_start = time.time()
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        chunk_agents, chunk_tasks = _copy_agents_tasks_for_chunk()
        crew = Crew(
            agents=chunk_agents,
            tasks=chunk_tasks,
            process=crew_process,
            tracing=False,
            verbose=False,  # Disable verbose for parallel execution to reduce overhead
            memory=False,  # Disabled for parallel chunking to avoid database locking
            long_term_memory_config=None,
            short_term_memory=None,
            entity_memory=None,
        )
        if has_timing_logger:
            crew_creation_time = time.time() - crew_start
            if crew_creation_time > 0.01:  # Only log if it takes significant time
                timing_logger.debug(f"    Crew creation took: {crew_creation_time:.3f}s")
        return crew

    def process_chunk_with_timing(chunk_data):
        """Process a single chunk with detailed timing."""
        idx, ch = chunk_data
        chunk_start = time.time()
        
        # Create crew for this chunk
        crew_creation_start = time.time()
        crew = create_crew_for_chunk()
        crew_creation_time = time.time() - crew_creation_start
        
        # Run extraction
        exec_start = time.time()
        raw = _run_crew_on_retry(crew, ch["text"], input_key, default_result, extra_inputs)
        exec_time = time.time() - exec_start
        
        chunk_time = time.time() - chunk_start
        
        return {
            "idx": idx,
            "chunk": ch,
            "raw": raw,
            "timing": {
                "total": chunk_time,
                "crew_creation": crew_creation_time,
                "execution": exec_time,
            }
        }

    parallel_start = time.time()
    chunk_times = {}  # Track individual chunk times
    crew_creation_times = []
    execution_times = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_chunk_with_timing, (idx, ch)): idx
            for idx, ch in enumerate(chunks)
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            
            try:
                chunk_result = fut.result()
                idx = chunk_result["idx"]
                chunk = chunk_result["chunk"]
                raw = chunk_result["raw"]
                timing = chunk_result["timing"]
                
                chunk_times[idx] = timing["total"]
                crew_creation_times.append(timing["crew_creation"])
                execution_times.append(timing["execution"])

                if "error" in raw:
                    errors.append({"scope": "chunk", "index": idx, "error": raw["error"]})

                all_raw_results.append(raw)

                # Only add to all_results when there was no error, so merge keeps "last good" from successful chunks
                post_start = time.time()
                if "error" not in raw:
                    if post_process:
                        try:
                            processed = post_process(full_text, full_doc, chunk, raw)
                            all_results.append(processed)
                        except Exception as e:
                            errors.append({
                                "scope": "chunk",
                                "index": idx,
                                "error": f"Post-processing failed: {e}"
                            })
                            all_results.append(raw)
                    else:
                        all_results.append(raw)
                post_time = time.time() - post_start
                
                if has_timing_logger and (idx < 5 or idx % 10 == 0):  # Log first 5 and every 10th chunk
                    timing_logger.info(f"    Chunk {idx}: {timing['total']:.3f}s (crew: {timing['crew_creation']:.3f}s, exec: {timing['execution']:.3f}s, post: {post_time:.3f}s)")
            except Exception as e:
                # Extreme safety: if even result() fails, record error and skip
                msg = f"Unexpected future error: {e}"
                print(f"[WARN] {msg}")
                errors.append({"scope": "chunk", "index": idx, "error": msg})
                continue

    parallel_time = time.time() - parallel_start
    if has_timing_logger:
        avg_chunk_time = sum(chunk_times.values()) / len(chunk_times) if chunk_times else 0
        max_chunk_time = max(chunk_times.values()) if chunk_times else 0
        min_chunk_time = min(chunk_times.values()) if chunk_times else 0
        avg_crew_creation = sum(crew_creation_times) / len(crew_creation_times) if crew_creation_times else 0
        avg_execution = sum(execution_times) / len(execution_times) if execution_times else 0
        total_crew_creation = sum(crew_creation_times)
        total_execution = sum(execution_times)
        
        timing_logger.info(f"  Parallel execution: {parallel_time:.3f}s")
        timing_logger.info(f"    - Total crew creation time: {total_crew_creation:.3f}s (avg: {avg_crew_creation:.3f}s per chunk)")
        timing_logger.info(f"    - Total execution time: {total_execution:.3f}s (avg: {avg_execution:.3f}s per chunk)")
        timing_logger.info(f"    - Average chunk time: {avg_chunk_time:.3f}s")
        timing_logger.info(f"    - Min chunk time: {min_chunk_time:.3f}s")
        timing_logger.info(f"    - Max chunk time: {max_chunk_time:.3f}s")
        timing_logger.info(f"    - Throughput: {len(chunks)/parallel_time:.2f} chunks/sec")
        timing_logger.info(f"    - Efficiency: {(total_execution/parallel_time)*100:.1f}% (execution time / wall time)")

    return {
        "results": all_results,
        "raw_results": all_raw_results,
        "errors": errors,
    }


async def run_crew_extraction_async(
    crew: Crew,
    text: str,
    chunk_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    input_key: str = "input_text",
    default_result: Optional[Dict[str, Any]] = None,
    post_process: Optional[Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
    extra_inputs: Optional[Dict[str, Any]] = None,
    max_chunk_chars: Optional[int] = None,
    agent_context: Optional[AgentContext] = None,
    shared_memory: Optional[ThreadSafeMemory] = None,
    context_manager: Optional[ContextWindowManager] = None,
) -> Dict[str, Any]:
    """
    Async version of run_crew_extraction using native async execution.
    
    Uses CrewAI's akickoff() for better concurrency with I/O-bound workloads.
    Recommended for high-concurrency scenarios. See: https://docs.crewai.com/en/learn/kickoff-async
    
    This function NEVER raises due to crew/task failures.
    It returns best-effort results plus an 'errors' list.

    Args:
        crew: The Crew instance to run
        text: The input text to process
        chunk_size: Maximum chunk size in characters (None = no chunking)
        max_workers: Maximum concurrent tasks (None = auto, but async can handle more)
        input_key: Key to use in crew inputs dict (default: "input_text")
        default_result: Default result structure if parsing fails
        post_process: Optional function to post-process results.
                     Signature: (full_text, full_doc, chunk, raw_result) -> processed_result
        extra_inputs: Optional extra key-value inputs merged into crew inputs
        max_chunk_chars: Optional cap on chunk_size (chars) so chunk + prompt stays under model context.
                        E.g. 25000 for 128k-token models. None = no cap.

    Returns:
        {
          "results": [...],  # List of processed results from all chunks
          "raw_results": [...],  # List of raw results before post-processing
          "errors": [
            {"scope": "chunk" | "full", "index": int | None, "error": str}, ...
          ]
        }
    """
    full_text = text
    errors: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    all_raw_results: List[Dict[str, Any]] = []

    if default_result is None:
        default_result = {}

    # Get timing logger if available
    timing_logger = logging.getLogger("timing")
    has_timing_logger = bool(timing_logger.handlers)
    
    # ---------------- NO CHUNKING PATH ----------------
    if not chunk_size or len(full_text) <= chunk_size:
        exec_start = time.time()
        raw = await _run_crew_on_retry_async(crew, full_text, input_key, default_result, extra_inputs)
        exec_time = time.time() - exec_start
        if has_timing_logger:
            timing_logger.info(f"  Single execution (no chunking, async): {exec_time:.3f}s")

        if "error" in raw:
            errors.append(
                {"scope": "full", "index": None, "error": raw["error"]}
            )

        all_raw_results.append(raw)

        # Only add to all_results when there was no error, so merge uses only good results
        if "error" not in raw:
            if post_process:
                try:
                    nlp = get_spacy_model()
                    full_doc = nlp(full_text)
                    processed = post_process(full_text, full_doc, {"text": full_text, "start": 0}, raw)
                    all_results.append(processed)
                except Exception as e:
                    errors.append({
                        "scope": "full",
                        "index": None,
                        "error": f"Post-processing failed: {e}"
                    })
                    all_results.append(raw)
            else:
                all_results.append(raw)

        return {
            "results": all_results,
            "raw_results": all_raw_results,
            "errors": errors,
        }

    # ---------------- CHUNKED + ASYNC PARALLEL PATH ----------------
    # Cap chunk_size by max_chunk_chars so (chunk + prompt) stays under model context (token-safe)
    effective_chunk_size = chunk_size
    if max_chunk_chars is not None and (effective_chunk_size is None or effective_chunk_size > max_chunk_chars):
        effective_chunk_size = max_chunk_chars
        if chunk_size and chunk_size > max_chunk_chars and has_timing_logger:
            timing_logger.info(f"  Chunk size capped to {max_chunk_chars} chars (max_chunk_chars) for model context limit")
    chunking_start = time.time()
    nlp = get_spacy_model()
    full_doc = nlp(full_text)
    chunks = _chunk_doc_by_sentences(full_doc, max_chars=effective_chunk_size)
    chunking_time = time.time() - chunking_start
    if has_timing_logger:
        timing_logger.info(f"  Chunking: {chunking_time:.3f}s ({len(chunks)} chunks created)")
        timing_logger.info(f"  Note: Creating {len(chunks)} Crew instances (one per chunk) with isolated agents/tasks")

    # Extract agent and task from the crew for creating new instances per async task
    agents = crew.agents if hasattr(crew, 'agents') else []
    tasks = crew.tasks if hasattr(crew, 'tasks') else []
    memory_config = {
        'long_term_memory_config': getattr(crew, 'long_term_memory_config', None),
        'short_term_memory': getattr(crew, 'short_term_memory', None),
        'entity_memory': getattr(crew, 'entity_memory', None),
    }
    # Disable memory for parallel chunking to avoid SQLite database locking issues
    # Multiple Crew instances writing to the same database causes "database is locked" errors
    original_has_memory = any(memory_config.values())
    has_memory = False  # Disabled for parallel chunking
    if original_has_memory:
        logger.warning("Memory disabled for parallel chunking to avoid SQLite database locking issues. Memory will be available for non-chunked execution.")
    crew_process = crew.process if hasattr(crew, 'process') else Process.sequential

    def _copy_agents_tasks_async():
        """Return a copy of agents and tasks so each chunk's Crew has isolated state.

        NOTE: Recreates agents/tasks from scratch instead of deep-copying to avoid
        "cannot pickle '_thread.RLock' object" error with CrewAI's internal locks.
        """
        try:
            # Try deep copy first (faster if it works)
            return copy.deepcopy(agents), copy.deepcopy(tasks)
        except Exception as e:
            # Deep copy failed (likely due to thread locks)
            # Recreate agents/tasks from config instead
            logger.debug(
                f"Deep-copy failed ({e}). Recreating agents/tasks from config for chunk isolation."
            )

            # Recreate agents
            new_agents = []
            for agent in agents:
                try:
                    new_agent = Agent(
                        role=agent.role,
                        goal=agent.goal,
                        backstory=agent.backstory,
                        llm=agent.llm,
                        tools=agent.tools.copy() if hasattr(agent.tools, 'copy') else agent.tools,
                        allow_delegation=agent.allow_delegation if hasattr(agent, 'allow_delegation') else False,
                        verbose=agent.verbose if hasattr(agent, 'verbose') else False,
                    )
                    new_agents.append(new_agent)
                except Exception as agent_error:
                    logger.warning(f"Failed to recreate agent: {agent_error}. Using original.")
                    new_agents.append(agent)

            # Recreate tasks
            new_tasks = []
            for i, task in enumerate(tasks):
                try:
                    new_task = Task(
                        description=task.description,
                        expected_output=task.expected_output,
                        agent=new_agents[i] if i < len(new_agents) else task.agent,
                        output_pydantic=task.output_pydantic if hasattr(task, 'output_pydantic') else None,
                    )
                    new_tasks.append(new_task)
                except Exception as task_error:
                    logger.warning(f"Failed to recreate task: {task_error}. Using original.")
                    new_tasks.append(task)

            return new_agents, new_tasks

    async def process_chunk_async(chunk_data):
        """Process a single chunk with async execution and detailed timing.
        
        Note: We create a new Crew instance per chunk for thread safety.
        Each chunk gets its own copy of agents/tasks to avoid context accumulation (token limit errors).
        Memory is disabled for parallel chunking to avoid SQLite database locking.
        """
        idx, ch = chunk_data
        chunk_start = time.time()
        
        # Create crew for this chunk with isolated agents/tasks (avoids token accumulation)
        crew_creation_start = time.time()
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        chunk_agents, chunk_tasks = _copy_agents_tasks_async()
        chunk_crew = Crew(
            agents=chunk_agents,
            tasks=chunk_tasks,
            process=crew_process,
            tracing=False,  # Disable tracing to reduce log noise
            verbose=False,  # Disable verbose for parallel execution
            memory=False,  # Disabled for parallel chunking to avoid database locking
            long_term_memory_config=None,
            short_term_memory=None,
            entity_memory=None,
        )
        crew_creation_time = time.time() - crew_creation_start
        
        # Run extraction with async
        exec_start = time.time()
        raw = await _run_crew_on_retry_async(chunk_crew, ch["text"], input_key, default_result, extra_inputs)
        exec_time = time.time() - exec_start
        
        chunk_time = time.time() - chunk_start
        
        return {
            "idx": idx,
            "chunk": ch,
            "raw": raw,
            "timing": {
                "total": chunk_time,
                "crew_creation": crew_creation_time,
                "execution": exec_time,
            }
        }

    parallel_start = time.time()
    chunk_times = {}
    crew_creation_times = []
    execution_times = []
    
    # Use asyncio.gather for native async concurrency
    # This is more efficient than ThreadPoolExecutor for I/O-bound workloads
    # Note: We create one Crew per chunk (required for thread safety)
    # but agents/tasks are reused, so the overhead is minimal
    chunk_tasks = [process_chunk_async((idx, ch)) for idx, ch in enumerate(chunks)]
    
    # Limit concurrency if max_workers is specified
    # For async, we can handle more concurrent tasks than threads, but still respect limits
    if max_workers and max_workers < len(chunks):
        # Process in batches to respect max_workers limit
        results_list = []
        for i in range(0, len(chunk_tasks), max_workers):
            batch = chunk_tasks[i:i + max_workers]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results_list.extend(batch_results)
        chunk_results = results_list
    else:
        # Process all chunks concurrently (async can handle many more than threads)
        # Default to reasonable limit if no max_workers specified
        effective_max = max_workers if max_workers else min(50, len(chunks))  # Cap at 50 for safety
        if effective_max < len(chunks):
            # Process in batches
            results_list = []
            for i in range(0, len(chunk_tasks), effective_max):
                batch = chunk_tasks[i:i + effective_max]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results_list.extend(batch_results)
            chunk_results = results_list
        else:
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Process results
    for result in chunk_results:
        if isinstance(result, Exception):
            msg = f"Unexpected async error: {result}"
            print(f"[WARN] {msg}")
            errors.append({"scope": "chunk", "index": None, "error": msg})
            continue
            
        idx = result["idx"]
        chunk = result["chunk"]
        raw = result["raw"]
        timing = result["timing"]
        
        chunk_times[idx] = timing["total"]
        crew_creation_times.append(timing["crew_creation"])
        execution_times.append(timing["execution"])

        if "error" in raw:
            errors.append({"scope": "chunk", "index": idx, "error": raw["error"]})

        all_raw_results.append(raw)

        # Only add to all_results when there was no error, so merge keeps "last good" from successful chunks
        post_start = time.time()
        if "error" not in raw:
            if post_process:
                try:
                    processed = post_process(full_text, full_doc, chunk, raw)
                    all_results.append(processed)
                except Exception as e:
                    errors.append({
                        "scope": "chunk",
                        "index": idx,
                        "error": f"Post-processing failed: {e}"
                    })
                    all_results.append(raw)
            else:
                all_results.append(raw)
        post_time = time.time() - post_start
        
        if has_timing_logger and (idx < 5 or idx % 10 == 0):
            timing_logger.info(f"    Chunk {idx}: {timing['total']:.3f}s (crew: {timing['crew_creation']:.3f}s, exec: {timing['execution']:.3f}s, post: {post_time:.3f}s)")

    parallel_time = time.time() - parallel_start
    if has_timing_logger:
        avg_chunk_time = sum(chunk_times.values()) / len(chunk_times) if chunk_times else 0
        max_chunk_time = max(chunk_times.values()) if chunk_times else 0
        min_chunk_time = min(chunk_times.values()) if chunk_times else 0
        avg_crew_creation = sum(crew_creation_times) / len(crew_creation_times) if crew_creation_times else 0
        avg_execution = sum(execution_times) / len(execution_times) if execution_times else 0
        total_crew_creation = sum(crew_creation_times)
        total_execution = sum(execution_times)
        
        timing_logger.info(f"  Async parallel execution: {parallel_time:.3f}s")
        timing_logger.info(f"    - Total crew creation time: {total_crew_creation:.3f}s (avg: {avg_crew_creation:.3f}s per chunk)")
        timing_logger.info(f"    - Total execution time: {total_execution:.3f}s (avg: {avg_execution:.3f}s per chunk)")
        timing_logger.info(f"    - Average chunk time: {avg_chunk_time:.3f}s")
        timing_logger.info(f"    - Min chunk time: {min_chunk_time:.3f}s")
        timing_logger.info(f"    - Max chunk time: {max_chunk_time:.3f}s")
        timing_logger.info(f"    - Throughput: {len(chunks)/parallel_time:.2f} chunks/sec")
        timing_logger.info(f"    - Efficiency: {(total_execution/parallel_time)*100:.1f}% (execution time / wall time)")

    return {
        "results": all_results,
        "raw_results": all_raw_results,
        "errors": errors,
    }


# ============================================================
# AGENT & TASK INITIALIZATION
# ============================================================
def initialize_agent_and_task(
    agent_config: Dict[str, Any],
    task_config: Dict[str, Any],
    agent_key: str,
    task_key: str,
    embedder_config: Optional[Dict[str, Any]] = None,
    tools: List = None,
    pydantic_output: Optional[Any] = None,
) -> Tuple[Optional[Agent], Optional[Task]]:
    """Initialize an agent and its associated task from configuration dictionaries.

    Args:
        agent_config: Dictionary containing agent configurations (keyed by agent_key)
        task_config: Dictionary containing task configurations (keyed by task_key)
        agent_key: Key for the agent in agent_config dictionary
        task_key: Key for the task in task_config dictionary
        embedder_config: Optional embedder configuration
        tools: Optional list of tools for the agent
        pydantic_output: Optional Pydantic class for structured output

    Returns:
        Tuple containing the initialized (agent, task), or (None, None) if initialization fails
    """
    try:
        if agent_key not in agent_config:
            print(f"[ERROR] Agent key '{agent_key}' not found in agent_config")
            return None, None
        
        if task_key not in task_config:
            print(f"[ERROR] Task key '{task_key}' not found in task_config")
            return None, None

        # Match the usage pattern from app.py - pass dict directly
        agent_init = DynamicAgent(
            agents_config=agent_config[agent_key],  # Pass dict directly (type hint may be incorrect)
            embedder_config=embedder_config or {},
            tools=tools or [],
        )
        
        task_init = DynamicAgentTask(
            tasks_config=task_config[task_key]
        )

        # Build agent - structsense build_agent() returns a single Agent
        # but it accesses self.agents_config directly, so we need to set it properly
        # The implementation seems to access it as a dict, so we'll work around it
        agent = agent_init.build_agent()

        if not agent:
            print(f"[ERROR] Failed to build agent for {agent_key}")
            return None, None

        task = task_init.build_task(agent=agent, pydantic_output=pydantic_output)

        if not task:
            print(f"[ERROR] {agent_key}/{task_key} initialization failed")
            return None, None

        print(f"[INFO] Successfully initialized {agent_key} and {task_key}")
        return agent, task

    except Exception as e:
        print(f"[ERROR] {agent_key}/{task_key} initialization failed: {e}")
        return None, None


# ============================================================
# MEMORY INITIALIZATION
# ============================================================
def initialize_memory(
    embedder_config: Dict[str, Any],
    memory_path: Optional[Path] = None,
) -> Tuple[Optional[LongTermMemory], Optional[ShortTermMemory], Optional[EntityMemory]]:
    """
    Initialize memory storage systems for CrewAI.
    
    Args:
        embedder_config: Embedder configuration dictionary
        memory_path: Optional path for memory storage (default: ./crew_memory)
        
    Returns:
        Tuple of (long_term_memory, short_term_memory, entity_memory)
        Each can be None if initialization fails
    """
    try:
        if memory_path is None:
            memory_path = Path(os.getcwd()) / "crew_memory"
        
        os.environ["CREWAI_STORAGE_DIR"] = str(memory_path)
        storage_path = db_storage_path()

        # Debug storage path
        logger.info(f"Storage path: {storage_path}")
        logger.info(f"Path exists: {os.path.exists(storage_path)}")
        logger.info(
            f"Is writable: {os.access(storage_path, os.W_OK) if os.path.exists(storage_path) else 'Path does not exist'}"
        )

        # Create with proper permissions
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, mode=0o755, exist_ok=True)
            logger.info(f"Created storage directory: {storage_path}")

        # Check if embedder config is compatible
        # Handle different config formats:
        # 1. {provider: "ollama", config: {api_base: ..., model: ...}}
        # 2. {embedder_config: {provider: ..., config: ...}}
        # 3. Direct embedder config dict
        embedder_config_for_rag = None
        if isinstance(embedder_config, dict):
            if 'provider' in embedder_config:
                # Format: {provider: "ollama", config: {...}}
                embedder_config_for_rag = embedder_config
            elif 'embedder_config' in embedder_config:
                # Nested format: {embedder_config: {provider: ..., config: ...}}
                embedder_config_for_rag = embedder_config['embedder_config']
            else:
                # Direct config dict
                embedder_config_for_rag = embedder_config
        else:
            embedder_config_for_rag = embedder_config  # fallback

        # Check if using Ollama embedder but not running Ollama
        if (isinstance(embedder_config_for_rag, dict) and
            embedder_config_for_rag.get('provider') == 'ollama' and
            not check_ollama_health()):
            logger.warning("Ollama embedder configured but Ollama not available. Disabling memory to prevent errors.")
            return None, None, None

        rag_storage_config = {
            "embedder_config": embedder_config_for_rag,
            "type": "short_term",
            "path": str(storage_path),
        }

        long_term_storage = f"{storage_path}/long_term_memory_storage.db"

        # Initialize memory components with error handling
        long_term_memory = None
        try:
            long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(db_path=str(long_term_storage))
            )
            logger.info("Long-term memory initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize long-term memory: {e}")
            long_term_memory = None

        short_term_memory = None
        try:
            short_term_memory = ShortTermMemory(
                storage=RAGStorage(**rag_storage_config)
            )
            logger.info("Short-term memory initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize short-term memory: {e}")
            short_term_memory = None

        entity_memory = None
        try:
            entity_memory = EntityMemory(
                storage=RAGStorage(**rag_storage_config)
            )
            logger.info("Entity memory initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize entity memory: {e}")
            entity_memory = None

        if all([long_term_memory, short_term_memory, entity_memory]):
            logger.info("All memory systems initialized successfully")
        else:
            logger.warning("Some memory systems failed to initialize - continuing without full memory support")

        return long_term_memory, short_term_memory, entity_memory

    except Exception as e:
        error_msg = f"Failed to initialize memory systems: {str(e)}"
        logger.error(error_msg)
        logger.info("Continuing without memory systems")
        return None, None, None