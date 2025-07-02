import concurrent.futures
import logging
import os
import time
import tracemalloc
import asyncio

# Filter warnings at the beginning
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

warnings.filterwarnings("ignore")
from crewai import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.paths import db_storage_path
from dotenv import load_dotenv

from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from utils.ontology_knowedge_tool import OntologyKnowledgeTool
from utils.text_chunking import split_text_into_chunks, merge_json_chunks  # merge_chunk_results
from utils.types import AlignedTermsDynamic, ExtractedTermsDynamic, JudgedTermsDynamic
from utils.utils import (
    has_modifications,
    load_config,
    process_input_data,
    replace_api_key,
)

from .humanloop import HumanInTheLoop, ProgrammaticFeedbackHandler

# Add new import for hardcoded configs
# from .default_config_sie_ner import get_agent_config, NER_TASK_CONFIG, EMBEDDER_CONFIG, HUMAN_IN_LOOP_CONFIG, SEARCH_ONTOLOGY_KNOWLEDGE_CONFIG

# Start memory tracking
tracemalloc.start()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s")
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class StructSenseFlow(Flow):
    """A workflow for structured information extraction, alignment, and judgment using CrewAI.
    Includes improved crew communication and shared memory.
    """

    def __init__(
        self,
        agent_config: str,
        task_config: str,
        embedder_config: str,
        source_text: str,
        knowledge_config: Optional[str] = None,
        enable_human_feedback: bool = False,
        agent_feedback_config: Dict[str, bool] = None,
        env_file: Optional[str] = None,
        enable_chunking: bool = False,
    ):
        super().__init__()
        logger.info(f"Initializing StructSenseFlow with source: {source_text}")
        self.source_text = source_text
        self.enable_human_feedback = enable_human_feedback
        self.human = HumanInTheLoop(enable_human_feedback=enable_human_feedback, agent_feedback_config=agent_feedback_config)
        try:
            self.agent_config = load_config(agent_config, "agent_config")
            self.task_config = load_config(task_config, "task_config")
            self.embedder_config = load_config(embedder_config, "embedder_config")
            if knowledge_config is None:
                os.environ["ENABLE_KG_SOURCE"] = "false"
                self.knowledge_config = {"search_key": {}}
            else:
                self.knowledge_config = load_config(knowledge_config, "knowledge_config")
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ConfigError(f"Failed to load configurations: {str(e)}")
        self._setup_monitoring()
        self._initialize_memory()
        self.shared_state = {
            "extracted_terms": None,
            "aligned_terms": None,
            "judged_terms": None,
            "feedback_terms": None,
            "current_step": None,
            "last_error": None,
        }
        self.enable_chunking = enable_chunking

    @start()
    def process_inputs(self):
        """Start processing the input data."""
        logger.info("Starting structured information processing flow")
        self._update_shared_state("process_inputs", self.source_text)

    @listen(process_inputs)
    async def extracted_structured_information(self):
        """Extract structured information from the source text."""
        start_time = time.time()
        logger.info("Starting structured information extraction")

        # Check Ollama health
        if not self._check_ollama_health():
            return None

        # Detect task type
        task_type = self._detect_task_type()
        logger.info(f"Detected task type: {task_type}")

        # Initialize extractor components
        extractor_agent, extractor_task = self._initialize_agent_and_task("extractor_agent", "extraction_task", ExtractedTermsDynamic)

        if not extractor_agent or not extractor_task:
            logger.error("Extractor initialization failed")
            return None

        if self.enable_chunking:
            logger.info("Chunking is enabled - processing text in chunks")
            return await self._extract_with_chunking(extractor_agent, extractor_task, start_time, task_type)
        else:
            logger.info("Chunking is disabled - processing text as single chunk")
            return await self._extract_without_chunking(extractor_agent, extractor_task, start_time, task_type)

    async def _extract_with_chunking(self, extractor_agent, extractor_task, start_time, task_type):
        """Extract information using chunking approach."""
        # Split text into chunks for parallel processing
        logger.info("Splitting text into chunks for parallel processing...")
        try:
            chunks = split_text_into_chunks(self.source_text)
            logger.info(f"Split text into {len(chunks)} chunks for parallel processing")
            logger.debug(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
        except Exception as e:
            logger.error(f"Failed to split text into chunks: {e}")
            # Fallback to single chunk
            chunks = [self.source_text]
            logger.info("Using fallback single chunk processing")

        if not chunks:
            logger.error("No chunks created from text splitting")
            return None

        chunk_results = []
        total_chunks = len(chunks)

        def process_chunk(chunk, idx):
            print("*" * 100)
            print(f"[Thread] Starting processing for chunk {idx+1}/{total_chunks} (size: {len(chunk)} chars)")
            print("#" * 100)
            logger.info(f"[Thread] Starting processing for chunk {idx+1}/{total_chunks} (size: {len(chunk)} chars)")
            chunk_start_time = time.time()
            try:
                inputs = {"literature": chunk}
                extractor_crew = self._create_crew_with_knowledge(extractor_agent, extractor_task)
                print("@" * 100)
                print("@" * 100)
                print(f"Extractor agent with chunk data {inputs}")
                print("@" * 100)
                print("@" * 100)
                chunk_result = extractor_crew.kickoff(inputs=inputs)
                elapsed = time.time() - chunk_start_time

                print("*" * 100)
                print(f"[Thread] Raw chunk result type: {type(chunk_result)}")
                print(f"[Thread] Raw chunk result: {chunk_result}")
                print("*" * 100)

                if chunk_result:
                    # Convert to dict and add debugging
                    result_dict = chunk_result.to_dict() if hasattr(chunk_result, "to_dict") else chunk_result
                    print("*" * 100)
                    print(f"[Thread] Result dict: {result_dict}")
                    print(f"[Thread] Result dict keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'Not a dict'}")
                    print("*" * 100)

                    # Check for expected keys and count items
                    expected_keys = ["terms", "extracted_terms", "extracted_resources", "extracted_structured_information"]
                    found_keys = []
                    total_items = 0

                    for key in expected_keys:
                        if isinstance(result_dict, dict) and key in result_dict:
                            found_keys.append(key)
                            items = result_dict[key]
                            item_count = len(items) if isinstance(items, list) else "non-list"
                            total_items += len(items) if isinstance(items, list) else 0
                            print(f"[Thread] Found key '{key}' with {item_count} items")

                    if not found_keys:
                        print(
                            f"[Thread] WARNING: No expected keys found in result. Available keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'Not a dict'}"
                        )
                        # Try to extract any list-like data
                        if isinstance(result_dict, dict):
                            for key, value in result_dict.items():
                                if isinstance(value, list) and len(value) > 0:
                                    print(f"[Thread] Found list data in key '{key}' with {len(value)} items")
                                    # Create a proper structure
                                    result_dict = {"terms": value}
                                    total_items = len(value)
                                    break

                    logger.info(f"[Thread] Finished chunk {idx+1}/{total_chunks} in {elapsed:.2f}s, got {total_items} total items")
                    print("*" * 100)
                    print(f"[Thread] Finished chunk {idx+1}/{total_chunks} in {elapsed:.2f}s, returning: {result_dict}")
                    print("#" * 100)
                    return result_dict
                else:
                    print("*" * 100)
                    print(f"[Thread] No result for chunk {idx + 1}/{total_chunks} (took {elapsed:.2f}s)")
                    print("#" * 100)
                    logger.warning(f"[Thread] No result for chunk {idx+1}/{total_chunks} (took {elapsed:.2f}s)")
                    return None
            except Exception as exc:
                logger.error(f"[Thread] Exception in chunk {idx+1}/{total_chunks}: {exc}")
                print(f"[Thread] Exception details: {exc}")
                import traceback

                print(f"[Thread] Traceback: {traceback.format_exc()}")
                return None

        logger.info(f"Processing {total_chunks} chunks in parallel using ThreadPoolExecutor...")
        print("*" * 100)
        print("#" * 100)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    if result:
                        chunk_results.append(result)
                        print("*" * 100)
                        print(chunk_results)
                        print("--" * 100)
                        print(f"[Main] Collected result {result} for chunk {i+1}/{total_chunks}")
                        print("*" * 100)
                        logger.info(f"[Main] Collected result for chunk {i+1}/{total_chunks}")
                    else:
                        logger.warning(f"[Main] No result for chunk {i+1}/{total_chunks}")
                except Exception as exc:
                    logger.error(f"[Main] Exception collecting chunk {i+1}/{total_chunks}: {exc}")

        print("*" * 100)
        print(f"All chunks processed. {(chunk_results)} out of {total_chunks} returned results.")
        logger.info(f"All chunks processed. {len(chunk_results)} out of {total_chunks} returned results.")
        print("$*" * 100)
        if not chunk_results:
            logger.warning("No results from any chunks")
            return None

        # Validate chunk results before merging
        valid_results = []
        total_items_before_merge = 0

        for i, result in enumerate(chunk_results):
            if result and isinstance(result, dict):
                valid_results.append(result)
                # Count items in this result
                items_in_result = 0
                for key, value in result.items():
                    if isinstance(value, list):
                        items_in_result += len(value)
                total_items_before_merge += items_in_result
                logger.info(f"Valid result from chunk {i+1}: {items_in_result} total items across all keys")
            else:
                logger.warning(f"Invalid result from chunk {i+1}: {type(result)}")

        if not valid_results:
            logger.error("No valid results from any chunks")
            return None

        logger.info(f"Total items before merging: {total_items_before_merge}")

        # Merge chunk results
        try:
            combined_result = merge_json_chunks(valid_results)
            logger.info(f"Successfully merged {len(valid_results)} chunk results")

            # Count items after merging
            total_items_after_merge = 0
            for key, value in combined_result.items():
                if isinstance(value, list):
                    total_items_after_merge += len(value)
            logger.info(f"Total items after merging: {total_items_after_merge}")

        except Exception as e:
            logger.error(f"Failed to merge chunk results: {e}")
            # Fallback: combine all terms manually
            all_terms = []
            for result in valid_results:
                for key, value in result.items():
                    if isinstance(value, list):
                        all_terms.extend(value)
            combined_result = {"terms": all_terms}
            logger.info(f"Used fallback merging, got {len(all_terms)} total terms")

        # Transform the data into the expected format for alignment
        logger.info(f"Combined result before transformation: {combined_result}")
        transformed_terms = self._transform_extracted_data(combined_result)
        logger.info(f"Transformed terms: {transformed_terms}")

        # Wrap data according to task type
        wrapped_result = self._wrap_data_for_next_step(transformed_terms, task_type, "extraction")
        logger.info(f"Wrapped result: {wrapped_result}")

        # Update shared state with transformed terms
        self._update_shared_state("extracted_terms", wrapped_result)

        total_time = time.time() - start_time
        logger.info(f"Extraction complete with {len(transformed_terms)} terms in {total_time:.2f} seconds")
        return wrapped_result

    async def _extract_without_chunking(self, extractor_agent, extractor_task, start_time, task_type):
        """Extract information without chunking - process entire text at once."""
        logger.info("Processing entire text without chunking")

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                inputs = {"literature": self.source_text}
                extractor_crew = self._create_crew_with_knowledge(extractor_agent, extractor_task)

                logger.info(f"Running extractor crew on full text (attempt {attempt + 1}/{max_retries})")
                extractor_result = extractor_crew.kickoff(inputs=inputs)

                if not extractor_result:
                    logger.warning("Extractor crew returned no results")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    return None

                result_dict = extractor_result.to_dict()
                logger.info(f"Extraction complete with {len(result_dict.get('terms', []))} terms")

                # Transform the data into the expected format for alignment
                logger.info(f"Result dict before transformation: {result_dict}")
                transformed_terms = self._transform_extracted_data(result_dict)
                logger.info(f"Transformed terms: {transformed_terms}")

                # Wrap data according to task type
                wrapped_result = self._wrap_data_for_next_step(transformed_terms, task_type, "extraction")

                # Update shared state with transformed terms
                self._update_shared_state("extracted_terms", wrapped_result)

                total_time = time.time() - start_time
                logger.info(f"Extraction complete with {len(transformed_terms)} terms in {total_time:.2f} seconds")
                return wrapped_result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error during non-chunked extraction (attempt {attempt + 1}/{max_retries}): {error_msg}")

                # Check if it's a timeout error
                if "timeout" in error_msg.lower() or "connection timed out" in error_msg.lower():
                    logger.warning("Timeout detected - this might be due to Ollama model being overloaded")
                    logger.info("💡 Tips to improve Ollama performance:")
                    logger.info("   1. Ensure you have enough RAM (at least 8GB free)")
                    logger.info("   2. Close other applications using GPU/CPU")
                    logger.info("   3. Try using a smaller model (e.g., llama3.1:8b instead of llama3.2:latest)")
                    logger.info("   4. Restart Ollama: 'ollama stop && ollama start'")
                    logger.info("   5. Check Ollama logs: 'ollama logs'")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay * (attempt + 1)} seconds...")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue

                # For other errors, don't retry immediately
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue

                logger.error(f"All {max_retries} attempts failed. Returning None.")
                return None

    def _check_ollama_health(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            import httpx
            import json

            # Check if any agent is using Ollama
            using_ollama = False
            for agent_key, agent_config in self.agent_config.items():
                llm_config = agent_config.get("llm", {})
                if isinstance(llm_config, dict) and "ollama" in llm_config.get("model", "").lower():
                    using_ollama = True
                    break

            if not using_ollama:
                return True  # Not using Ollama, so no need to check

            # Test Ollama connection
            with httpx.Client(timeout=10.0) as client:
                response = client.get("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
                if response.status_code == 200:
                    logger.info("Ollama is running and responsive")
                    return True
                else:
                    logger.warning(f"Ollama responded with status code: {response.status_code}")
                    return False

        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def _setup_monitoring(self) -> None:
        """Set up monitoring tools if enabled."""
        if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
            try:
                import weave

                weave.init(project_name="StructSense")
                logger.info("Weights & Biases monitoring enabled")
            except ImportError:
                logger.warning("Weights & Biases package not found, monitoring disabled")

        if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
            try:
                import mlflow

                mlflow.crewai.autolog()
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
                mlflow.set_experiment("StructSense")
                logger.info("MLflow monitoring enabled")
            except ImportError:
                logger.warning("MLflow package not found, monitoring disabled")

    def _initialize_memory(self) -> None:
        """Initialize memory storage systems for the flow."""
        try:
            memory_path = Path(os.getcwd()) / "crew_memory"
            os.environ["CREWAI_STORAGE_DIR"] = str(memory_path)
            storage_path = db_storage_path()

            # Debug storage path
            logger.info(f"Storage path: {storage_path}")
            logger.info(f"Path exists: {os.path.exists(storage_path)}")
            logger.info(f"Is writable: {os.access(storage_path, os.W_OK) if os.path.exists(storage_path) else 'Path does not exist'}")

            # Create with proper permissions
            if not os.path.exists(storage_path):
                os.makedirs(storage_path, mode=0o755, exist_ok=True)
                logger.info(f"Created storage directory: {storage_path}")

            # DEBUG: Print embedder_config before passing to RAGStorage
            print("DEBUG: embedder_config for RAGStorage:", self.embedder_config)

            # Check if embedder config is compatible
            embedder_config_for_rag = None
            if isinstance(self.embedder_config, dict) and "provider" in self.embedder_config:
                embedder_config_for_rag = self.embedder_config
            elif isinstance(self.embedder_config, dict) and "embedder_config" in self.embedder_config:
                embedder_config_for_rag = self.embedder_config["embedder_config"]
            else:
                embedder_config_for_rag = self.embedder_config  # fallback

            # Check if using Ollama embedder but not running Ollama
            if (
                isinstance(embedder_config_for_rag, dict)
                and embedder_config_for_rag.get("provider") == "ollama"
                and not self._check_ollama_health()
            ):
                logger.warning("Ollama embedder configured but Ollama not available. Disabling memory to prevent errors.")
                # Initialize with None to disable memory
                self.long_term_memory = None
                self.short_term_memory = None
                self.entity_memory = None
                return

            rag_storage_config = {
                "embedder_config": embedder_config_for_rag,
                "type": "short_term",
                "path": str(storage_path),
            }

            long_term_storage = f"{storage_path}/long_term_memory_storage.db"

            # Initialize memory components with error handling
            try:
                self.long_term_memory = LongTermMemory(storage=LTMSQLiteStorage(db_path=str(long_term_storage)))
                logger.info("Long-term memory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize long-term memory: {e}")
                self.long_term_memory = None

            try:
                self.short_term_memory = ShortTermMemory(storage=RAGStorage(**rag_storage_config))
                logger.info("Short-term memory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize short-term memory: {e}")
                self.short_term_memory = None

            try:
                self.entity_memory = EntityMemory(storage=RAGStorage(**rag_storage_config))
                logger.info("Entity memory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize entity memory: {e}")
                self.entity_memory = None

            if all([self.long_term_memory, self.short_term_memory, self.entity_memory]):
                logger.info("All memory systems initialized successfully")
            else:
                logger.warning("Some memory systems failed to initialize - continuing without full memory support")

        except Exception as e:
            error_msg = f"Failed to initialize memory systems: {str(e)}"
            logger.error(error_msg)
            # Don't raise error, just disable memory
            self.long_term_memory = None
            self.short_term_memory = None
            self.entity_memory = None
            logger.info("Continuing without memory systems")

    def _initialize_agent_and_task(self, agent_key: str, task_key: str, pydantic_output_class) -> Tuple[Optional[object], Optional[object]]:
        """Initialize an agent and its associated task.

        Args:
            agent_key: Key for the agent in agent configuration
            task_key: Key for the task in task configuration
            pydantic_output_class: Pydantic class for structured output

        Returns:
            Tuple containing the initialized agent and task
        """
        try:
            print(f"[DEBUG] agent_key: {agent_key}")
            print(f"[DEBUG] task_key: {task_key}")
            print(f"[DEBUG] agent_config for {agent_key}: {self.agent_config.get(agent_key)}")
            print(f"[DEBUG] task_config for {task_key}: {self.task_config.get(task_key)}")
            agent_init = DynamicAgent(agents_config=self.agent_config[agent_key], embedder_config=self.embedder_config)
            task_init = DynamicAgentTask(tasks_config=self.task_config[task_key])

            agent = agent_init.build_agent()
            task = task_init.build_task(pydantic_output=pydantic_output_class, agent=agent)

            if not task:
                logger.error(f"{task_key} initialization failed")
                return None, None

            logger.info(f"Successfully initialized {agent_key} and {task_key}")
            return agent, task

        except Exception as e:
            logger.error(f"{agent_key}/{task_key} initialization failed: {e}")
            return None, None

    def _should_enable_knowledge_source(self) -> bool:
        """Check if knowledge source should be enabled."""
        try:
            # Check if knowledge source is explicitly disabled
            if os.getenv("ENABLE_KG_SOURCE", "false").lower() == "false":
                logger.info("Knowledge source explicitly disabled via ENABLE_KG_SOURCE=false")
                return False

            # Check if knowledge config is properly configured
            if not self.knowledge_config or not isinstance(self.knowledge_config, dict):
                logger.info("No knowledge config provided, disabling knowledge source")
                return False

            if "search_key" not in self.knowledge_config:
                logger.info("No search_key in knowledge config, disabling knowledge source")
                return False

            search_keys = self.knowledge_config.get("search_key", [])
            if not search_keys or not isinstance(search_keys, list) or len(search_keys) == 0:
                logger.info("Empty or invalid search_key in knowledge config, disabling knowledge source")
                return False

            logger.info(f"Knowledge source enabled with search keys: {search_keys}")
            return True

        except Exception as e:
            logger.warning(f"Error checking knowledge source configuration: {e}")
            return False

    def _create_crew_with_knowledge(self, agent, task, data_for_knowledge_tool: Optional[Dict] = None) -> Crew:
        """Create a Crew instance with or without knowledge sources based on configuration.

        Args:
            agent: The agent to use in the crew
            task: The task to execute
            data_for_knowledge_tool: Data to provide to the knowledge tool if enabled

        Returns:
            Configured Crew instance
        """
        crew_config = {
            "agents": [agent],
            "tasks": [task],
            "verbose": True,
        }

        # Add memory only if available
        if self.long_term_memory is not None:
            crew_config["long_term_memory_config"] = self.long_term_memory
        if self.short_term_memory is not None:
            crew_config["short_term_memory"] = self.short_term_memory
        if self.entity_memory is not None:
            crew_config["entity_memory"] = self.entity_memory

        # Only enable memory if at least one memory system is available
        if any([self.long_term_memory, self.short_term_memory, self.entity_memory]):
            crew_config["memory"] = True
        else:
            crew_config["memory"] = False
            logger.info("No memory systems available - running without memory")

        # Add knowledge sources if enabled
        if self._should_enable_knowledge_source() and data_for_knowledge_tool:
            try:
                # Format the data for the OntologyKnowledgeTool
                formatted_data = {}
                if isinstance(data_for_knowledge_tool, dict):
                    for key, value in data_for_knowledge_tool.items():
                        if isinstance(value, list):
                            formatted_data[key] = value
                        elif isinstance(value, dict):
                            formatted_data[key] = [value]
                        else:
                            formatted_data[key] = [{"entity": str(value)}]
                else:
                    formatted_data = {"terms": [{"entity": str(data_for_knowledge_tool)}]}

                custom_source = OntologyKnowledgeTool(formatted_data, self.knowledge_config["search_key"])

                logger.debug("Knowledge source result:")
                logger.debug(custom_source)

                ksrc = StringKnowledgeSource(content=custom_source)
                crew_config["knowledge_sources"] = [ksrc]
                logger.info("Knowledge source added to crew configuration")

            except Exception as e:
                logger.warning(f"Failed to create knowledge source: {e}")
                logger.info("Continuing without knowledge source")
        else:
            logger.info("Knowledge source not enabled or no data provided")

        return Crew(**crew_config)

    def _update_shared_state(self, key: str, value: Any) -> None:
        """Update shared state and notify other crews"""
        self.shared_state[key] = value
        self.shared_state["current_step"] = key
        logger.info(f"Updated shared state: {key}")

    def _get_shared_state(self, key: str) -> Any:
        """Get value from shared state"""
        return self.shared_state.get(key)

    def _convert_sets_to_lists(self, data):
        """Convert any sets in the data structure to lists."""
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: self._convert_sets_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_sets_to_lists(item) for item in data]
        return data

    def _transform_extracted_data(self, combined_result):
        """Transform extracted data from chunk results into the expected format."""
        if not combined_result:
            logger.warning("No combined result to transform")
            return []

        logger.info(f"Transforming combined result of type: {type(combined_result)}")

        # Handle different possible structures
        if isinstance(combined_result, dict):
            # For BBQS tasks, look for extracted_structured_information first
            if "extracted_structured_information" in combined_result:
                extracted_info = combined_result["extracted_structured_information"]
                logger.info(f"Found 'extracted_structured_information' with type: {type(extracted_info)}")

                
                # Handle nested dictionary structure (like extracted_terms with numbered keys)

                if isinstance(extracted_info, dict):
                    terms = []
                    for key, value in extracted_info.items():
                        if isinstance(value, list):
                            terms.extend(value)
                            logger.debug(f"Added {len(value)} items from list key '{key}'")
                        elif isinstance(value, dict):
                            terms.append(value)
                            logger.debug(f"Added 1 item from dict key '{key}'")
                        else:
                            terms.append(value)
                            logger.debug(f"Added 1 item from key '{key}'")
                    logger.info(f"Extracted {len(terms)} entities from nested structure")
                elif isinstance(extracted_info, list):
                    terms = extracted_info
                    logger.info(f"Using list of {len(terms)} resources")
                else:
                    terms = [extracted_info]
                    logger.info("Converted non-dict/list to list")

                    
            elif 'extracted_terms' in combined_result:
                extracted_terms = combined_result['extracted_terms']
                logger.info(f"Found 'extracted_terms' key with type: {type(extracted_terms)}")
                
                # Handle nested dictionary structure (like numbered keys "1", "2", etc.)
                if isinstance(extracted_terms, dict):
                    terms = []
                    for key, value in extracted_terms.items():
                        if isinstance(value, list):
                            terms.extend(value)
                            logger.debug(f"Added {len(value)} items from list key '{key}'")
                        elif isinstance(value, dict):
                            terms.append(value)
                            logger.debug(f"Added 1 item from dict key '{key}'")
                        else:
                            terms.append(value)
                            logger.debug(f"Added 1 item from key '{key}'")
                    logger.info(f"Extracted {len(terms)} entities from extracted_terms structure")
                elif isinstance(extracted_terms, list):
                    terms = extracted_terms
                    logger.info(f"Using list of {len(terms)} extracted terms")
                else:
                    terms = [extracted_terms]
                    logger.info("Converted extracted_terms to list")
                    
            elif 'extracted_resources' in combined_result:
                extracted_resources = combined_result['extracted_resources']
                logger.info(f"Found 'extracted_resources' key with type: {type(extracted_resources)}")
                
                # Handle nested dictionary structure
                if isinstance(extracted_resources, dict):
                    terms = []
                    for key, value in extracted_resources.items():
                        if isinstance(value, list):
                            terms.extend(value)
                            logger.debug(f"Added {len(value)} items from list key '{key}'")
                        elif isinstance(value, dict):
                            terms.append(value)
                            logger.debug(f"Added 1 item from dict key '{key}'")
                        else:
                            terms.append(value)
                            logger.debug(f"Added 1 item from key '{key}'")
                    logger.info(f"Extracted {len(terms)} entities from extracted_resources structure")
                elif isinstance(extracted_resources, list):
                    terms = extracted_resources
                    logger.info(f"Using list of {len(terms)} extracted resources")
                else:
                    terms = [extracted_resources]
                    logger.info("Converted extracted_resources to list")

            else:
                # If no clear structure, try to extract terms from the dict
                terms = []
                for key, value in combined_result.items():
                    logger.debug(f"Processing key '{key}' with value type: {type(value)}")
                    if isinstance(value, list):
                        terms.extend(value)
                        logger.debug(f"Added {len(value)} items from list key '{key}'")
                    elif isinstance(value, dict) and "terms" in value:
                        terms.extend(value["terms"])
                        logger.debug(f"Added {len(value['terms'])} items from dict key '{key}'")
                    elif isinstance(value, dict) and "extracted_terms" in value:
                        terms.extend(value["extracted_terms"])
                        logger.debug(f"Added {len(value['extracted_terms'])} items from dict key '{key}'")
                    elif isinstance(value, dict) and "extracted_resources" in value:
                        terms.extend(value["extracted_resources"])
                        logger.debug(f"Added {len(value['extracted_resources'])} items from dict key '{key}'")
                logger.info(f"Extracted {len(terms)} terms from dictionary structure")
        elif isinstance(combined_result, list):
            terms = combined_result
            logger.info(f"Processing list structure with {len(terms)} items")
        else:
            logger.warning(f"Unexpected combined_result type: {type(combined_result)}")
            return []

        # Ensure terms is a list
        if not isinstance(terms, list):
            logger.warning(f"Terms is not a list: {type(terms)}, converting...")
            try:
                terms = list(terms) if hasattr(terms, "__iter__") else [terms]
            except Exception as e:
                logger.error(f"Failed to convert terms to list: {e}")
                return []

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for i, term in enumerate(terms):
            try:
                if isinstance(term, dict):
                    # Create a hashable representation for comparison
                    term_key = str(sorted(term.items()))
                    if term_key not in seen:
                        seen.add(term_key)
                        unique_terms.append(term)
                else:
                    if term not in seen:
                        seen.add(term)
                        unique_terms.append(term)
            except Exception as e:
                logger.warning(f"Error processing term {i}: {e}")
                continue

        logger.info(f"Transformed {len(terms)} terms to {len(unique_terms)} unique terms")
        return unique_terms

    @listen(extracted_structured_information)
    async def align_structured_information(self, extracted_result):
        """Align extracted information with ontology using CrewAI."""
        if not extracted_result:
            logger.warning("No structured information extracted. Skipping alignment.")
            return None

        print("#" * 100)
        print(extracted_result)

        logger.info("Starting alignment process")

        # Detect task type
        task_type = self._detect_task_type()
        logger.info(f"Detected task type for alignment: {task_type}")

        # Initialize alignment components
        alignment_agent, alignment_task = self._initialize_agent_and_task("alignment_agent", "alignment_task", AlignedTermsDynamic)

        print("-" * 100)
        print(f"alignment_agent {alignment_agent}")
        print("+" * 100)
        print(f"alignment_task {alignment_task}")
        print("#" * 100)

        if not alignment_agent or not alignment_task:
            logger.error("Alignment agent initialization failed")
            return None

        # Convert any sets to lists in the extracted result
        processed_result = self._convert_sets_to_lists(extracted_result)

        # Extract the actual extracted data from the wrapper
        actual_extracted_data = self._extract_data_from_result(processed_result, task_type, "extraction")
        logger.info(f"Extracted data for alignment: {type(actual_extracted_data)}")

        # Convert any sets in shared_state to lists
        processed_shared_state = self._convert_sets_to_lists(self.shared_state)

        # Create and run the alignment crew
        alignment_crew = self._create_crew_with_knowledge(alignment_agent, alignment_task, actual_extracted_data)

        # Process the alignment
        alignment_result = alignment_crew.kickoff(
            inputs={
                "extracted_structured_information": actual_extracted_data,
                "shared_state": processed_shared_state,
                "alignment_context": "Process the extracted information and align it with the ontology. Also take note of the shared_state that contains results from other agents as well.",
            }
        )

        print("=" * 100)
        print("Alignment Result:")
        print(alignment_result)
        print("=" * 100)

        if alignment_result:
            result_dict = alignment_result.to_dict()
            print("=" * 100)
            print("Result Dict:")
            print(result_dict)
            print("=" * 100)

            # Extract the actual aligned data from the wrapper
            actual_aligned_data = self._extract_data_from_result(result_dict, task_type, "alignment")

            # Wrap data according to task type
            wrapped_result = self._wrap_data_for_next_step(actual_aligned_data, task_type, "alignment")

            # Update shared state with the correct structure
            self._update_shared_state("aligned_terms", wrapped_result)
            logger.info(f"Alignment complete with aligned data")
            return wrapped_result
        else:
            logger.warning("Alignment crew returned no results")
            return None

    @listen(align_structured_information)
    async def judge_alignment(self, aligned_info):
        """Judge the quality of the alignment between extracted and reference terms."""
        if not aligned_info:
            logger.warning("No aligned information available. Skipping judgment.")
            return None

        logger.info("Starting judgment of aligned information")

        # Detect task type
        task_type = self._detect_task_type()
        logger.info(f"Detected task type for judgment: {task_type}")

        # Initialize judge components
        judge_agent, judge_task = self._initialize_agent_and_task("judge_agent", "judge_task", JudgedTermsDynamic)

        if not judge_agent or not judge_task:
            logger.error("Judge initialization failed")
            return None

        # Create and run the judge crew with access to all previous results
        judge_crew = self._create_crew_with_knowledge(judge_agent, judge_task, aligned_info)

        # Extract the actual aligned data from the wrapper
        actual_aligned_data = self._extract_data_from_result(aligned_info, task_type, "alignment")
        logger.info(f"Extracted aligned data for judgment: {type(actual_aligned_data)}")

        judge_result = judge_crew.kickoff(
            inputs={
                "aligned_structured_information": actual_aligned_data,
                "shared_state": self.shared_state,  # Pass shared state
            }
        )

        if not judge_result:
            logger.warning("Judge crew returned no results")
            return None

        # Update shared state and return results
        result_dict = judge_result.to_dict()

        # Extract the actual judged data from the wrapper
        actual_judged_data = self._extract_data_from_result(result_dict, task_type, "judgment")

        # Wrap data according to task type
        wrapped_result = self._wrap_data_for_next_step(actual_judged_data, task_type, "judgment")

        self._update_shared_state("judged_terms", wrapped_result)
        logger.info(f"Judgment complete with judged data")
        return wrapped_result

    @listen(judge_alignment)
    async def human_feedback(self, judge_result):
        """Process human feedback and generate improved final output."""
        if not judge_result:
            logger.warning("No judge result available. Skipping human feedback processing.")
            return None

        logger.info("Starting human feedback processing")

        # Detect task type
        task_type = self._detect_task_type()
        logger.info(f"Detected task type for human feedback: {task_type}")

        # First, request feedback on the judge's results
        if self.enable_human_feedback:
            # Extract the actual judged data from the wrapper
            actual_judged_data = self._extract_data_from_result(judge_result, task_type, "judgment")
            logger.info(f"Extracted judged data for feedback: {type(actual_judged_data)}")

            feedback_dict = self.human.request_feedback(
                data=actual_judged_data, step_name="human_feedback_processing", agent_name="humanfeedback_agent"
            )
            print(f"Debug: Feedback dictionary passed to human feedback agent: {feedback_dict}")
        else:
            # if not enabled human feedback we return the judge result as default
            feedback_dict = judge_result

        # If no feedback was provided, use the judge result
        if feedback_dict is None:
            logger.info("No feedback provided. Using judge result.")
            return judge_result

        # Ensure feedback_dict is a dictionary
        if isinstance(feedback_dict, str):
            feedback_dict = {"user_feedback_text": feedback_dict}

        # Check if user actually modified the data (not just approved)
        # The human feedback handler returns the original data for option 1 (Approve)
        # and modified data for option 3 (Modify)
        user_modified_data = has_modifications(feedback_dict, actual_judged_data)

        # Only run the human feedback agent if the user actually modified the data
        if user_modified_data:
            logger.info("User modified data, running human feedback agent")
            logger.info("*" * 100)
            logger.info("Data modified, running modification crew")
            logger.info("*" * 100)

            # Check for natural language text in feedback
            if isinstance(feedback_dict, dict) and "user_feedback_text" in feedback_dict:
                logger.info(f"Natural language feedback: {feedback_dict['user_feedback_text']}")

            # Initialize human feedback components
            humanfeedback_agent, humanfeedback_task = self._initialize_agent_and_task(
                "humanfeedback_agent", "humanfeedback_task", JudgedTermsDynamic
            )

            if not humanfeedback_agent or not humanfeedback_task:
                logger.error("Human feedback agent initialization failed")
                return judge_result

            # Format the data for the OntologyKnowledgeTool
            formatted_data = {}
            if isinstance(feedback_dict, dict):
                for key, value in feedback_dict.items():
                    if isinstance(value, list):
                        formatted_data[key] = value
                    elif isinstance(value, dict):
                        formatted_data[key] = [value]
                    else:
                        formatted_data[key] = [{"entity": str(value)}]
            else:
                formatted_data = {"terms": [{"entity": str(feedback_dict)}]}

            # Create and run the modification crew with modified data
            modification_crew = self._create_crew_with_knowledge(humanfeedback_agent, humanfeedback_task, formatted_data)

            # Process the modifications
            user_feedback_text = feedback_dict.get("user_feedback_text", "") if isinstance(feedback_dict, dict) else ""

            modified_result = modification_crew.kickoff(
                inputs={
                    "judged_structured_information_with_human_feedback": feedback_dict,
                    "aligned_structured_information": feedback_dict,
                    "shared_state": self.shared_state,
                    "modification_context": "Process the requested user feedback. Also take note of the shared_state that contains results from other agents as well. User Feedback Handling: If the input includes modifications previously made based on human/user feedback: Detect and respect these changes (e.g., altered extracted terms). Do not overwrite user-modified terms. Instead, annotate in remarks that user-defined values were retained and evaluated accordingly.",
                    "user_feedback_text": user_feedback_text,
                }
            )

            if modified_result:
                feedback_dict = modified_result.to_dict()

                # Extract the actual feedback data from the wrapper
                actual_feedback_data = self._extract_data_from_result(feedback_dict, task_type, "human_feedback")

                # Wrap data according to task type
                wrapped_result = self._wrap_data_for_next_step(actual_feedback_data, task_type, "human_feedback")

                self._update_shared_state("feedback_terms", wrapped_result)
            else:
                logger.warning("Modification processing returned no results")
                return judge_result
        else:
            logger.info("User approved data without modifications, skipping human feedback agent")
            # Use the original judge result since user just approved
            feedback_dict = judge_result

        # Update state with final results
        self.state["humanfeedback_information"] = feedback_dict
        self.state["current_step"] = "human_feedback_complete"

        logger.info(f"Human feedback processing complete")
        return feedback_dict

    def _detect_task_type(self) -> str:
        """Detect the type of task based on agent configurations."""
        try:
            extractor_role = self.agent_config.get("extractor_agent", {}).get("role", "").lower()

            if "ner" in extractor_role or "named entity" in extractor_role:
                return "ner"
            elif "bbqs" in extractor_role or "resource" in extractor_role:
                return "bbqs"
            elif "extract" in extractor_role:
                return "generic"
            else:
                return "generic"
        except Exception as e:
            logger.warning(f"Could not detect task type: {e}")
            return "generic"

    def _get_expected_output_keys(self, task_type: str, step: str) -> dict:
        """Get expected output keys for different task types and steps."""
        key_mappings = {
            "ner": {
                "extraction": "extracted_terms",
                "alignment": "aligned_ner_terms",
                "judgment": "judge_ner_terms",
                "human_feedback": "judge_ner_terms",
            },
            "bbqs": {
                "extraction": "extracted_resources",
                "alignment": "aligned_resources",
                "judgment": "judge_resource",
                "human_feedback": "judge_resource",
            },
            "generic": {
                "extraction": "extracted_terms",
                "alignment": "aligned_terms",
                "judgment": "judged_terms",
                "human_feedback": "judged_terms",
            },
        }

        return key_mappings.get(task_type, key_mappings["generic"])

    def _extract_data_from_result(self, result: dict, task_type: str, step: str) -> Any:
        """Extract the actual data from result wrapper based on task type and step."""
        if not result or not isinstance(result, dict):
            return result

        expected_keys = self._get_expected_output_keys(task_type, step)
        step_key = step.replace("_", "")  # Remove underscores for key matching

        # Try the expected key for this step
        if step_key in expected_keys:
            expected_key = expected_keys[step_key]
            if expected_key in result:
                logger.info(f"Found data under expected key '{expected_key}' for {task_type} {step}")
                return result[expected_key]

        # Try common wrapper keys
        common_wrappers = [
            "extracted_terms",
            "extracted_resources",
            "extracted_structured_information",
            "aligned_terms",
            "aligned_resources",
            "aligned_ner_terms",
            "aligned_structured_information",
            "judged_terms",
            "judge_resource",
            "judge_ner_terms",
            "judged_structured_information",
        ]

        for wrapper in common_wrappers:
            if wrapper in result:
                logger.info(f"Found data under wrapper key '{wrapper}' for {task_type} {step}")
                return result[wrapper]

        # If no wrapper found, return the result as-is
        logger.info(f"No wrapper key found, using result directly for {task_type} {step}")
        return result

    def _wrap_data_for_next_step(self, data: Any, task_type: str, step: str) -> dict:
        """Wrap data in the expected structure for the next step."""
        expected_keys = self._get_expected_output_keys(task_type, step)
        step_key = step.replace("_", "")

        if step_key in expected_keys:
            wrapper_key = expected_keys[step_key]
            return {wrapper_key: data}
        else:
            # Fallback to generic wrapper
            return {f"{step}_data": data}


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() == "true"


def kickoff(
    agentconfig: Union[str, dict],
    taskconfig: Union[str, dict],
    embedderconfig: Union[str, dict],
    input_source: Union[str, dict],
    knowledgeconfig: Optional[Union[str, dict]] = None,
    enable_human_feedback: bool = True,
    agent_feedback_config: Optional[Dict[str, bool]] = None,
    feedback_handler: Optional[ProgrammaticFeedbackHandler] = None,
    env_file: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_chunking: bool = False,
) -> Union[Dict[str, Any], str]:
    """Kickoff the StructSense flow with the given configurations.

    Args:
        agentconfig: Agent configuration file path or dictionary
        taskconfig: Task configuration file path or dictionary
        embedderconfig: Embedder configuration file path or dictionary
        input_source: Input source file path or dictionary
        knowledgeconfig: Optional knowledge configuration file path or dictionary
        enable_human_feedback: Whether to enable human feedback
        agent_feedback_config: Optional agent feedback configuration
        feedback_handler: Optional feedback handler
        env_file: Optional environment file path
        api_key: Optional API key to replace in configs
        enable_chunking: Whether to enable chunking

    Returns:
        Union[Dict[str, Any], str]: The result of the flow execution
    """
    try:
        logger.info("Starting StructSense flow...")
        logger.info("#" * 100)
        logger.info(f"Starting {enable_human_feedback}")
        logger.info("#" * 100)
        enable_human_feedback = str_to_bool(enable_human_feedback)
        try:
            if agent_feedback_config is not None:
                agent_feedback_config = load_config(agent_feedback_config, "agent_feedback_config")
                agent_feedback_config_bool = {k: str_to_bool(v) for k, v in agent_feedback_config.items()}
            else:
                agent_feedback_config_bool = {
                    "extractor_agent": False,
                    "alignment_agent": False,
                    "judge_agent": False,
                    "humanfeedback_agent": True,
                }
        except Exception:
            agent_feedback_config_bool = {
                "extractor_agent": False,
                "alignment_agent": False,
                "judge_agent": False,
                "humanfeedback_agent": True,
            }

        # Load environment variables from env_file if provided, else use the one setup using export command
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

        # Process input data
        processed_string = process_input_data(input_source)

        # Load configs if they are file paths
        if isinstance(agentconfig, str):
            agentconfig = load_config(agentconfig, "agent")
        if isinstance(taskconfig, str):
            taskconfig = load_config(taskconfig, "task")
        if isinstance(embedderconfig, str):
            embedderconfig = load_config(embedderconfig, "embedder")
        if isinstance(knowledgeconfig, str):
            knowledgeconfig = load_config(knowledgeconfig, "knowledge")

        # Replace API key if provided
        if api_key:
            agentconfig = replace_api_key(agentconfig, api_key)
            embedderconfig = replace_api_key(embedderconfig, api_key)

        # Initialize and run the flow
        flow = StructSenseFlow(
            agent_config=agentconfig,
            task_config=taskconfig,
            embedder_config=embedderconfig,
            knowledge_config=knowledgeconfig,
            source_text=processed_string,
            enable_human_feedback=enable_human_feedback,
            agent_feedback_config=agent_feedback_config_bool,
            env_file=env_file,
            enable_chunking=enable_chunking,
        )

        # Use custom feedback handler if provided
        if feedback_handler:
            flow.human = feedback_handler
            # Set the feedback handler's input and output handlers to use print
            feedback_handler.input_handler = lambda x: "3"  # Always return "3" for modify
            feedback_handler.output_handler = print

        # Run the flow
        result = flow.kickoff()

        # If feedback is required, return the feedback request
        if result == "feedback":
            if feedback_handler and feedback_handler.has_pending_feedback():
                return "feedback"
            else:
                logger.warning("No pending feedback found but result was 'feedback'")
                return None

        logger.info("Flow completed successfully")
        return result

    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}")
        raise
