import concurrent.futures
import logging
import os
import time
import tracemalloc

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
from dotenv import load_dotenv

from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from utils.ontology_knowedge_tool import OntologyKnowledgeTool
from utils.text_chunking import merge_chunk_results, split_text_into_chunks
from utils.types import AlignedTermsDynamic, ExtractedTermsDynamic, JudgedTermsDynamic
from utils.utils import (
    has_modifications,
    load_config,
    process_input_data,
    replace_api_key,
    transform_extracted_data,
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
    ):
        super().__init__()
        logger.info("Initializing StructSenseFlow")
        self.source_text = source_text
        self.enable_human_feedback = enable_human_feedback
        self.human = HumanInTheLoop(enable_human_feedback=enable_human_feedback, agent_feedback_config=agent_feedback_config)
        try:
            self.agent_config = load_config(agent_config, "agent")
            self.task_config = load_config(task_config, "task")
            self.embedder_config = load_config(embedder_config, "embedder")
            if knowledge_config is None:
                os.environ["ENABLE_KG_SOURCE"] = "false"
                self.knowledge_config = {"search_key": {}}
            else:
                self.knowledge_config = load_config(knowledge_config, "knowledge")
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

        # Initialize extractor components
        extractor_agent, extractor_task = self._initialize_agent_and_task("extractor_agent", "extraction_task", ExtractedTermsDynamic)

        if not extractor_agent or not extractor_task:
            logger.error("Extractor initialization failed")
            return None

        agent_name = "extractor_agent"

        # Split text into chunks for parallel processing
        logger.info("Splitting text into chunks for parallel processing...")
        chunks = split_text_into_chunks(self.source_text)
        logger.info(f"Split text into {len(chunks)} chunks for parallel processing")
        logger.debug(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

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
                if chunk_result:
                    logger.info(
                        f"[Thread] Finished chunk {idx+1}/{total_chunks} in {elapsed:.2f}s, got {len(chunk_result.to_dict().get('terms', [])) if hasattr(chunk_result, 'to_dict') else 'unknown'} terms"
                    )
                    print("*" * 100)
                    print(f"[Thread] Finished chunk {idx+1}/{total_chunks} in {elapsed:.2f}s, got terms =  {chunk_result}")
                    print("#" * 100)
                    return chunk_result.to_dict()
                else:
                    print("*" * 100)
                    print(f"[Thread] No result for chunk {idx + 1}/{total_chunks} (took {elapsed:.2f}s)")
                    print("#" * 100)
                    logger.warning(f"[Thread] No result for chunk {idx+1}/{total_chunks} (took {elapsed:.2f}s)")
                    return None
            except Exception as exc:
                logger.error(f"[Thread] Exception in chunk {idx+1}/{total_chunks}: {exc}")
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

        # Merge chunk results
        combined_result = merge_chunk_results(chunk_results)

        # Transform the data into the expected format for alignment
        transformed_terms = transform_extracted_data(combined_result)

        # Update shared state with transformed terms
        self._update_shared_state("extracted_terms", {"terms": transformed_terms})

        total_time = time.time() - start_time
        logger.info(f"Extraction complete with {len(transformed_terms)} terms in {total_time:.2f} seconds")
        return {"terms": transformed_terms}

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
            memory_path = Path("crew_memory")
            memory_path.mkdir(exist_ok=True)

            # DEBUG: Print embedder_config before passing to RAGStorage
            print("DEBUG: embedder_config for RAGStorage:", self.embedder_config)
            # Pass the correct structure to RAGStorage
            if isinstance(self.embedder_config, dict) and "provider" in self.embedder_config:
                embedder_config_for_rag = self.embedder_config
            elif isinstance(self.embedder_config, dict) and "embedder_config" in self.embedder_config:
                embedder_config_for_rag = self.embedder_config["embedder_config"]
            else:
                embedder_config_for_rag = self.embedder_config  # fallback

            rag_storage_config = {
                "embedder_config": embedder_config_for_rag,
                "type": "short_term",
                "path": str(memory_path),
            }

            # Initialize memory components
            self.long_term_memory = LongTermMemory(storage=LTMSQLiteStorage(db_path=str(memory_path / "long_term_memory_storage.db")))
            self.short_term_memory = ShortTermMemory(storage=RAGStorage(**rag_storage_config))
            self.entity_memory = EntityMemory(storage=RAGStorage(**rag_storage_config))

            logger.info("Memory systems initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize memory systems: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)

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
        return os.getenv("ENABLE_KG_SOURCE", "false").lower() == "true"

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
            "memory": True,
            "long_term_memory_config": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "entity_memory": self.entity_memory,
            "verbose": True,
        }

        # Add knowledge sources if enabled
        if self._should_enable_knowledge_source() and data_for_knowledge_tool:
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

        return Crew(**crew_config)

    def _update_shared_state(self, key: str, value: Any) -> None:
        """Update shared state and notify other crews"""
        self.shared_state[key] = value
        self.shared_state["current_step"] = key
        logger.info(f"Updated shared state: {key}")

    def _get_shared_state(self, key: str) -> Any:
        """Get value from shared state"""
        return self.shared_state.get(key)

    @start()
    def process_inputs(self):
        """Start processing the input data."""
        logger.info("Starting structured information processing flow")
        self._update_shared_state("process_inputs", self.source_text)

    @listen(extracted_structured_information)
    async def align_structured_information(self, extracted_info):
        """Align extracted structured information with knowledge sources."""
        if not extracted_info:
            logger.warning("No structured information extracted. Skipping alignment.")
            return None

        logger.info("Starting structured information alignment")

        # Initialize alignment components
        alignment_agent, alignment_task = self._initialize_agent_and_task("alignment_agent", "alignment_task", AlignedTermsDynamic)

        if not alignment_agent or not alignment_task:
            logger.error("Alignment initialization failed")
            return None

        agent_name = "alignment_agent"

        # Create and run the alignment crew with access to extracted terms
        alignment_crew = self._create_crew_with_knowledge(alignment_agent, alignment_task, extracted_info)

        alignment_result = alignment_crew.kickoff(
            inputs={
                "extracted_structured_information": extracted_info,
                "shared_state": self.shared_state,  # Pass shared state
            }
        )

        if not alignment_result:
            logger.warning("Alignment crew returned no results")
            return None

        # Update shared state and return results
        result_dict = alignment_result.to_dict()
        self._update_shared_state("aligned_terms", result_dict)
        logger.info(f"Alignment complete with {len(result_dict.get('aligned_terms', []))} aligned terms")

        # Request human feedback on alignment results
        #  # disabled feedback for alignment
        # if self.enable_human_feedback and self.human.is_feedback_enabled_for_agent(agent_name):
        #     print("#"*100)
        #     print("requesting human feedback")
        #     print("#" * 100)
        #     feedback_dict = self.human.request_feedback(
        #         data=result_dict,
        #         step_name="information alignment",
        #         agent_name=agent_name
        #     )
        #
        #     if has_modifications(feedback_dict, result_dict):
        #         logger.info("Processing modifications based on human feedback")
        #         print("*" * 100)
        #         print("Data modified, running alignment crew again")
        #         print("*" * 100)
        #
        #         # Run the alignment crew again with modified data
        #         modified_result = alignment_crew.kickoff(inputs={
        #             "extracted_structured_information": extracted_info,
        #             "user_feedback_data": feedback_dict,
        #             "shared_state": self.shared_state,
        #             "modification_context": "Process the requested user feedback on aligned data. Also take note of the shared_state that contains results from other agents as well. User Feedback Handling: If the input includes modifications previously made based on human/user feedback: Detect and respect these changes (e.g., altered extracted terms). Do not overwrite user-modified terms. Instead, annotate in remarks that user-defined values were retained and evaluated accordingly."
        #         })
        #
        #         if modified_result:
        #             feedback_dict = modified_result.to_dict()
        #             self._update_shared_state("aligned_terms", feedback_dict)
        #         else:
        #             logger.warning("Modification processing returned no results")
        #             return result_dict
        #
        #     # Update shared state with feedback results
        #     if feedback_dict:
        #         self._update_shared_state("aligned_terms", feedback_dict)
        #         result_dict = feedback_dict

        return result_dict

    @listen(align_structured_information)
    async def judge_alignment(self, aligned_info):
        """Judge the quality of the alignment between extracted and reference terms."""
        if not aligned_info:
            logger.warning("No aligned information available. Skipping judgment.")
            return None

        logger.info("Starting judgment of aligned information")

        # Initialize judge components
        judge_agent, judge_task = self._initialize_agent_and_task("judge_agent", "judge_task", JudgedTermsDynamic)

        if not judge_agent or not judge_task:
            logger.error("Judge initialization failed")
            return None

        agent_name = "judge_agent"

        # Create and run the judge crew with access to all previous results
        judge_crew = self._create_crew_with_knowledge(judge_agent, judge_task, aligned_info)

        # Request human approval before judgment
        # if self.enable_human_feedback and self.human.is_feedback_enabled_for_agent(agent_name):
        #     # Provide observation before judgment
        #     self.human.provide_observation(
        #         message="Starting judgment process with the following aligned information:",
        #         data=f"Number of aligned terms: {len(aligned_info.get('aligned_terms', []))}",
        #         agent_name=agent_name
        #     )

        judge_result = judge_crew.kickoff(
            inputs={
                "aligned_structured_information": aligned_info,
                "shared_state": self.shared_state,  # Pass shared state
            }
        )

        if not judge_result:
            logger.warning("Judge crew returned no results")
            return None

        # Update shared state and return results
        result_dict = judge_result.to_dict()
        self._update_shared_state("judged_terms", result_dict)
        logger.info(f"Judgment complete with {len(result_dict.get('judged_terms', []))} judged terms")

        # Request human feedback on judgment results
        # if self.enable_human_feedback and self.human.is_feedback_enabled_for_agent(agent_name):
        #     feedback_dict = self.human.request_feedback(
        #         data=result_dict,
        #         step_name="judgment of alignment",
        #         agent_name=agent_name
        #     )
        #
        #     # Check if any modifications were made
        #     if has_modifications(feedback_dict, result_dict):
        #         logger.info("Processing modifications based on human feedback")
        #         logger.info("*" * 100)
        #         logger.info("Data modified, running judgment crew again")
        #         logger.info("*" * 100)
        #
        #         # Run the judge crew again with modified data
        #         modified_result = judge_crew.kickoff(inputs={
        #             "aligned_structured_information": aligned_info,
        #             "user_feedback_data": feedback_dict,
        #             "shared_state": self.shared_state,
        #             "modification_context": "Process the requested user feedback on judge agent output. Also take note of the shared_state that contains results from other agents as well. User Feedback Handling: If the input includes modifications previously made based on human/user feedback: Detect and respect these changes (e.g., altered extracted terms). Do not overwrite user-modified terms. Instead, annotate in remarks that user-defined values were retained and evaluated accordingly."
        #         })
        #
        #         if modified_result:
        #             feedback_dict = modified_result.to_dict()
        #             self._update_shared_state("judged_terms", feedback_dict)
        #         else:
        #             logger.warning("Modification processing returned no results")
        #             return result_dict
        #
        #     # Update shared state with feedback results
        #     if feedback_dict:
        #         self._update_shared_state("judged_terms", feedback_dict)
        #         result_dict = feedback_dict

        return result_dict

    @listen(judge_alignment)
    async def human_feedback(self, judge_result):
        """Process human feedback and generate improved final output."""
        if not judge_result:
            logger.warning("No judge result available. Skipping human feedback processing.")
            return None

        logger.info("Starting human feedback processing")
        agent_name = "humanfeedback_agent"

        # First, request feedback on the judge's results
        if self.enable_human_feedback:
            feedback_dict = self.human.request_feedback(data=judge_result, step_name="human_feedback_processing", agent_name=agent_name)
            print(f"Debug: Feedback dictionary passed to human feedback agent: {feedback_dict}")
        else:
            # if not enabled human feedback we return the judge result as default
            feedback_dict = judge_result

        # Ensure feedback_dict is a dictionary
        if isinstance(feedback_dict, str):
            feedback_dict = {"user_feedback_text": feedback_dict}

        # Check if any modifications were made
        if has_modifications(feedback_dict, judge_result):
            logger.info("Processing modifications based on human feedback")
            logger.info("*" * 100)
            logger.info("Data modified, running modification crew")
            logger.info("*" * 100)

            # Check for natural language text in feedback
            if "user_feedback_text" in feedback_dict:
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
            user_feedback_text = feedback_dict.get("user_feedback_text", "")

            modified_result = modification_crew.kickoff(
                inputs={
                    "judged_structured_information_with_human_feedback": feedback_dict,
                    "shared_state": self.shared_state,
                    "modification_context": "Process the requested user feedback. Also take note of the shared_state that contains results from other agents as well. User Feedback Handling: If the input includes modifications previously made based on human/user feedback: Detect and respect these changes (e.g., altered extracted terms). Do not overwrite user-modified terms. Instead, annotate in remarks that user-defined values were retained and evaluated accordingly.",
                    "user_feedback_text": user_feedback_text,
                }
            )

            if modified_result:
                feedback_dict = modified_result.to_dict()
                self._update_shared_state("feedback_terms", feedback_dict)
            else:
                logger.warning("Modification processing returned no results")
                return judge_result

        # Update state with final results
        self.state["humanfeedback_information"] = feedback_dict
        self.state["current_step"] = "human_feedback_complete"

        logger.info(f"Human feedback processing complete with {len(feedback_dict.get('judged_terms', []))} final terms")
        return feedback_dict


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
