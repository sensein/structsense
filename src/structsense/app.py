import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

warnings.filterwarnings("ignore")

import logging
import os
import sys
import tracemalloc
from dataclasses import dataclass

from crewai import Crew
from crewai.flow.flow import Flow, start, listen
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from dotenv import load_dotenv
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crew.dynamic_agent import DynamicAgent
from crew.dynamic_agent_task import DynamicAgentTask
from utils.utils import load_config
from utils.ontology_knowedge_tool import OntologyKnowledgeTool
from utils.utils import process_input_data
from utils.utils import extract_json_from_text

@dataclass
class FlowConfig:
    agent_config: str
    task_config: str
    embedder_config: str
    flow_config: str
    knowledge_config: Optional[str]
    source_text: str


class StructSenseError(Exception):
    """Base exception for StructSense errors"""
    pass


class ConfigError(StructSenseError):
    """Raised when there's an error in configuration"""
    pass


class FlowExecutionError(StructSenseError):
    """Raised when there's an error during flow execution"""
    pass


tracemalloc.start()
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize monitoring tools if enabled
if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
    import weave

    weave.init(project_name="StructSense")

if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
    import mlflow

    mlflow.crewai.autolog()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
    mlflow.set_experiment("StructSense")


class StructSenseFlow(Flow):
    def __init__(
            self,
            agent_config: str,
            task_config: str,
            embedder_config: str,
            flow_config: str,
            knowledge_config: Optional[str],
            source_text: str,
    ):
        super().__init__()
        try:
            self.source_text = source_text
            self._load_configurations(agent_config, task_config, embedder_config, flow_config, knowledge_config)
            self._initialize_memory()
            self._initialize_state()
        except Exception as e:
            raise ConfigError(f"Failed to initialize StructSenseFlow: {str(e)}")

    def _load_configurations(
            self,
            agent_config: str,
            task_config: str,
            embedder_config: str,
            flow_config: str,
            knowledge_config: Optional[str]
    ) -> None:
        """Load and validate all configurations"""
        try:
            raw_agent_config = load_config(agent_config, "agent")
            raw_task_config = load_config(task_config, "task")
            self.agentconfig = {agent["id"]: agent for agent in raw_agent_config["agents"]}
            self.taskconfig = {task["id"]: task for task in raw_task_config["tasks"]}
            self.embedderconfig = load_config(embedder_config, "embedder")
            self.flowconfig = load_config(flow_config, "flow")

            if knowledge_config is None:
                os.environ["ENABLE_KG_SOURCE"] = "false"
                self.knowledgeconfig = {"search_key": {}}
            else:
                self.knowledgeconfig = load_config(knowledge_config, "knowledge")
        except Exception as e:
            raise ConfigError(f"Failed to load configurations: {str(e)}")

    def _initialize_memory(self) -> None:
        """Initialize memory storage systems"""
        try:
            memory_path = Path("crew_memory")
            memory_path.mkdir(exist_ok=True)

            self.long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(db_path=str(memory_path / "long_term_memory_storage.db"))
            )
            self.short_term_memory = ShortTermMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path=str(memory_path),
                )
            )
            self.entity_memory = EntityMemory(
                storage=RAGStorage(
                    embedder_config=self.embedderconfig.get("embedder_config"),
                    type="short_term",
                    path=str(memory_path),
                )
            )
        except Exception as e:
            raise ConfigError(f"Failed to initialize memory systems: {str(e)}")

    def _initialize_state(self) -> None:
        """Initialize the flow state"""
        if not hasattr(self, "state") or self.state is None:
            self.__dict__["state"] = {}
        self.state["source_text"] = self.source_text

    def interpolate(self, template, context):
        import re

        pattern = re.compile(r"\{\{(.*?)\}\}")

        def resolve_path(path, ctx):
            keys = path.strip().split(".")
            val = ctx
            for key in keys:
                val = val.get(key) if isinstance(val, dict) else None
                if val is None:
                    break
            return val

        def replacer(match):
            path = match.group(1).strip()
            val = resolve_path(path, context)
            return str(val) if val is not None else ""

        result = pattern.sub(replacer, template)
        logger.debug(f"Interpolated: {template} -> {result}")
        return result

    @start("start")
    def kickoff_flow(self) -> Dict[str, Any]:
        """Start the flow execution"""
        try:
            # Validate flow configuration
            if not self.flowconfig or "flow" not in self.flowconfig:
                raise ConfigError("Flow configuration is missing or invalid")

            # Execute each step in the flow
            for step in self.flowconfig["flow"]:
                if not isinstance(step, dict):
                    raise ConfigError(f"Invalid step configuration: {step}")

                # Add step ID if missing
                if "id" not in step:
                    step["id"] = f"step_{len(self.flowconfig['flow'])}"

                # Execute the step and return output if it's the last step
                result = self.run_step(step)
                if step == self.flowconfig["flow"][-1]:
                    return result

            return {"output": None}

        except Exception as e:
            logger.error(f"Flow execution failed: {str(e)}")
            raise FlowExecutionError(f"Flow execution failed: {str(e)}")

    def run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the flow"""
        try:
            # Debug logging for step object
            logger.debug(f"Received step object: {step}")

            # Validate step object
            if not isinstance(step, dict):
                raise ConfigError(f"Invalid step object: {step}")

            # Get step ID with fallback
            step_id = step.get('id', 'unknown_step')
            logger.info(f"Running step: {step_id}")

            # Validate required fields
            required_fields = ['agent_key', 'task_key']
            missing_fields = [field for field in required_fields if field not in step]
            if missing_fields:
                raise ConfigError(f"Step {step_id} is missing required fields: {', '.join(missing_fields)}")

            agent_key = step["agent_key"]
            task_key = step["task_key"]

            if agent_key not in self.agentconfig:
                raise ConfigError(f"Agent {agent_key} not found in configuration")
            if task_key not in self.taskconfig:
                raise ConfigError(f"Task {task_key} not found in configuration")

            agent_def = self.agentconfig[agent_key]
            agent_builder = DynamicAgent([agent_def], self.embedderconfig, [])
            agents_by_id = agent_builder.build_agents()
            agent = agents_by_id[agent_key]

            task_builder = DynamicAgentTask([self.taskconfig[task_key]])
            tasks = task_builder.build_tasks(agents_by_id)
            task = tasks[0]

            inputs = {
                k: self.interpolate(v, self.state)
                for k, v in step.get("inputs", {}).items()
            }

            crew_kwargs = self._prepare_crew_kwargs(agent, task, step)
            crew = Crew(**crew_kwargs)
            result = crew.kickoff(inputs=inputs)

            # Log step output for debugging
            if hasattr(task, 'output') and task.output:
                logger.debug(f"Step {step_id} output: {task.output.raw}")

            self._update_state(agent_key, task, step)

            # Format and return the output
            if hasattr(task, 'output') and task.output:
                output = task.output.raw
                if isinstance(output, str):
                    try:
                        output = extract_json_from_text(output)
                    except ValueError:
                        # If JSON extraction fails, return the raw string
                        pass
                return {"output": output}

            return {"output": None}

        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            raise FlowExecutionError(f"Step execution failed: {str(e)}")

    def _prepare_crew_kwargs(self, agent: Any, task: Any, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare crew configuration"""
        crew_kwargs = {
            "agents": [agent],
            "tasks": [task],
            "memory": True,
            "long_term_memory_config": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "entity_memory": self.entity_memory,
            "verbose": False,
        }

        if self._should_enable_knowledge_source(step):
            ksrc = self._create_knowledge_source(step)
            if ksrc:
                crew_kwargs["knowledge_sources"] = [ksrc]

        return crew_kwargs

    def _should_enable_knowledge_source(self, step: Dict[str, Any]) -> bool:
        """Check if knowledge source should be enabled"""
        return (
                os.getenv("ENABLE_KG_SOURCE", "false").lower() == "true"
                and "knowledge_source" in step
                and step["knowledge_source"] in self.state
        )

    def _create_knowledge_source(self, step: Dict[str, Any]) -> Optional[StringKnowledgeSource]:
        """Create knowledge source if needed"""
        try:
            src_key = step["knowledge_source"]
            logger.info(f"Knowledge source response str: {self.knowledgeconfig['search_key']}")
            custom_source = OntologyKnowledgeTool(
                self.state[src_key], self.knowledgeconfig["search_key"]
            )
            logger.debug("*"*100)
            logger.debug(custom_source)
            logger.debug("*"*100)
            return StringKnowledgeSource(content=custom_source)
        except Exception as e:
            logger.warning(f"Failed to create knowledge source: {str(e)}")
            return None

    def _update_state(self, agent_key: str, task: Any, step: Dict[str, Any]) -> None:
        """Update the flow state with task output"""
        try:
            # Get step ID with fallback
            step_id = step.get('id', 'unknown_step')
            output_var = self.agentconfig.get(agent_key, {}).get("output_variable", step_id)
            self.state[output_var] = task.output.raw

            # Check if this is the last step
            if step == self.flowconfig["flow"][-1]:
                self.final_result = task.output
        except Exception as e:
            logger.error(f"Failed to update state: {str(e)}")
            raise FlowExecutionError(f"Failed to update state: {str(e)}")


def kickoff(
        agentconfig: str,
        taskconfig: str,
        embedderconfig: str,
        flowconfig: str,
        input_source: str,
        knowledgeconfig: Optional[str] = None
) -> Dict[str, Any]:
    """Run the StructSense flow with the given configurations"""
    try:
        processed_string = process_input_data(input_source)

        flow = StructSenseFlow(
            agent_config=agentconfig,
            task_config=taskconfig,
            embedder_config=embedderconfig,
            flow_config=flowconfig,
            knowledge_config=knowledgeconfig,
            source_text=processed_string,
        )

        final_response =  flow.kickoff()
        logger.info(f"Returning {final_response}")
        return final_response
    except Exception as e:
        logger.error(f"Flow execution failed: {str(e)}")
        raise
