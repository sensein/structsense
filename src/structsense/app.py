import warnings

warnings.filterwarnings("ignore")

import logging
import os
import sys
import tracemalloc


from crewai import Crew
from crewai.flow.flow import Flow, start
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
import os

tracemalloc.start()
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
    import weave
    weave.init(project_name="StructSense")

# Optional MLflow setup
if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
    import mlflow
    mlflow.crewai.autolog()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
    mlflow.set_experiment("StructSense")


class StructSenseFlow(Flow):
    def __init__(
        self,
        agent_config,
        task_config,
        embedder_config,
        flow_config,
        knowledge_config,
        source_text,
    ):
        super().__init__()

        self.source_text = source_text
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

        self.long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="crew_memory/long_term_memory_storage.db")
        )
        self.short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                embedder_config=self.embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/",
            )
        )
        self.entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config=self.embedderconfig.get("embedder_config"),
                type="short_term",
                path="crew_memory/",
            )
        )

        if not hasattr(self, "state") or self.state is None:
            self.__dict__["state"] = {}

        self.state["source_text"] = source_text

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

    def run_step(self, step):
        logger.info(f"Running step: {step['id']}")

        agent_key = step["agent_key"]
        task_key = step["task_key"]
        tools = []

        agent_def = self.agentconfig[agent_key]
        agent_builder = DynamicAgent([agent_def], self.embedderconfig, tools)
        agents_by_id = agent_builder.build_agents()
        agent = agents_by_id[agent_key]

        task_builder = DynamicAgentTask([self.taskconfig[task_key]])
        tasks = task_builder.build_tasks(agents_by_id)
        task = tasks[0]

        inputs = {
            k: self.interpolate(v, self.state)
            for k, v in step.get("inputs", {}).items()
        }

        ksrc = None
        enable_kg_source = os.getenv("ENABLE_KG_SOURCE", "false").lower() == "true"
        if not enable_kg_source:
            logger.info("Knowledge source disabled via ENABLE_KG_SOURCE=false")

        if enable_kg_source:
            if "knowledge_source" in step:
                src_key = step["knowledge_source"]
                if src_key in self.state:
                    logger.info(
                        f"Knowledge source response str: {self.knowledgeconfig['search_key']}"
                    )
                    # Tool is not used because of https://github.com/crewAIInc/crewAI/issues/949
                    custom_source = OntologyKnowledgeTool(
                        self.state[src_key], self.knowledgeconfig["search_key"]
                    )
                    logger.info(f"Knowledge source response: {custom_source}")
                    ksrc = StringKnowledgeSource(content=custom_source)

        crew_kwargs = {
            "agents": [agent],
            "tasks": [task],
            "memory": True,
            "long_term_memory_config": self.long_term_memory,
            "short_term_memory": self.short_term_memory,
            "entity_memory": self.entity_memory,
            "verbose": True,
        }

        if enable_kg_source and ksrc:
            crew_kwargs["knowledge_sources"] = [ksrc]

        crew = Crew(**crew_kwargs)

        result = crew.kickoff(inputs=inputs)
        task_output = task.output

        output_var = self.agentconfig.get(agent_key, {}).get(
            "output_variable", step["id"]
        )
        logger.debug(
            f"[DEBUG] Output variable for step '{step['id']}' is '{output_var}'"
        )

        logger.info(f"JSON Output: {task_output, type(task_output)}")

        self.state[output_var] = task_output.raw
        logger.info(
            f"State updated with output_variable '{output_var}' => {self.state[output_var]}"
        )
        return result

    @start("kickoff")
    def kickoff_flow(self):
        for step in self.flowconfig["flow"]:
            self.run_step(step)
        return self.state

def kickoff(
    agentconfig: str,
    taskconfig: str,
    embedderconfig: str,
    flowconfig: str,
    input_source: str,
    knowledgeconfig: str = None
):
    processed_string = process_input_data(input_source)

    flow = StructSenseFlow(
        agent_config=agentconfig,
        task_config=taskconfig,
        embedder_config=embedderconfig,
        flow_config=flowconfig,
        knowledge_config=knowledgeconfig,
        source_text=processed_string,
    )
    result = flow.kickoff()
    # print(result)
