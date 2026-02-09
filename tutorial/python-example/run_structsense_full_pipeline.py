import yaml
from structsense.app import StructSenseFlow
import asyncio
import json
#load the configuration file for NER
with open("ner-config.yaml") as f:
    all_config = yaml.safe_load(f)
    
flow = StructSenseFlow(
    agent_config=all_config["agent_config"],
    task_config=all_config["task_config"],
    embedder_config=all_config.get("embedder_config", {}),
    input_source="test_small.pdf",
    enable_chunking=True,
    chunk_size=2000,
    max_workers=8,
    env_file=".env_example",
    api_key="sk-or-v1-change",
)
result = asyncio.run(flow.information_extraction_task())

with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)