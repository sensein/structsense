"""Testing Simple NER examples."""

from pathlib import Path
import asyncio
import yaml

import pytest

from structsense.app import StructSenseFlow
from .conftest import skip_if_no_openrouter

pytestmark = [pytest.mark.usefixtures("load_env"), pytest.mark.requires_openrouter]

CONFIG_PATH = Path(__file__).parent / "configs/ner-config_extractonly.yaml"
SOURCE_TEXT = "Retinal ganglion cell (RGC) axons and synapses were genetically labeled via AAV transduction"


@skip_if_no_openrouter
def test_ner_1():
    """Test the NER extraction with a simple text input.
    It only checks if it extracts any entities, since I observed inconsistency.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    flow = StructSenseFlow(
        agent_config=config["agent_config"],
        task_config=config["task_config"],
        embedder_config=config.get("embedder_config", {}),
        source_text=SOURCE_TEXT,
    )
    enr_result = asyncio.run(flow.information_extraction_task())

    # uses OPENROUTER_API_KEY set as environment variable for authentication
    assert enr_result is not None
    assert "entities" in enr_result
    assert len(enr_result["entities"]) > 0
    # print the extracted entities for visual inspection (hard to assert exact entities, at least with this model)
    print(f"Extracted entities: {[el['entity'] for el in enr_result['entities']]}")
