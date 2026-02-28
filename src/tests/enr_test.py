"""Testing Simple NER examples."""

from pathlib import Path
import json

import pytest
from click.testing import CliRunner

from structsense.cli import cli

pytestmark = pytest.mark.requires_openrouter

CONFIG_PATH = str(Path(__file__).parent / "configs/ner-config_free.yaml")
ENV_PATH = str(Path(__file__).parent / "configs/.env_example")
SOURCE_TEXT = "Retinal ganglion cell (RGC) axons and synapses were genetically labeled via AAV transduction"
ENTITIES_EXPECTED = ["synapses"]


def test_enr_1(tmp_path):
    """Test the ENR extraction with a simple text input and a free model."""
    runner = CliRunner()
    # uses OPENROUTER_API_KEY set as environment variable (not provided in the env file) for authentication (the model is free)
    result = runner.invoke(
        cli,
        [
            "extract",
            "--env_file",
            ENV_PATH,
            "--config",
            CONFIG_PATH,
            "--source_text",
            SOURCE_TEXT,
            "--save_file",
            str(tmp_path / "enr_result.json"),
        ],
    )

    with open(tmp_path / "enr_result.json", "r") as f:
        enr_result = json.load(f)

    # testing if we get any entities
    assert result.exit_code == 0
    assert "entities" in enr_result
    assert len(enr_result["entities"]) > 0

    # testing if we get the expected entity types (it's not a comprehensive test of all expected entity)
    entities_enr = [el["entity"] for el in enr_result["entities"]]
    for expected_entity in ENTITIES_EXPECTED:
        assert expected_entity in entities_enr
