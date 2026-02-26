"""Tests for StructSenseFlow initialization and source validation."""

from pathlib import Path

import pytest
from structsense.app import StructSenseFlow, ConfigError
from utils.utils import load_config


@pytest.fixture
def config_parser():
    config_path = Path(__file__).parent / "configs/ner-config_free.yaml"
    all_config = load_config(str(config_path), "all")
    agent_config = all_config.get("agent_config", {})
    embedder_config = all_config.get("embedder_config", {})
    task_config = all_config.get("task_config", {})
    knowledge_config = all_config.get("knowledge_config", {})
    return agent_config, embedder_config, task_config, knowledge_config


def test_invalid_source_path_error(config_parser):
    """StructSenseFlow raises ValueError when source path does not exist."""
    agent_config, embedder_config, task_config, knowledge_config = config_parser

    with pytest.raises(ValueError, match="File not found"):
        StructSenseFlow(
            source="/nonexistent/path/to/file.txt",
            agent_config=agent_config,
            task_config=task_config,
            embedder_config=embedder_config,
        )


def test_multiple_source_error(config_parser):
    """StructSenseFlow raises ConfigError when both source and source_text are provided."""
    agent_config, embedder_config, task_config, knowledge_config = config_parser

    with pytest.raises(ConfigError):
        StructSenseFlow(
            source_text="This is a text string input",
            source="/path/to/file.txt",
            agent_config=agent_config,
            task_config=task_config,
            embedder_config=embedder_config,
        )


def test_source_text(config_parser):
    """StructSenseFlow initializes correctly with source_text input."""
    agent_config, embedder_config, task_config, knowledge_config = config_parser

    flow = StructSenseFlow(
        source_text="This is a text string input",
        agent_config=agent_config,
        task_config=task_config,
        embedder_config=embedder_config,
    )

    assert flow.source_text == "This is a text string input"


def test_source_file(config_parser, tmp_path):
    """StructSenseFlow initializes correctly with valid source file."""
    agent_config, embedder_config, task_config, knowledge_config = config_parser

    # Create a temporary file with some content
    temp_file = tmp_path / "test_input.txt"
    temp_file.write_text("This is a test file input.")

    flow = StructSenseFlow(
        source=str(temp_file),
        agent_config=agent_config,
        task_config=task_config,
        embedder_config=embedder_config,
    )

    assert flow.source_text == "This is a test file input."
