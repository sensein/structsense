"""Tests for task_type detection in StructSenseFlow."""

from pathlib import Path
import os

import pytest
from dotenv import load_dotenv

from structsense.app import StructSenseFlow

skip_if_no_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)

ENV_PATH = Path(__file__).parent / "configs/.env_example"
SOURCE_TEXT_SHORT = "Retinal ganglion cell"

LLM_CONFIG = {
    "model": "openrouter/openai/gpt-4o-mini",
    "base_url": "https://openrouter.ai/api/v1",
}

BASE_AGENT_CONFIG = {
    "agent_key": {
        "role": "Neuroscience NER Extractor Agent",
        "goal": "Extract named entities from neuroscience text {input_text}.",
        "backstory": "You are an AI assistant for neuroscientists",
    }
}

BASE_TASK_CONFIG = {
    "task_key": {
        "description": "Extract entities from the input text. Use the NER tool on {input_text}.",
        "agent_id": "agent_key",
    }
}


@pytest.fixture(autouse=True)
def load_env():
    load_dotenv(ENV_PATH, override=True)


def make_flow(agent_config=None, task_config=None):
    return StructSenseFlow(
        agent_config=agent_config or BASE_AGENT_CONFIG,
        task_config=task_config or BASE_TASK_CONFIG,
        embedder_config={},
        source_text=SOURCE_TEXT_SHORT,
    )


@pytest.mark.parametrize("task_type", ["ner", "extraction"])
def test_task_type_from_config(task_type):
    """task_type is explicitly set in agent_config — returned directly without LLM or heuristic."""
    task_config = {
        "task_key": {
            **BASE_TASK_CONFIG["task_key"],
            "task_type": task_type,
        }
    }
    flow = make_flow(task_config=task_config)
    detected = flow._get_detected_task_type("agent_key", "task_key")
    assert detected == task_type


@pytest.mark.parametrize(
    "description,expected_task_type",
    [
        (None, "ner"),  # default to "ner" if no description is provided
        ("Extract resources such as datasets and tools mentioned in {input_text}.", "resource"),
    ],
)
def test_task_type_from_heuristic(monkeypatch, description, expected_task_type):
    """task_type is inferred from keywords in task description — no LLM call."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if description:
        task_config = {
            "task_key": {
                **BASE_TASK_CONFIG["task_key"],
                "description": description,
            }
        }
    else:
        task_config = BASE_TASK_CONFIG
    flow = make_flow(task_config=task_config)
    task_type = flow._get_detected_task_type("agent_key", "task_key")
    assert task_type == expected_task_type


@pytest.mark.requires_openrouter
@skip_if_no_openrouter
def test_task_type_from_llm():
    """task_type is detected by LLM call via detect_task_type."""
    agent_config = {
        "agent_key": {
            **BASE_AGENT_CONFIG["agent_key"],
            "llm": LLM_CONFIG,
        }
    }
    flow = make_flow(agent_config=agent_config)
    task_type = flow._get_detected_task_type("agent_key", "task_key")
    assert task_type == "ner"
