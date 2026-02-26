"""Tests for the CLI commands."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from structsense.cli import cli

CONFIG_PATH = str(Path(__file__).parent / "configs/ner-config_free.yaml")

SOURCE_TEXT = "Retinal ganglion cell (RGC) axons and synapses were genetically labeled via AAV transduction"


def test_extract_invalid_source_path():
    """CLI extract command exits with an error when --source path does not exist."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["extract", "--config", CONFIG_PATH, "--source", "/nonexistent/path/to/file.txt"],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "--source" in result.output
