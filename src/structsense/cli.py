"""This module defines CLI commands for the PipePal application."""

import logging

import click
import yaml

from utils.utils import load_config

from .app import kickoff

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """CLI commands for the Structsense Framework application"""
    pass


@cli.command()
@click.option("--config", required=True, type=str, help="Path to the single YAML config file (ner_config.yaml).")
@click.option("--api_key", required=False, type=str, help="Open router API key.")
@click.option(
    "--source",
    required=True,
    help=("The source—whether a file (text or PDF), a folder, or a text string."),
)
@click.option("--env_file", required=False, type=str, help="Optional path to an environment file to override the default .env file.")
@click.option("--save_file", required=False, type=str, help="Optional path to save the result as a JSON file.")
@click.option("--chunking", required=False, default=False, help="Enable text chunking for parallel processing (default: False).")
def extract(config, api_key, source, env_file, save_file, chunking):
    """Extract the terms along with sentence using a single config file."""
    # Load the config file
    all_config = load_config(config, "all")

    # Extract the different config sections
    agent_config = all_config.get("agent_config", {})
    embedder_config = all_config.get("embedder_config", {})
    task_config = all_config.get("task_config", {})
    knowledge_config = all_config.get("knowledge_config", {})
    human_in_loop_config = all_config.get("human_in_loop_config", {})

    # Run the extraction
    result = kickoff(
        agentconfig=agent_config,
        taskconfig=task_config,
        embedderconfig=embedder_config,
        knowledgeconfig=knowledge_config,
        input_source=source,
        enable_human_feedback=True,
        agent_feedback_config=human_in_loop_config,
        env_file=env_file,
        api_key=api_key,
        enable_chunking=chunking,
    )

    # Output results
    click.echo("*" * 100)
    click.echo("Result")
    click.echo(result)
    click.echo("*" * 100)

    # Save to file if requested
    if save_file:
        import json

        with open(save_file, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Result saved to {save_file}")


# @cli.command(
#     help="Run the Structured Information Extraction (SIE) pipeline using default configurations. For custom configs, use 'extract'."
# )
# @click.option(
#     "--api_key",
#     required=False,
#     type=str,
#     help="API key (e.g., OpenRouter)."
# )
# @click.option(
#     "--source",
#     required=True,
#     type=str,
#     help="The source—whether a file (text or PDF), a folder, or a text string."
# )
# @click.option(
#     "--env_file",
#     required=False,
#     type=str,
#     help="Optional path to an environment file to override the default .env file."
# )
# @click.option(
#     "--save_file",
#     required=False,
#     type=str,
#     help="Optional path to save the result as a JSON file."
# )
# @click.option(
#     "--chunking",
#     is_flag=True,
#     default=False,
#     help="Enable text chunking for parallel processing (default: False)."
# )
# def sie(api_key, source, env_file, save_file, chunking):
#     """
#     Run the Structured Information Extraction (SIE) pipeline using the default config file.
#     """
#     import os
#     default_config_path = os.path.join(
#         os.path.dirname(__file__), "default_config_sie", "ner_config.yaml"
#     )
#     with open(default_config_path, 'r') as f:
#         all_config = yaml.safe_load(f)
#     agent_config = all_config.get('agent_config', {})
#     embedder_config = all_config.get('embedder_config', {})
#     task_config = all_config.get('task_config', {})
#     knowledge_config = all_config.get('knowledge_config', {})
#     human_in_loop_config = all_config.get('human_in_loop_config', {})
#     result = kickoff(
#         agentconfig=agent_config,
#         taskconfig=task_config,
#         embedderconfig=embedder_config,
#         knowledgeconfig=knowledge_config,
#         input_source=source,
#         enable_human_feedback=True,
#         agent_feedback_config=human_in_loop_config,
#         env_file=env_file,
#         api_key=api_key,
#         enable_chunking=chunking
#     )
#     click.echo("*" * 100)
#     click.echo("Result")
#     click.echo(result)
#     click.echo("*" * 100)
#     if save_file:
#         import json
#         with open(save_file, 'w') as f:
#             json.dump(result, f, indent=2)
#         click.echo(f"Result saved to {save_file}")


if __name__ == "__main__":
    cli()
