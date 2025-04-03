"""This module defines CLI commands for the PipePal application."""

import logging

import click

from .app import kickoff

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """CLI commands for the Structsense Framework application"""
    pass


@cli.command()
@click.option(
    "--agentconfig",
    required=True,
    type=str,
    help=("Path to the agent configuration in YAML file format or dictionary"),
)
@click.option(
    "--taskconfig",
    required=True,
    type=str,
    help=("Path to the agent task configuration in YAML format or or dictionary"),
)
@click.option(
    "--embedderconfig",
    required=True,
    type=str,
    help=("Path to the embedding configuration in YAML format or or dictionary"),
)
@click.option(
    "--flowconfig",
    required=True,
    type=str,
    help=(
        "Path to the flow configuration in YAML format or or dictionary. The flow configuration describes the flow of the agent."
    ),
)
@click.option(
    "--knowledgeconfig",
    required=False,
    type=str,
    help=(
        "Path to the configuration in YAML format or or dictionary that specify the search knowledge search key."
    ),
)
@click.option(
    "--source",
    required=True,
    help=("The source—whether a file (text or PDF), a folder, or a text string."),
)
def extract(
    agentconfig, taskconfig, embedderconfig, flowconfig, knowledgeconfig, source
):
    """Extract the terms along with sentence."""
    logger.info(
        f"Processing source: {source} with agent config: {agentconfig}, task config: {taskconfig}, embedderconfig: {embedderconfig}, flowconfig: {flowconfig}, knowledgeconfig: {knowledgeconfig}"
    )
    click.echo(
        f"Processing source: {source} with agent config: {agentconfig}, task config: {taskconfig}, embedderconfig: {embedderconfig} knowledgeconfig: {knowledgeconfig} flowconfig: {flowconfig}"
    )
    result = kickoff(
        agentconfig=agentconfig,
        taskconfig=taskconfig,
        embedderconfig=embedderconfig,
        flowconfig=flowconfig,
        knowledgeconfig=knowledgeconfig,
        input_source=source,
    )

    click.echo("*"*100)
    click.echo("Result")
    click.echo(result)
    click.echo("*"*100)


if __name__ == "__main__":
    cli()
