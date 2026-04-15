"""This module defines CLI commands for the PipePal application."""

import logging

import click
import yaml

from utils.utils import load_config
from dotenv import load_dotenv
import os
import asyncio

from .app import StructSenseFlow

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """CLI commands for the Structsense Framework application"""
    pass


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to the single YAML config file (ner_config.yaml).")
@click.option("--api_key", required=False, type=str, help="Open router API key.")
@click.option(
    "--source",
    type=click.Path(exists=True),
    help=("The path to file to process, (PDF, csv or txt). This is an alternative to providing a text string with --source_text."),
)
@click.option(
    "--source_text",
    type=str,
    help=("The text string that should be used as input directly. This is an alternative to providing a file path with --source."),
)
@click.option(
    "--env_file",
    required=False,
    type=click.Path(exists=True),
    help="Optional path to an environment file to override the default .env file.",
)
@click.option("--save_file", required=False, type=str, help="Optional path to save the result as a JSON file.")
@click.option("--chunk_size", required=False, type=int, default=None, help="Chunk size in characters for extraction (None = no chunking).")
@click.option("--max_workers", required=False, type=int, default=None, help="Maximum parallel workers (None = auto).")
@click.option(
    "--enable_chunking",
    required=False,
    default=False,
    is_flag=True,
    help="Enable chunking (uses default chunk_size if --chunk_size not provided).",
)
@click.option(
    "--downstream_max_input_chars",
    required=False,
    type=int,
    default=None,
    help="Max input chars for alignment/judge/humanfeedback (default 80000).",
)
@click.option(
    "--max_extraction_chunk_chars",
    required=False,
    type=int,
    default=None,
    help="Cap extraction chunk size in chars so chunk+prompt stays under model context (default 25000 for 128k models). None = no cap.",
)
@click.option(
    "--skip_alignment_llm",
    required=False,
    default=None,
    type=click.Choice(["true", "false", "auto"], case_sensitive=False),
    help=(
        "Skip the alignment LLM and call the concept mapping tool directly instead. "
        "'auto' (default): skip when CONCEPT_MAPPING_BACKEND=local and task is NER/keyphrase/resource. "
        "'true': always skip. 'false': always run the alignment LLM. "
        "Can also be set via env var SKIP_ALIGNMENT_LLM=true/false/auto."
    ),
)
@click.option(
    "--skip_judge_llm",
    required=False,
    default=None,
    type=click.Choice(["true", "false"], case_sensitive=False),
    help=(
        "Skip the judge LLM and inject default judge_score=1.0 / remarks='auto-approved' instead. "
        "'true': always skip. 'false': always run the judge LLM (default). "
        "Can also be set via env var SKIP_JUDGE_LLM=true/false."
    ),
)
@click.option(
    "--downstream_chunk_size",
    required=False,
    type=int,
    default=None,
    help=(
        "Entities per chunk for parallel alignment/judge/humanfeedback. "
        "When --enable_chunking is set, downstream stages are split into chunks of this size "
        "and run in parallel (default: auto-calculated as ceil(total_entities / max_workers)). "
        "E.g. with 800 entities and max_workers=8, default is 100 entities/chunk = 8 parallel jobs."
    ),
)
@click.option(
    "--model_context_window",
    required=False,
    type=int,
    default=None,
    help=(
        "Override the auto-detected model context window (tokens) used for downstream chunk sizing. "
        "Use this when your model is not in the built-in registry or is behind a custom proxy. "
        "E.g. --model_context_window 200000 for a 200k-token model. "
        "When omitted, the context window is auto-detected from the model name in the config."
    ),
)
@click.option(
    "--agent_max_iter",
    required=False,
    type=int,
    default=None,
    help=(
        "Maximum number of reasoning iterations each CrewAI agent is allowed per run "
        "before it is forced to return its best answer. Applies to every agent in the "
        "pipeline (extractor, alignment, judge, humanfeedback). "
        "CrewAI default is 20. Lower values (e.g. 3-5) reduce cost and latency for "
        "straightforward tasks; higher values give agents more attempts on complex inputs. "
        "Can also be set via env var AGENT_MAX_ITER=<int>."
    ),
)
@click.option(
    "--agent_max_execution_time",
    required=False,
    type=int,
    default=None,
    help=(
        "Maximum wall-clock seconds each CrewAI agent is allowed to run before it is "
        "interrupted and forced to return its current best answer. Default: 30s. "
        "Set to a larger value for complex tasks. "
        "Can also be set via env var AGENT_MAX_EXECUTION_TIME=<int>."
    ),
)
@click.option(
    "--agent_max_retry_limit",
    required=False,
    type=int,
    default=None,
    help=(
        "Maximum number of times each CrewAI agent retries after a recoverable error "
        "(e.g. tool failure, parse error). Default: 0 (fail fast). "
        "Can also be set via env var AGENT_MAX_RETRY_LIMIT=<int>."
    ),
)
@click.option(
    "--preload_stage",
    "preload_stages",
    required=False,
    multiple=True,
    metavar="TASK_KEY:FILE",
    help=(
        "Skip a pipeline stage by loading its output from a saved JSON file. "
        "Format: TASK_KEY:path/to/file.json  (e.g. extraction_task:00_extractor_agent_extraction_task.json). "
        "Repeat the flag for each stage you want to skip. "
        "Valid task keys: extraction_task, alignment_task, judge_task, humanfeedback_task."
    ),
)
@click.option(
    "--skip_stage",
    "skip_stages",
    required=False,
    multiple=True,
    metavar="TASK_KEY",
    help=(
        "Omit a pipeline stage entirely. The previous stage's output is passed directly "
        "to the next non-skipped stage. Repeat the flag for each stage to skip. "
        "Examples:\n\n"
        "  Run extraction + alignment only:\n"
        "    --skip_stage judge_task --skip_stage humanfeedback_task\n\n"
        "  Run extraction + judge (skip alignment):\n"
        "    --skip_stage alignment_task\n\n"
        "Valid task keys: alignment_task, judge_task, humanfeedback_task. "
        "(extraction_task cannot be skipped; use --preload_stage for that.)"
    ),
)
def extract(
    config,
    api_key,
    source,
    source_text,
    env_file,
    save_file,
    chunk_size,
    max_workers,
    enable_chunking,
    downstream_max_input_chars,
    max_extraction_chunk_chars,
    skip_alignment_llm,
    skip_judge_llm,
    downstream_chunk_size,
    model_context_window,
    agent_max_iter,
    agent_max_execution_time,
    agent_max_retry_limit,
    preload_stages,
    skip_stages,
):
    """Extract the terms along with sentence using a single config file."""
    import json

    # Parse --preload_stage KEY:FILE entries into a dict
    preloaded_stages_dict = {}
    for entry in preload_stages:
        if ":" not in entry:
            raise click.UsageError(f"--preload_stage must be in TASK_KEY:FILE format, got: {entry!r}")
        task_key, file_path = entry.split(":", 1)
        if not os.path.exists(file_path):
            raise click.UsageError(f"Preload file not found: {file_path!r}")
        with open(file_path) as fh:
            preloaded_stages_dict[task_key] = json.load(fh)
        click.echo(f"Preloaded stage '{task_key}' from {file_path}")

    # Load the config file
    if source and source_text:
        raise click.UsageError("Please provide either --source or --source_text, not both.")
    elif not source and not source_text:
        raise click.UsageError("Please provide either --source or --source_text.")
    all_config = load_config(config, "all")

    # Extract the different config sections
    agent_config = all_config.get("agent_config", {})
    embedder_config = all_config.get("embedder_config", {})
    task_config = all_config.get("task_config", {})
    # Human-in-the-loop: from YAML human_in_loop_config (env ENABLE_HUMAN_FEEDBACK still overrides inside StructSenseFlow)
    human_in_loop = all_config.get("human_in_loop_config") or {}
    enable_human_feedback = bool(human_in_loop.get("humanfeedback_agent", False))
    if "ENABLE_HUMAN_FEEDBACK" in os.environ:
        from utils.utils import str_to_bool

        enable_human_feedback = str_to_bool(os.environ["ENABLE_HUMAN_FEEDBACK"])

    # Use StructSenseFlow as the single entry point
    flow = StructSenseFlow(
        agent_config=agent_config,
        task_config=task_config,
        embedder_config=embedder_config,
        source=source,
        source_text=source_text,
        env_file=env_file,
        api_key=api_key,
        enable_human_feedback=enable_human_feedback,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        max_workers=max_workers,
        downstream_max_input_chars=downstream_max_input_chars,
        max_extraction_chunk_chars=max_extraction_chunk_chars,
        downstream_chunk_size=downstream_chunk_size,
        model_context_window=model_context_window,
        skip_alignment_llm=(None if skip_alignment_llm == "auto" or skip_alignment_llm is None else skip_alignment_llm.lower() == "true"),
        skip_judge_llm=(None if skip_judge_llm is None else skip_judge_llm.lower() == "true"),
        skip_stages=list(skip_stages) if skip_stages else None,
        agent_max_iter=agent_max_iter,
        agent_max_execution_time=agent_max_execution_time,
        agent_max_retry_limit=agent_max_retry_limit,
    )

    # Run the full pipeline (extraction → alignment → judge → humanfeedback)
    result = asyncio.run(flow.information_extraction_task(preloaded_stages=preloaded_stages_dict if preloaded_stages_dict else None))

    # Output results
    click.echo("*" * 100)
    click.echo("Result")
    click.echo(result)
    click.echo("*" * 100)

    # Save to file if requested
    if save_file:
        with open(save_file, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Result saved to {save_file}")


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to the single YAML config file.")
@click.option("--agent_key", required=True, type=str, help="Key for the agent in agent_config (e.g., 'extractor_agent').")
@click.option("--task_key", required=True, type=str, help="Key for the task in task_config (e.g., 'extraction_task').")
@click.option(
    "--source",
    type=click.Path(exists=True),
    help="Path to the file to process (PDF, CSV or TXT). Alternative to --source_text.",
)
@click.option(
    "--source_text",
    type=str,
    help="Text string to use as input directly. Alternative to --source.",
)
@click.option("--api_key", required=False, type=str, help="Open router API key.")
@click.option("--env_file", required=False, type=click.Path(exists=True), help="Optional path to an environment file.")
@click.option("--save_file", required=False, type=str, help="Optional path to save the result as a JSON file.")
@click.option("--chunk_size", required=False, type=int, default=None, help="Chunk size in characters for extraction (None = no chunking).")
@click.option("--max_workers", required=False, type=int, default=None, help="Maximum parallel workers (None = auto).")
@click.option(
    "--enable_chunking",
    required=False,
    default=False,
    is_flag=True,
    help="Enable chunking (uses default chunk_size if --chunk_size not provided).",
)
@click.option(
    "--downstream_max_input_chars",
    required=False,
    type=int,
    default=None,
    help="Max input chars for alignment/judge/humanfeedback when running pipeline (default 80000).",
)
@click.option(
    "--max_extraction_chunk_chars",
    required=False,
    type=int,
    default=None,
    help="Cap extraction chunk size in chars for model context (default 25000). None = no cap.",
)
@click.option(
    "--agent_max_iter",
    required=False,
    type=int,
    default=None,
    help=("Maximum reasoning iterations per agent run before forced answer. " "Can also be set via env var AGENT_MAX_ITER=<int>."),
)
@click.option(
    "--agent_max_execution_time",
    required=False,
    type=int,
    default=None,
    help=(
        "Maximum wall-clock seconds per agent run before forced answer. Default: 30s. "
        "Can also be set via env var AGENT_MAX_EXECUTION_TIME=<int>."
    ),
)
@click.option(
    "--agent_max_retry_limit",
    required=False,
    type=int,
    default=None,
    help=("Maximum agent-level retries on recoverable errors. Default: 0. " "Can also be set via env var AGENT_MAX_RETRY_LIMIT=<int>."),
)
def run_agent(
    config,
    agent_key,
    task_key,
    source,
    source_text,
    api_key,
    env_file,
    save_file,
    chunk_size,
    max_workers,
    enable_chunking,
    downstream_max_input_chars,
    max_extraction_chunk_chars,
    agent_max_iter,
    agent_max_execution_time,
    agent_max_retry_limit,
):
    """Run a specific agent-task combination directly with full control.

    This command gives you direct control over how each agent runs without
    using the default flow pattern. You can specify exactly which agent
    and task to run, with custom chunking and parallel processing settings.
    """
    if source and source_text:
        raise click.UsageError("Please provide either --source or --source_text, not both.")
    elif not source and not source_text:
        raise click.UsageError("Please provide either --source or --source_text.")
    # Load environment variables
    if env_file:
        load_dotenv(env_file, override=True)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        load_dotenv()
        logger.info("Loaded environment variables from default .env")

    # Set API key if provided
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        logger.info("Set OPENROUTER_API_KEY in environment")

    # Load the config file
    all_config = load_config(config, "all")

    # Extract the different config sections
    agent_config = all_config.get("agent_config", {})
    embedder_config = all_config.get("embedder_config", {})
    task_config = all_config.get("task_config", {})
    knowledge_config = all_config.get("knowledge_config", {})

    # Replace API key if provided
    if api_key:
        from utils.utils import replace_api_key

        agent_config = replace_api_key(agent_config, api_key)
        embedder_config = replace_api_key(embedder_config, api_key)

    # Initialize the flow
    flow = StructSenseFlow(
        agent_config=agent_config,
        task_config=task_config,
        embedder_config=embedder_config,
        knowledge_config=knowledge_config,
        source=source,
        source_text=source_text,
        env_file=env_file,
        api_key=api_key,
        enable_human_feedback=False,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        max_workers=max_workers,
        downstream_max_input_chars=downstream_max_input_chars,
        max_extraction_chunk_chars=max_extraction_chunk_chars,
        agent_max_iter=agent_max_iter,
        agent_max_execution_time=agent_max_execution_time,
        agent_max_retry_limit=agent_max_retry_limit,
    )

    # Run the specific agent-task directly
    result = asyncio.run(
        flow.run_agent_task(
            agent_key=agent_key,
            task_key=task_key,
            chunk_size=chunk_size if chunk_size else (flow.chunk_size if enable_chunking else None),
            max_workers=max_workers,
            pydantic_output_class=None,
        )
    )

    # Output results
    click.echo("*" * 100)
    click.echo(f"Result for agent '{agent_key}' / task '{task_key}'")
    click.echo(f"Elapsed time: {result.get('elapsed_time', 'N/A')} seconds")
    click.echo(f"Errors: {len(result.get('errors', []))}")
    click.echo("*" * 100)

    # Pretty print results
    import json

    click.echo(json.dumps(result, indent=2, default=str))

    # Save to file if requested
    if save_file:
        with open(save_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"\nResult saved to {save_file}")


if __name__ == "__main__":
    cli()
