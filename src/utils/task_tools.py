# -*- coding: utf-8 -*-
"""
Resolve task-type tool names to CrewAI tool instances.

Defines which tools are used at which stage (agent) and for which task type:
  - GENERIC_TOOLS_BY_STAGE: agent_key -> list of tool names (applied to all tasks for that agent)
  - TOOLS_BY_STAGE_AND_TASK: (agent_key, task_type) -> list of task-specific tool names
  - get_tools_for_agent(agent_key, task_type): returns generic + task-specific, resolved

Final tools for (agent, task) = generic_tools(agent) + task_specific_tools(agent, task).
"""
import logging
from typing import Any, Dict, List

from .task_detection import get_tool_names_for_task_type, TOOLS_BY_TASK_TYPE

logger = logging.getLogger(__name__)

# ----------------------------
# Generic tools per stage (agent)
# ----------------------------
# Tools that apply to ALL task types for this agent. Combined with task-specific
# tools when resolving; order is generic first, then task-specific (deduped by name).
GENERIC_TOOLS_BY_STAGE: Dict[str, List[str]] = {
    "extractor_agent": [],
    "alignment_agent": [],
    "judge_agent": [],
    "human_feedback": [],
}

# ----------------------------
# Task-specific tools by stage (agent) and task type
# ----------------------------
# Matrix: agent_key -> task_type -> list of tool names.
# Different stages use different tools; different task types within a stage can too.
#
# Current mapping:
#   Stage               | Task types with tools     | Task-specific tools
#   --------------------|---------------------------|----------------------
#   extractor_agent     | ner, keyphrase_extraction | extract_ner_terms
#   extractor_agent     | extraction, resource, ... | (none)
#   alignment_agent     | (all)                     | (none)
#   judge_agent         | (all)                     | (none)
#   human_feedback      | (all)                     | (none)
#
# Generic tools (if any) are in GENERIC_TOOLS_BY_STAGE and apply to all tasks for that agent.
TOOLS_BY_STAGE_AND_TASK: Dict[str, Dict[str, List[str]]] = {
    "extractor_agent": TOOLS_BY_TASK_TYPE,
    "alignment_agent": {},
    "judge_agent": {},
    "human_feedback": {},
}
# Stages that may have tools (subset of keys above; empty = no tools for any task).
AGENTS_THAT_USE_TOOLS = frozenset(TOOLS_BY_STAGE_AND_TASK.keys())

# Lazy resolution: name -> CrewAI tool instance (avoids circular imports)
_TOOL_REGISTRY: dict = {}


def _resolve_tool(name: str) -> Any:
    """Resolve a tool name to a CrewAI tool instance (lazy import)."""
    global _TOOL_REGISTRY
    if name not in _TOOL_REGISTRY:
        if name == "extract_ner_terms":
            from .tools import extract_ner_terms
            _TOOL_REGISTRY[name] = extract_ner_terms
        else:
            logger.warning(f"Unknown tool name '{name}', skipping")
            return None
    return _TOOL_REGISTRY.get(name)


def get_tools_for_task_type(task_type: str) -> List[Any]:
    """
    Return the list of CrewAI tools to attach to the agent for the given task type.

    Tool names come from task_detection.TOOLS_BY_TASK_TYPE (taxonomy-aligned).
    Only task types listed there get tools; all others return an empty list.

    Use get_tools_for_agent(agent_key, task_type) when initializing agents so that
    tools are only attached for the extractor stage; alignment, judge, human_feedback
    get no tools.

    Args:
        task_type: Detected task type from taxonomy (e.g. from detect_task_type).

    Returns:
        List of tool instances to pass to the agent. Empty list if no tools.
    """
    names = get_tool_names_for_task_type(task_type)
    if not names:
        logger.info(f"Task type '{task_type}' -> no tools")
        return []
    tools = []
    for n in names:
        t = _resolve_tool(n)
        if t is not None:
            tools.append(t)
    logger.info(f"Task type '{task_type}' -> attaching tools: {names}")
    return tools


def get_tool_names_for_agent(agent_key: str, task_type: str) -> List[str]:
    """
    Return tool names for the given stage (agent_key) and task type.

    Combines generic tools (GENERIC_TOOLS_BY_STAGE[agent_key]) with
    task-specific tools (TOOLS_BY_STAGE_AND_TASK[agent_key][task_type]).
    Order: generic first, then task-specific; duplicates by name are skipped.

    Args:
        agent_key: Key for the agent (e.g. extractor_agent, alignment_agent).
        task_type: Detected task type from taxonomy.

    Returns:
        List of tool names; empty list if this (stage, task_type) has no tools.
    """
    if not agent_key:
        return []
    seen = set()
    out: List[str] = []
    # Generic tools for this agent (apply to all tasks)
    for name in GENERIC_TOOLS_BY_STAGE.get(agent_key, []):
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    # Task-specific tools
    if task_type:
        stage_map = TOOLS_BY_STAGE_AND_TASK.get(agent_key, {})
        key = str(task_type).strip().lower()
        for name in stage_map.get(key, []):
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def get_tools_for_agent(agent_key: str, task_type: str) -> List[Any]:
    """
    Return resolved tool instances for the given stage (agent_key) and task type.

    Combines generic tools (GENERIC_TOOLS_BY_STAGE) and task-specific tools
    (TOOLS_BY_STAGE_AND_TASK), then resolves names to CrewAI tool instances.
    E.g. extractor + ner -> generic_tools(extractor) + [extract_ner_terms].

    Args:
        agent_key: Key for the agent in agent_config (e.g. extractor_agent, alignment_agent).
        task_type: Detected task type from taxonomy.

    Returns:
        List of tool instances; empty list if this (stage, task_type) has no tools.
    """
    names = get_tool_names_for_agent(agent_key, task_type)
    if not names:
        logger.info(f"Stage '{agent_key}' + task '{task_type}' -> no tools")
        return []
    tools = []
    for n in names:
        t = _resolve_tool(n)
        if t is not None:
            tools.append(t)
    logger.info(f"Stage '{agent_key}' + task '{task_type}' -> tools: {names}")
    return tools
