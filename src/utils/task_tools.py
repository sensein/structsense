# -*- coding: utf-8 -*-
"""
Resolve task-type tool names to CrewAI tool instances.

Tool names and which task types use them are defined in task_detection
(TOOLS_BY_TASK_TYPE). This module only resolves names to instances so
task_detection does not need to import tools (avoids circular deps).
"""
import logging
from typing import Any, List

from .task_detection import get_tool_names_for_task_type

logger = logging.getLogger(__name__)

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
