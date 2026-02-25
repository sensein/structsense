# -*- coding: utf-8 -*-
"""Resolve task-type tool names to CrewAI tool instances for the pipeline.

This module defines which tools are attached to which agent and task type.
Tools are resolved lazily (e.g. NER tool created with domain context when
agent_config/task_config/task_key are provided).

Module-level mappings (for generated docs)
------------------------------------------
- **GENERIC_TOOLS_BY_STAGE** (dict): ``agent_key`` → list of tool names applied to
  all task types for that agent (e.g. alignment_agent → ``["concept_mapping_tool"]``).
- **TOOLS_BY_STAGE_AND_TASK** (dict): ``agent_key`` → ``task_type`` → list of
  task-specific tool names. E.g. extractor_agent + ``ner`` → ``["extract_ner_terms"]``.
- **AGENTS_THAT_USE_TOOLS** (frozenset): Agent keys that may have tools (others get []).

Key functions
-------------
- :func:`get_tools_for_agent` – Return resolved tool instances for a given
  agent and task type (used by :class:`structsense.app.StructSenseFlow`).
- :func:`get_tool_names_for_agent` – Return tool names only (generic + task-specific).
- :func:`get_tools_for_task_type` – Return tools by task type only (no agent filter).
- :func:`_resolve_tool` – Resolve a single tool name to a CrewAI tool instance;
  for ``extract_ner_terms``, pass agent_config/task_config/agent_key/task_key for
  domain-aware LLM NER.

See Also
--------
- :mod:`task_detection` – Defines :data:`TOOLS_BY_TASK_TYPE` used here.
- :mod:`structsense.app` – Uses :func:`get_tools_for_agent` when initializing agents.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from .task_detection import get_tool_names_for_task_type, TOOLS_BY_TASK_TYPE

logger = logging.getLogger(__name__)

# ----------------------------
# Generic tools per stage (agent)
# ----------------------------
# Tools that apply to ALL task types for this agent. Combined with task-specific
# tools when resolving; order is generic first, then task-specific (deduped by name).
GENERIC_TOOLS_BY_STAGE: Dict[str, List[str]] = {
    "extractor_agent": ["repair_json"],
    "alignment_agent": ["concept_mapping_tool", "repair_json"],
    "judge_agent": ["repair_json"],
    "human_feedback": ["repair_json"],
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


def _resolve_tool(
    name: str,
    agent_config: Optional[Dict[str, Any]] = None,
    task_config: Optional[Dict[str, Any]] = None,
    agent_key: Optional[str] = None,
    task_key: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Any:
    """Resolve a tool name to a CrewAI tool instance (lazy import).

    For ``extract_ner_terms``, when agent_config, task_config, agent_key, and
    task_key are provided, returns a domain-aware NER tool that uses the
    extractor agent's role, goal, and task description for LLM-based NER.
    Otherwise returns the default (ML-only) NER tool. For ``concept_mapping_tool``,
    returns :class:`.conceptmappingtool.ConceptMappingTool` (requires BIOPORTAL_API_KEY).
    For ``repair_json``, sets LLM context and default schema from task_type when provided.

    Parameters
    ----------
    name : str
        Tool name (e.g. ``extract_ner_terms``, ``concept_mapping_tool``).
    agent_config : dict, optional
        Agent config dict; used for domain-aware NER when name is ``extract_ner_terms``.
    task_config : dict, optional
        Task config dict; used for task description in NER context.
    agent_key : str, optional
        Agent key (e.g. ``extractor_agent``); used to read role/goal from agent_config.
    task_key : str, optional
        Task key (e.g. ``extraction_task``); used to read description from task_config.
    task_type : str, optional
        Task type (e.g. ``ner``, ``resource``); used to set default schema for repair_json.

    Returns
    -------
    object or None
        CrewAI-compatible tool instance, or None if unknown name or (for concept_mapping_tool)
        registration failed (e.g. missing API key).
    """
    global _TOOL_REGISTRY
    if name == "extract_ner_terms":
        if agent_config and task_config and agent_key and task_key:
            from .ner_tool import set_ner_domain_context, extract_ner_terms
            agent_cfg = agent_config.get(agent_key) or {}
            task_cfg = task_config.get(task_key)
            agent_role = (agent_cfg.get("role") or "").strip()
            agent_goal = (agent_cfg.get("goal") or "").strip()
            task_description = ""
            if isinstance(task_cfg, dict):
                task_description = (task_cfg.get("description") or "").strip()
            else:
                task_description = str(task_cfg or "").strip()
            llm_config = agent_cfg.get("llm") or {}
            api_key = (llm_config.get("api_key") or os.environ.get("OPENROUTER_API_KEY") or
                       os.environ.get("OPENAI_API_KEY"))
            set_ner_domain_context(
                agent_role=agent_role,
                agent_goal=agent_goal,
                task_description=task_description,
                llm_config=llm_config,
                api_key=api_key,
                enable_llm_ner=True,
            )
        if name not in _TOOL_REGISTRY:
            from .ner_tool import extract_ner_terms
            _TOOL_REGISTRY[name] = extract_ner_terms
        return _TOOL_REGISTRY.get(name)
    if name == "concept_mapping_tool":
        if name not in _TOOL_REGISTRY:
            try:
                from .conceptmappingtool import ConceptMappingTool
                _TOOL_REGISTRY[name] = ConceptMappingTool()
            except ValueError as e:
                logger.warning(f"ConceptMappingTool not registered (missing BIOPORTAL_API_KEY): {e}")
                return None
        return _TOOL_REGISTRY.get(name)
    if name == "repair_json":
        from .json_repair_tool import (
            get_default_schema_for_task_type,
            repair_json_tool,
            set_repair_json_default_schema,
            set_repair_json_llm_context,
        )
        if agent_config and agent_key:
            agent_cfg = agent_config.get(agent_key) or {}
            llm_config = agent_cfg.get("llm") or {}
            api_key = (
                llm_config.get("api_key")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
            set_repair_json_llm_context(llm_config=llm_config, api_key=api_key)
        if task_type:
            set_repair_json_default_schema(get_default_schema_for_task_type(task_type))
        if name not in _TOOL_REGISTRY:
            _TOOL_REGISTRY[name] = repair_json_tool
        return _TOOL_REGISTRY.get(name)
    if name not in _TOOL_REGISTRY:
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


def get_tools_for_agent(
    agent_key: str,
    task_type: str,
    agent_config: Optional[Dict[str, Any]] = None,
    task_config: Optional[Dict[str, Any]] = None,
    task_key: Optional[str] = None,
) -> List[Any]:
    """
    Return resolved tool instances for the given stage (agent_key) and task type.

    Combines generic tools (GENERIC_TOOLS_BY_STAGE) and task-specific tools
    (TOOLS_BY_STAGE_AND_TASK), then resolves names to CrewAI tool instances.
    E.g. extractor + ner -> generic_tools(extractor) + [extract_ner_terms].

    When agent_config, task_config, and task_key are provided, the NER tool
    (if used) is created with domain context so LLM-based NER can use the
    extractor agent's role and task description.

    Args:
        agent_key: Key for the agent in agent_config (e.g. extractor_agent, alignment_agent).
        task_type: Detected task type from taxonomy.
        agent_config: Optional agent config dict (for domain-aware NER tool).
        task_config: Optional task config dict (for domain-aware NER tool).
        task_key: Optional task key (e.g. extraction_task) for task description.

    Returns
    -------
    list
        Resolved CrewAI tool instances; empty list if this (stage, task_type) has no tools.

    See Also
    --------
    get_tool_names_for_agent : Tool names only (no resolution).
    _resolve_tool : Resolve a single tool name with optional domain context.
    """
    names = get_tool_names_for_agent(agent_key, task_type)
    if not names:
        logger.info(f"Stage '{agent_key}' + task '{task_type}' -> no tools")
        return []
    tools = []
    for n in names:
        t = _resolve_tool(
            n,
            agent_config=agent_config,
            task_config=task_config,
            agent_key=agent_key,
            task_key=task_key,
            task_type=task_type,
        )
        if t is not None:
            tools.append(t)
    logger.info(f"Stage '{agent_key}' + task '{task_type}' -> tools: {names}")
    return tools
