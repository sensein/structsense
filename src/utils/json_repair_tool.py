# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------

"""Tool to repair malformed JSON from LLM output.
Two stage repair: first json_repair library (no LLM). Handles trailing commas, single quotes,
markdown fences, Python booleans/None, truncated JSON.


Second stage, when json_repair library fails: Uses TrustCall with the calling agent's LLM config.
Schema comes from (in order): (1) agent-provided ``schema`` argument,
(2) default from task type (task_tools._resolve_tool), (3) **inferred from the
malformed input** (top-level keys and types). No hardcoded generic shape—unknown
tasks use the structure found in the input to build the schema and fix. If
extraction fails, we retry with the agent's schema when the agent passed one.

Add to poetry or install:
pip install json-repair
Optional for second stage: pip install trustcall and either langchain-openrouter or langchain-openai
"""

import json
import re
import logging
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from crewai.tools import tool
from json_repair import repair_json as jr_repair

logger = logging.getLogger(__name__)

# LLM context for TrustCall fallback (set from agent config when tool is resolved)
_repair_llm_context: Optional[Dict[str, Any]] = None
# Default schema hint for Layer 2 when agent does not pass schema (set from task type at resolve time)
_repair_default_schema: Optional[str] = None

# Default JSON schema hints by pipeline task type (for TrustCall Layer 2 when agent omits schema).
# Unknown task types do not use a hardcoded fallback; schema is inferred from the malformed input instead.
DEFAULT_SCHEMAS_BY_TASK: Dict[str, str] = {
    "ner": '{"entities": "array", "key_terms": "array", "resources": "array"}',
    "resource": '{"resources": "array"}',
    "structured_extraction": '{"resources": "array", "entities": "array", "key_terms": "array"}',
    "keyphrase_extraction": '{"key_terms": "array", "entities": "array"}',
    "relation_extraction": '{"entities": "array", "relations": "array"}',
}


def _infer_schema_from_raw(raw: str) -> Optional[str]:
    """Infer a minimal JSON schema hint from the malformed input (top-level keys and types).

    Used when no agent schema and no task-type default; avoids hardcoded shapes so
    any task (custom keys) can be repaired. Returns e.g. '{"my_key": "array", "meta": "object"}'.
    """
    if not raw or not raw.strip():
        return None
    # 1) Try to parse as dict and infer from structure
    try:
        obj = jr_repair(raw, return_objects=True)
        if isinstance(obj, dict):
            hint: Dict[str, str] = {}
            for k, v in obj.items():
                key = str(k).strip()
                if not key:
                    continue
                if isinstance(v, list):
                    hint[key] = "array"
                elif isinstance(v, dict):
                    hint[key] = "object"
                else:
                    hint[key] = "string"
            if hint:
                return json.dumps(hint)
    except Exception:
        pass
    # 2) Regex fallback: find top-level "key": or 'key': and next token [ { or literal
    hint = {}
    # Match "key": or 'key': or key: (unquoted)
    for m in re.finditer(r"""["']?([^"'\s:{}]+)["']?\s*:\s*""", raw):
        key = (m.group(1) or "").strip()
        if not key or key in ("true", "false", "null"):
            continue
        rest = raw[m.end() : m.end() + 200].lstrip()
        if rest.startswith("["):
            hint[key] = "array"
        elif rest.startswith("{"):
            hint[key] = "object"
        elif rest.startswith('"') or rest.startswith("'"):
            hint[key] = "string"
        else:
            hint[key] = "string"
    if hint:
        return json.dumps(hint)
    return None


def get_default_schema_for_task_type(task_type: str) -> Optional[str]:
    """Return the default schema hint for TrustCall Layer 2 for the given task type.

    Unknown task types return None; the tool will then infer schema from the input (no hardcoded shape).
    """
    if not task_type:
        return None
    key = str(task_type).strip().lower()
    return DEFAULT_SCHEMAS_BY_TASK.get(key)


def set_repair_json_default_schema(schema: Optional[str] = None) -> None:
    """Set default schema hint for Layer 2 when the agent does not pass schema.

    Called from task_tools._resolve_tool when attaching repair_json with a task_type.
    """
    global _repair_default_schema
    _repair_default_schema = schema


def clear_repair_json_default_schema() -> None:
    """Clear the default schema hint."""
    global _repair_default_schema
    _repair_default_schema = None


def set_repair_json_llm_context(
    llm_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> None:
    """Set LLM config for TrustCall fallback so it uses the same model as the calling agent.

    Called from task_tools._resolve_tool when attaching repair_json to an agent.
    """
    global _repair_llm_context
    _repair_llm_context = (
        {"llm_config": llm_config or {}, "api_key": api_key}
        if (llm_config and (llm_config.get("model") or llm_config.get("base_url")))
        else None
    )


def clear_repair_json_llm_context() -> None:
    """Clear stored LLM context."""
    global _repair_llm_context
    _repair_llm_context = None


@tool("repair_json")
def repair_json_tool(malformed_json: str, schema: Optional[str] = None) -> str:
    """
    Repair malformed or broken JSON from LLM output.

    Layer 1: Lightweight parser (trailing commas, single quotes, markdown fences, etc.).
    Layer 2: If repair fails, uses TrustCall. Schema order: agent-provided > task-type
    default > inferred from malformed input (no hardcoded shape). Retries with
    agent's schema when the first attempt fails and the agent passed one.

    Args:
        malformed_json: The broken JSON string to repair.
        schema: Optional JSON schema hint. Takes precedence; used for retry if extraction with default/inferred fails.

    Returns:
        A valid JSON string, or {"error": "...", "raw_preview": "..."} if repair fails.
    """
    raw = malformed_json.strip()

    m = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    try:
        repaired = jr_repair(raw, return_objects=True)
        return json.dumps(repaired, indent=2)
    except Exception as e:
        logger.debug("json_repair failed: %s", e)

    # Layer 2 schema: agent > task-type default > inferred from input (no hardcoded "other").
    schema_hint = schema or _repair_default_schema or _infer_schema_from_raw(raw)
    if schema_hint and _repair_llm_context:
        try:
            return _trustcall_extract(raw, schema_hint)
        except Exception as e:
            logger.warning("TrustCall fallback failed with schema hint: %s", e)
            # No match or validation failed: retry with agent's schema when agent passed one.
            if schema and schema != schema_hint:
                try:
                    return _trustcall_extract(raw, schema)
                except Exception as e2:
                    logger.warning("TrustCall retry with agent schema failed: %s", e2)

    return json.dumps({"error": "Could not repair JSON", "raw_preview": raw[:500]})


def _trustcall_extract(raw: str, schema_hint: str) -> str:
    """Use TrustCall with the agent's LLM config (from set_repair_json_llm_context)."""
    global _repair_llm_context
    if not _repair_llm_context:
        raise ValueError("LLM context not set for repair_json; cannot use TrustCall fallback")

    try:
        from trustcall import create_extractor
    except ImportError as e:
        raise ValueError("TrustCall fallback requires: pip install trustcall") from e

    llm_config = _repair_llm_context.get("llm_config") or {}
    api_key = _repair_llm_context.get("api_key") or ""
    base_url = (llm_config.get("base_url") or "").strip().lower()
    model = llm_config.get("model") or "openai/gpt-4o-mini"
    if "openrouter" in base_url and isinstance(model, str) and model.startswith("openrouter/"):
        model = model.replace("openrouter/", "", 1)

    # Prefer OpenRouter client when config points to OpenRouter; else OpenAI-compatible (ChatOpenAI + base_url)
    if "openrouter" in base_url:
        try:
            from langchain_openrouter import ChatOpenRouter

            llm = ChatOpenRouter(
                model=model,
                temperature=0,
                **({"api_key": api_key} if api_key else {}),
            )
        except ImportError:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=llm_config.get("base_url") or "https://openrouter.ai/api/v1",
                temperature=0,
            )
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=llm_config.get("base_url") or None,
            temperature=0,
        )
    DynamicModel = _build_model_from_schema(schema_hint)
    extractor = create_extractor(llm, tools=[DynamicModel], tool_choice=DynamicModel.__name__)
    result = extractor.invoke(
        {"messages": [("user", f"Extract data matching this schema from the text.\nSchema: {schema_hint}\n\nText:\n{raw}")]}
    )
    if result.get("responses"):
        return result["responses"][0].model_dump_json(indent=2)
    raise ValueError("TrustCall returned no responses")


def _build_model_from_schema(schema_hint: str) -> type:
    """Build a Pydantic model from a JSON schema hint (e.g. {\"entities\": \"array\", \"key_terms\": \"array\"})."""

    try:
        hint = json.loads(jr_repair(schema_hint, return_objects=False))
    except Exception:
        hint = {}

    # Use typing so Pydantic/TrustCall can validate (list -> List[Any], dict -> Dict[str, Any])
    annotations: Dict[str, type] = {}
    defaults: Dict[str, Any] = {}
    for k, v in hint.items():
        val = str(v).lower()
        if val in ("list", "array"):
            annotations[k] = List[Any]
            defaults[k] = []
        elif val in ("dict", "object"):
            annotations[k] = Dict[str, Any]
            defaults[k] = {}
        elif val in ("string", "str"):
            annotations[k] = str
            defaults[k] = ""
        elif val in ("int", "integer"):
            annotations[k] = int
            defaults[k] = 0
        elif val in ("float", "number"):
            annotations[k] = float
            defaults[k] = 0.0
        elif val in ("bool", "boolean"):
            annotations[k] = bool
            defaults[k] = False
        else:
            annotations[k] = Any
            defaults[k] = None
    return type("ExtractedData", (BaseModel,), {"__annotations__": annotations, **defaults})
