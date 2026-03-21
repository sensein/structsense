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

# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : downstream_agent_helper.py
# @Software: PyCharm

"""
Helper functions for preparing context for downstream agents.

Handles token limit management and intelligent context compression
for alignment, judge, and human feedback agents.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .agent_context import AgentContext
from .context_window_manager import ContextWindowManager

logger = logging.getLogger(__name__)


def prepare_alignment_agent_input(
    extraction_results: Dict[str, Any],
    original_text: str,
    agent_context: Optional[AgentContext] = None,
    context_manager: Optional[ContextWindowManager] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Prepare input for alignment agent.

    Passes the full extraction output without any truncation or compression so the
    alignment agent always receives every entity and key term extracted.

    Args:
        extraction_results: Results from extraction agent (previous agent). Required.
        original_text: Original input text
        agent_context: Optional AgentContext for accessing extraction metadata
        context_manager: Unused — kept for API compatibility
        max_tokens: Unused — kept for API compatibility

    Returns:
        Prepared input dictionary for alignment agent
    """
    entity_count = len(extraction_results.get("entities") or []) if isinstance(extraction_results, dict) else 0
    logger.info(f"Preparing alignment agent input: {entity_count} entities (no truncation)")

    alignment_input = {
        "extracted_structured_information": extraction_results,
        "original_text": original_text,
        "compression_applied": False,
    }

    if agent_context:
        extraction_metadata = agent_context.get_latest_result("extractor_agent")
        if extraction_metadata:
            alignment_input["extraction_confidence"] = extraction_metadata.confidence
            alignment_input["extraction_metadata"] = extraction_metadata.metadata

    return alignment_input


def prepare_judge_agent_input(
    alignment_results: Dict[str, Any],
    extraction_results: Optional[Dict[str, Any]] = None,
    agent_context: Optional[AgentContext] = None,
    context_manager: Optional[ContextWindowManager] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Prepare input for judge agent.

    Passes the full alignment output without any truncation or compression so the
    judge agent always receives every entity aligned by the previous stage.
    Merges extraction results into alignment output so the judge sees the complete picture.

    Args:
        alignment_results: Results from alignment agent (previous agent). Required.
        extraction_results: Optional original extraction results (merged in to extend alignment).
        agent_context: Optional AgentContext
        context_manager: Unused — kept for API compatibility
        max_tokens: Unused — kept for API compatibility

    Returns:
        Prepared input dictionary for judge agent
    """
    # Merge extraction into alignment so judge receives all entities (extraction + alignment additions)
    if extraction_results and isinstance(extraction_results, dict):
        full_alignment = _extend_previous_stage(
            extraction_results, alignment_results,
            list_keys=["entities", "key_terms", "resources", "aligned_resources", "judge_resource"],
        )
    else:
        full_alignment = alignment_results

    entity_count = len(full_alignment.get("entities") or []) if isinstance(full_alignment, dict) else 0
    logger.info(f"Preparing judge agent input: {entity_count} entities (no truncation)")

    judge_input = {
        "aligned_structured_information": full_alignment,
        "compression_applied": False,
    }

    if agent_context:
        alignment_metadata = agent_context.get_latest_result("alignment_agent")
        if alignment_metadata:
            judge_input["alignment_confidence"] = alignment_metadata.confidence

    return judge_input


def prepare_humanfeedback_agent_input(
    judge_results: Dict[str, Any],
    user_feedback: str,
    alignment_results: Optional[Dict[str, Any]] = None,
    agent_context: Optional[AgentContext] = None,
    context_manager: Optional[ContextWindowManager] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Prepare input for human feedback agent with token limit management.

    Human feedback always receives the previous agent's (judge) output plus human input. No exception.

    Args:
        judge_results: Results from judge agent (previous agent). Required.
        user_feedback: Human feedback text. Required when running the agent.
        alignment_results: Optional alignment output; used by helper to extend judge when needed
        agent_context: Optional AgentContext
        context_manager: Optional ContextWindowManager
        max_tokens: Maximum tokens

    Returns:
        Prepared input dictionary for human feedback agent
    """
    if context_manager is None:
        context_manager = ContextWindowManager(max_tokens=max_tokens or 100000)

    if max_tokens is None:
        max_tokens = 100000

    # Merge alignment into judge so human feedback receives all entities (alignment + judge additions)
    if alignment_results and isinstance(alignment_results, dict):
        full_judge = _extend_previous_stage(
            alignment_results, judge_results,
            list_keys=["entities", "key_terms", "resources", "aligned_resources", "judge_resource"],
        )
    else:
        full_judge = judge_results

    entity_count = len(full_judge.get("entities") or []) if isinstance(full_judge, dict) else 0
    logger.info(f"Preparing humanfeedback agent input: {entity_count} entities (no truncation)")

    humanfeedback_input = {
        "judged_structured_information_with_human_feedback": full_judge,
        "user_feedback_text": user_feedback,
        "modification_context": user_feedback,
        "compression_applied": False,
    }

    if agent_context:
        judge_metadata = agent_context.get_latest_result("judge_agent")
        if judge_metadata:
            humanfeedback_input["judge_confidence"] = judge_metadata.confidence

    return humanfeedback_input


def split_structured_payload(
    payload: Dict[str, Any],
    list_keys: Optional[List[str]] = None,
    max_entities_per_chunk: int = 70,
    max_key_terms_per_chunk: int = 25,
    max_resources_per_chunk: int = 15,
    # Kept for API compatibility — no longer used for logic
    context_manager=None,
    max_tokens_per_chunk=None,
) -> List[Dict[str, Any]]:
    """
    Split a structured payload (entities, key_terms, resources) into entity-count chunks.
    Each chunk gets a slice of every list key so all parallel jobs receive roughly equal work.
    Run agents on each chunk in parallel with asyncio.gather, then merge results.
    """
    list_keys = list_keys or ["entities", "key_terms", "resources", "aligned_resources", "judge_resource"]
    base = {k: v for k, v in payload.items() if k not in list_keys or not isinstance(v, list)}
    lists = {k: list(payload[k]) if isinstance(payload.get(k), list) else [] for k in list_keys}
    n_entities = len(lists.get("entities", []))
    n_key_terms = len(lists.get("key_terms", []))
    n_resources = len(lists.get("resources", []))
    n_aligned = len(lists.get("aligned_resources", []))
    n_judge = len(lists.get("judge_resource", []))

    if n_entities == 0 and n_key_terms == 0 and n_resources == 0 and n_aligned == 0 and n_judge == 0:
        return [payload]

    n_chunks = max(
        1,
        (n_entities + max_entities_per_chunk - 1) // max_entities_per_chunk if n_entities else 1,
        (n_key_terms + max_key_terms_per_chunk - 1) // max_key_terms_per_chunk if n_key_terms else 1,
        (n_resources + max_resources_per_chunk - 1) // max_resources_per_chunk if n_resources else 1,
        (n_aligned + max_resources_per_chunk - 1) // max_resources_per_chunk if n_aligned else 1,
        (n_judge + 4) // 5 if n_judge else 1,
    )
    chunks: List[Dict[str, Any]] = []
    for i in range(n_chunks):
        chunk = dict(base)
        chunk["_chunk_index"] = i
        chunk["_chunk_total"] = n_chunks
        if "entities" in list_keys and lists.get("entities"):
            start = i * max_entities_per_chunk
            chunk["entities"] = lists["entities"][start : start + max_entities_per_chunk]
        if "key_terms" in list_keys and lists.get("key_terms"):
            start = i * max_key_terms_per_chunk
            chunk["key_terms"] = lists["key_terms"][start : start + max_key_terms_per_chunk]
        if "resources" in list_keys and lists.get("resources"):
            start = i * max_resources_per_chunk
            chunk["resources"] = lists["resources"][start : start + max_resources_per_chunk]
        if "aligned_resources" in list_keys and lists.get("aligned_resources"):
            start = i * max_resources_per_chunk
            chunk["aligned_resources"] = lists["aligned_resources"][start : start + max_resources_per_chunk]
        if "judge_resource" in list_keys and lists.get("judge_resource"):
            start = i * 5
            chunk["judge_resource"] = lists["judge_resource"][start : start + 5]
        chunks.append(chunk)
    logger.info(
        "Split structured payload into %d chunks (~%d entities/chunk)",
        len(chunks),
        max_entities_per_chunk,
    )
    return chunks


def _extract_entities_from_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the largest available entity list from a single chunk result.

    An LLM chunk output may have entities under several keys simultaneously:
      - "entities"          canonical list (may be partial — LLM wrote a few directly)
      - "aligned_ner_terms" dict-of-lists from alignment stage
      - "extracted_terms"   dict-of-lists from extraction stage
      - "judge_ner_terms"   dict-of-lists from judge stage

    Design principle: downstream stages only enrich, never drop.  We always take
    the largest available set so no entities are silently lost during chunked merges.
    """
    from utils.postprocessing import _flatten_container_to_list  # avoid circular at module level
    best: List[Dict[str, Any]] = chunk.get("entities") or []
    for raw_key in ("aligned_ner_terms", "extracted_terms", "judge_ner_terms"):
        container = chunk.get(raw_key)
        if container:
            promoted = _flatten_container_to_list(container)
            if len(promoted) > len(best):
                best = promoted
    return best


def merge_structured_chunk_results(
    chunk_results: List[Dict[str, Any]],
    list_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merge results from chunked agent runs: concatenate entities, key_terms, resources, etc.
    Keep scalar fields (judge_score, remarks, confidence) from last chunk or merge appropriately.

    Entities are collected via _extract_entities_from_chunk so that stage-specific keys
    (aligned_ner_terms, extracted_terms, judge_ner_terms) are included even when the LLM
    only partially populated the canonical "entities" list per chunk.
    """
    list_keys = list_keys or ["entities", "key_terms", "resources", "aligned_resources", "judge_resource"]
    if not chunk_results:
        return {}
    if len(chunk_results) == 1:
        out = dict(chunk_results[0])
        # Normalise single-chunk entities using the same "take largest" logic.
        out["entities"] = _extract_entities_from_chunk(out)
        out.pop("_chunk_index", None)
        out.pop("_chunk_total", None)
        return out

    merged: Dict[str, Any] = {}
    for key in list_keys:
        merged[key] = []
    for result in chunk_results:
        if not isinstance(result, dict):
            continue
        # Use the largest available entity set for this chunk (may come from
        # aligned_ner_terms / extracted_terms rather than the canonical "entities" key).
        merged["entities"].extend(_extract_entities_from_chunk(result))
        for key in [k for k in list_keys if k != "entities"]:
            val = result.get(key)
            if isinstance(val, list):
                merged[key].extend(val)
        for key, value in result.items():
            if key in list_keys or key.startswith("_chunk"):
                continue
            if key not in merged:
                merged[key] = value
            elif isinstance(value, (list, dict)) and not isinstance(merged.get(key), (list, dict)):
                merged[key] = value

    # Dedupe key_terms: by string value or by dict["term"] (keep order)
    if "key_terms" in merged and merged["key_terms"]:
        seen = set()
        deduped = []
        for t in merged["key_terms"]:
            key = (t if isinstance(t, str) else (t.get("term") if isinstance(t, dict) else None)) or ""
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(t)
        merged["key_terms"] = deduped

    merged.pop("_chunk_index", None)
    merged.pop("_chunk_total", None)
    logger.info("Merged %s chunk results", len(chunk_results))
    return merged


def _cap_result_for_context(
    result: Dict[str, Any],
    max_entities: int = 80,
    max_key_terms: int = 30,
    max_resources: int = 20,
    stage: str = "",
) -> Dict[str, Any]:
    """
    Cap list lengths so payload stays within model context (e.g. 128k tokens).
    Use before compressing for human feedback when extraction has hundreds of entities.
    """
    out = dict(result)
    caps = {
        "entities": max_entities,
        "key_terms": max_key_terms,
        "resources": max_resources,
        "aligned_resources": max_resources,
        "judge_resource": 5,
    }
    for key, limit in caps.items():
        if key not in out or not isinstance(out[key], list):
            continue
        lst = out[key]
        if len(lst) <= limit:
            continue
        out[key] = lst[:limit]
        out[f"{key}_truncated_for_context"] = True
        out[f"{key}_original_count"] = len(lst)
        logger.info(
            "[%s] Capped %s to %s items (was %s) to fit context limit",
            stage, key, limit, len(lst),
        )
    return out


def _extend_previous_stage(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    list_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Merge so that current stage output extends previous: use current's values where
    present and non-empty, else fall back to previous. Ensures judge receives
    extraction extended by alignment, and human feedback receives alignment extended by judge.
    """
    list_keys = list_keys or ["entities", "key_terms", "resources", "aligned_resources", "judge_resource"]
    out = dict(previous)
    for key, value in current.items():
        if key in list_keys and isinstance(value, list):
            # Use current's list if non-empty, else keep previous's
            if value:
                out[key] = value
            elif key in previous and isinstance(previous[key], list) and previous[key]:
                out[key] = previous[key]
            else:
                out[key] = value
        elif value is not None:
            # Non-list keys: current extends previous (judge_score, remarks, etc.)
            out[key] = value
    return out


def _ensure_list_keys_preserved(
    compressed: Dict[str, Any],
    original: Dict[str, Any],
    list_keys: List[str],
    caps: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Ensure compressed result keeps list keys (entities, key_terms, etc.) so downstream agents
    never receive empty when the previous stage had content. Alignment extends with concept mapping;
    judge extends with judgement; both expect these keys as non-empty lists when upstream had data.
    """
    caps = caps or {}
    out = dict(compressed)
    for key in list_keys:
        comp_val = out.get(key)
        orig_val = original.get(key) if isinstance(original, dict) else None
        if isinstance(orig_val, list) and not orig_val:
            continue
        if isinstance(comp_val, list) and comp_val:
            continue
        sample = out.get(f"{key}_sample")
        if isinstance(sample, list) and sample:
            out[key] = sample
            continue
        if isinstance(orig_val, list) and orig_val:
            cap = caps.get(key, 50)
            out[key] = orig_val[:cap]
            if len(orig_val) > cap:
                out[f"{key}_truncated"] = True
                out[f"{key}_original_count"] = len(orig_val)
        elif key not in out:
            out[key] = []
    return out


def _ensure_judge_input_structure(
    compressed: Dict[str, Any],
    original: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure compressed alignment has the list keys the judge expects (entities, key_terms, etc.).
    Judge extends aligned result with judgement; never pass empty when alignment had content.
    """
    return _ensure_list_keys_preserved(
        compressed,
        original,
        list_keys=["entities", "key_terms", "resources", "aligned_resources", "judge_resource"],
        caps={"entities": 50, "key_terms": 20, "resources": 20},
    )


def _extract_essentials(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only essential information from results (aggressive truncation).

    Keeps:
    - entities (top 10)
    - resources (top 5)
    - key_terms (top 10)
    - aligned_resources (top 5)
    - judge_resource (first item)
    - confidence scores

    Args:
        results: Results dictionary

    Returns:
        Dictionary with essential information only
    """
    essentials = {}

    # Extract top items from lists
    list_limits = {
        "entities": 10,
        "resources": 5,
        "key_terms": 10,
        "aligned_resources": 5,
        "judge_resource": 1,
    }

    for key, limit in list_limits.items():
        if key in results and isinstance(results[key], list):
            essentials[key] = results[key][:limit]
            if len(results[key]) > limit:
                essentials[f"{key}_truncated"] = True
                essentials[f"{key}_original_count"] = len(results[key])

    # Keep scalar values
    scalar_keys = ["confidence", "score", "status", "summary"]
    for key in scalar_keys:
        if key in results and not isinstance(results[key], (list, dict)):
            essentials[key] = results[key]

    essentials["_aggressively_truncated"] = True

    return essentials


def _create_extraction_summary(extraction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a concise summary of extraction results.

    Args:
        extraction_results: Full extraction results

    Returns:
        Summarized extraction results
    """
    summary = {}

    # Count items
    if "entities" in extraction_results:
        entities = extraction_results["entities"]
        summary["entity_count"] = len(entities)
        # Group by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        summary["entity_types"] = entity_types

    if "resources" in extraction_results:
        resources = extraction_results["resources"]
        summary["resource_count"] = len(resources)

    if "key_terms" in extraction_results:
        key_terms = extraction_results["key_terms"]
        summary["key_term_count"] = len(key_terms)

    # Add confidence
    if "confidence" in extraction_results:
        summary["extraction_confidence"] = extraction_results["confidence"]

    return summary
