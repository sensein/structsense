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
# @File    : postprocessing.py
# @Software: PyCharm

"""Task-specific post-processing and result merging for the pipeline.

This module provides post-processors (per-chunk cleanup/normalization) and
result mergers (combine chunk results into one structure with provenance).
The pipeline selects them by task type (ner, resource, extraction, etc.).

Key functions (for generated docs)
---------------------------------
- :func:`get_post_processor` – Return the post-processing function for a task type.
  Used after each chunk is processed by the extraction agent.
- :func:`get_result_merger` – Return the merging function for a task type.
  Used to combine chunk results (e.g. merge_ner_results, merge_resource_results).
- :func:`register_task_type` – Register a new task type with custom post-processor
  and merger (e.g. when adding a new use case;).
- :func:`get_registered_task_types` – List all registered task types.

Other important functions
-------------------------
- :func:`merge_downstream_chunk_results_with_provenance` – Merge downstream stage
  chunk outputs (alignment/judge) with provenance.
- :func:`verify_merged_result` – Verify entities/text against source; used after merging.
- :func:`normalize_final_result_for_output` – Normalize keys and structure for API output.

See Also
--------
- :mod:`task_detection` – Task type detection; task type drives selection here.
- :mod:`structsense.app` – Uses get_post_processor and get_result_merger per task type.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Tuple, cast
from collections import defaultdict, Counter

from .text_chunking import (
    _globalize_entities,
)

logger = logging.getLogger(__name__)

# ============================================================
# MODEL PRIORITY AND WEIGHTS
# ============================================================

# Model weights for weighted majority voting (higher = more reliable)
MODEL_WEIGHTS = {
    "d4data/biomedical-ner-all": 5.0,
    "alvaroalon2/biobert_genetic_ner": 4.0,
    "llm_ner": 3.9,  # LLM-based NER with domain context (extractor agent role/task) basically perform llm call
    "mobashgr/BC5CDR-chem-WLT-384-BioELECTRA-Pubmed-ENS-20-5": 3.0,
    "mobashgr/NCBI-disease-WLT-256-SciBERT-13INS": 2.0,
    "en_core_web_sm": 1.0,
}

# Mapping for model name variations to canonical names
MODEL_NAME_MAPPING = {
    "spacy (en_core_web_sm)": "en_core_web_sm",
    "spacy": "en_core_web_sm",
    "en_core_web_sm": "en_core_web_sm",
    "llm_ner": "llm_ner",
}

# Generic/uninformative labels to filter out (typically from spaCy)
GENERIC_LABELS_TO_REMOVE = {
    "CARDINAL",  # Numbers
    "ORDINAL",  # First, second, etc.
    "QUANTITY",  # Measurements
    "TIME",  # Time expressions
    "DATE",  # Date expressions
    "PERCENT",  # Percentages
    "MONEY",  # Monetary values
}


def normalize_model_name(source_model: str) -> str:
    """
    Normalize model name to canonical form.

    Args:
        source_model: Original model name

    Returns:
        Canonical model name
    """
    model_lower = source_model.lower().strip()
    return MODEL_NAME_MAPPING.get(model_lower, source_model)


def get_model_weight(source_model: str) -> float:
    """
    Get the weight for a given source model.

    Args:
        source_model: Model identifier

    Returns:
        Weight value (defaults to 1.0 if not found)
    """
    normalized = normalize_model_name(source_model)
    return MODEL_WEIGHTS.get(normalized, 1.0)


def should_filter_entity(entity: Dict[str, Any]) -> bool:
    """
    Determine if an entity should be filtered out.

    Args:
        entity: Entity dictionary with 'label' key

    Returns:
        True if entity should be removed, False otherwise
    """
    label = entity.get("label", "").upper()
    return label in GENERIC_LABELS_TO_REMOVE


def _entity_span(entity: Dict[str, Any]) -> tuple:
    """Get (start, end) from entity, preferring global_start/global_end."""
    gs, ge = entity.get("global_start"), entity.get("global_end")
    if gs is not None and ge is not None:
        return (gs, ge)
    return (entity.get("start", 0), entity.get("end", 0))


def calculate_overlap(e1: Dict[str, Any], e2: Dict[str, Any]) -> float:
    """
    Calculate overlap ratio between two entities.

    Args:
        e1, e2: Entity dictionaries with 'start'/'end' or 'global_start'/'global_end'

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    start1, end1 = _entity_span(e1)
    start2, end2 = _entity_span(e2)

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_end <= overlap_start:
        return 0.0

    overlap_len = overlap_end - overlap_start
    min_len = min(end1 - start1, end2 - start2)

    return overlap_len / min_len if min_len > 0 else 0.0


def entities_are_similar(e1: Dict[str, Any], e2: Dict[str, Any], overlap_threshold: float = 0.7) -> bool:
    """
    Check if two entities refer to the same thing.

    Args:
        e1, e2: Entity dictionaries
        overlap_threshold: Minimum overlap ratio to consider entities similar

    Returns:
        True if entities are similar, False otherwise
    """
    # Check overlap
    overlap = calculate_overlap(e1, e2)
    if overlap < overlap_threshold:
        return False

    # Check text similarity
    text1 = e1.get("entity", "").lower().strip()
    text2 = e2.get("entity", "").lower().strip()

    if text1 == text2:
        return True

    # Check if one is substring of other
    if text1 in text2 or text2 in text1:
        return True

    return False


def _merge_ner_entities_with_weighted_voting(all_entities: List[Dict[str, Any]], overlap_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Merge entities using weighted majority voting.

    Process:
    1. Filter out generic labels (CARDINAL, DATE, etc.)
    2. Group overlapping entities
    3. For each group, use weighted voting to determine final label
    4. Keep provenance (all source models and their predictions)
    5. Calculate weighted confidence score

    Args:
        all_entities: List of all entities from all models
        overlap_threshold: Minimum overlap to consider entities the same

    Returns:
        List of merged entities with weighted scores and provenance
    """
    if not all_entities:
        return []

    # Step 1: Filter out generic labels
    filtered_entities = [e for e in all_entities if not should_filter_entity(e)]

    logger.info(f"Filtered {len(all_entities) - len(filtered_entities)} generic entities")

    if not filtered_entities:
        return []

    # Sort by start position (use global span when present)
    filtered_entities = sorted(filtered_entities, key=lambda x: _entity_span(x)[0])

    # Step 2: Group overlapping entities
    groups = []
    used = set()

    for i, e1 in enumerate(filtered_entities):
        if i in used:
            continue

        group = [e1]
        used.add(i)

        # Find all similar entities
        for j in range(i + 1, len(filtered_entities)):
            if j in used:
                continue

            e2 = filtered_entities[j]
            if entities_are_similar(e1, e2, overlap_threshold):
                group.append(e2)
                used.add(j)

        groups.append(group)

    # Step 3: Process each group with weighted voting
    merged = []

    # Fallback when entity has no source_model: use first from group, else "extractor_agent" (preserve extractor provenance)
    def _effective_source_model(entity: Dict[str, Any], group: List[Dict[str, Any]]) -> str:
        raw = entity.get("source_model") or ""
        if isinstance(raw, str) and raw.strip():
            return normalize_model_name(raw.strip())
        for e in group:
            r = e.get("source_model")
            if isinstance(r, str) and r.strip():
                return normalize_model_name(r.strip())
        return "extractor_agent"

    for group in groups:
        # Collect label votes with weights
        label_votes = defaultdict(float)
        label_provenance = defaultdict(list)

        for entity in group:
            label = entity.get("label", "UNKNOWN")
            source = _effective_source_model(entity, group)
            weight = get_model_weight(source)

            label_votes[label] += weight
            label_provenance[label].append({"source_model": source, "weight": weight, "entity": entity.get("entity", "")})

        # Select winning label (highest weighted vote)
        winning_label = max(label_votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(label_votes.values())
        weighted_score = label_votes[winning_label] / total_weight if total_weight > 0 else 0.0

        # Use the longest/most complete text for the winning label
        winning_entities = [e for e in group if e.get("label") == winning_label]
        best_entity = max(winning_entities, key=lambda e: len(e.get("entity", "")))

        # Calculate span (use widest span from all entities; prefer global span)
        min_start = min(_entity_span(e)[0] for e in group)
        max_end = max(_entity_span(e)[1] for e in group)

        # Collect all source models involved (normalized; preserve from extractor, fallback extractor_agent)
        all_sources = list(set(_effective_source_model(e, group) for e in group))

        # Build provenance: all labels predicted with their sources
        provenance = []
        for label, sources in label_provenance.items():
            provenance.append({"label": label, "vote_weight": label_votes[label], "sources": sources})

        # Sort provenance by vote weight (descending)
        provenance = sorted(provenance, key=lambda x: x["vote_weight"], reverse=True)

        # Collect occurrences from all entities in group (start, end, global_start, global_end, sentence)
        seen_occurrences = set()
        occurrences = []
        for e in group:
            gs, ge = _entity_span(e)
            key = (gs, ge)
            if key in seen_occurrences:
                continue
            seen_occurrences.add(key)
            # Sentence-relative start/end (as in _globalize_entities / text_chunking)
            sent_start = e.get("sentence_start", e.get("start", 0))
            sent_end = e.get("sentence_end", e.get("end", 0))
            occurrences.append(
                {
                    "start": sent_start,
                    "end": sent_end,
                    "global_start": gs,
                    "global_end": ge,
                    "sentence": e.get("sentence", ""),
                }
            )
        occurrences = sorted(occurrences, key=lambda o: o["global_start"])

        merged_entity = {
            "entity": best_entity.get("entity", ""),
            "label": winning_label,
            "start": min_start,
            "end": max_end,
            "weighted_score": round(weighted_score, 3),
            "model_count": len(all_sources),
            "occurrences": occurrences,
            "provenance": provenance,  # All predictions with sources
        }

        merged.append(merged_entity)

    # Sort by start position
    merged = sorted(merged, key=lambda x: x["start"])

    return merged


# ============================================================
# NER POST-PROCESSING
# ============================================================
def ner_post_process(
    full_text: str,
    full_doc: Any,
    chunk: Dict[str, Any],
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Post-process NER (Named Entity Recognition) results.

    Args:
        full_text: The complete text document
        full_doc: spaCy document for the full text
        chunk: Dictionary with "entity" and "start" keys
        raw_result: Raw result from crew execution

    Returns:
        Dict with validated entities and key_terms
    """
    entities_raw = raw_result.get("entities", [])
    if not isinstance(entities_raw, list):
        entities_raw = []

    validated_entities = _globalize_entities(
        full_text,
        full_doc,
        chunk,
        entities_raw,
    )

    return {
        "entities": validated_entities,
        "key_terms": raw_result.get("key_terms", []),
    }


def merge_ner_results(
    results: List[Dict[str, Any]],
    full_text: str,
) -> Dict[str, Any]:
    """
    Merge NER results from multiple chunks using weighted majority voting.

    Args:
        results: List of processed results from all chunks
        full_text: The complete text document for validation

    Returns:
        Dict with merged entities (with weighted scores and provenance), key_terms, and metadata
    """
    all_entities = []
    all_terms = set()

    total_chunks = len(results)
    chunks_with_entities = 0
    chunks_with_key_terms = 0

    for processed in results:
        if "entities" in processed:
            chunk_entities = processed["entities"]
            if chunk_entities:
                chunks_with_entities += 1
                all_entities.extend(chunk_entities)
        if "key_terms" in processed:
            chunk_terms = processed["key_terms"]
            if chunk_terms:
                chunks_with_key_terms += 1
                for t in chunk_terms:
                    if isinstance(t, str):
                        all_terms.add(t)

    logger.info(
        f"Merging NER results: {total_chunks} chunks, {chunks_with_entities} with entities, {len(all_entities)} total entities before merging"
    )

    # Merge entities with weighted voting
    merged_entities = _merge_ner_entities_with_weighted_voting(all_entities)

    logger.info(f"After weighted voting: {len(merged_entities)} unique entities (merged from {len(all_entities)} total)")

    # Validate key_terms: keep only strings that actually appear in text
    text_lower = full_text.lower()
    key_terms = sorted({t for t in all_terms if isinstance(t, str) and t.lower() in text_lower})

    return {
        "entities": merged_entities,
        "key_terms": key_terms,
        "metadata": {
            "total_chunks": total_chunks,
            "chunks_with_entities": chunks_with_entities,
            "entities_before_merge": len(all_entities),
            "entities_after_merge": len(merged_entities),
        },
    }


# ============================================================
# GENERIC EXTRACTION POST-PROCESSING
# ============================================================
def generic_extraction_post_process(
    full_text: str,
    full_doc: Any,
    chunk: Dict[str, Any],
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic post-processing for extraction tasks.

    This is a pass-through function that returns the raw result
    without specific entity validation. Use this for tasks that
    don't require entity-level validation or that there doesn't exist task specific post-processing.

    Args:
        full_text: The complete text document
        full_doc: spaCy document for the full text
        chunk: Dictionary with "entity" and "start" keys
        raw_result: Raw result from crew execution

    Returns:
        Dict with the raw result (minimal processing)
    """
    return raw_result


def merge_generic_results(
    results: List[Dict[str, Any]],
    full_text: str,
) -> Dict[str, Any]:
    """
    Merge generic extraction results from multiple chunks.

    Concatenates all list values from chunk results into a single combined result.
    - If all results are lists, concatenate them into one list
    - If results are dicts, concatenate list values for each key
    - Non-list values are kept from the last result

    Args:
        results: List of processed results from all chunks
        full_text: The complete text document

    Returns:
        Dict with concatenated results, or list if all results were lists
    """
    if not results:
        return {"results": []}

    # Case 1: All results are lists - concatenate directly
    if all(isinstance(r, list) for r in results):
        concatenated = []
        for result_list in results:
            concatenated.extend(result_list)
        return concatenated

    # Case 2: Results are dicts - concatenate list values
    concatenated_dict = {}
    all_keys = set()

    # Collect all keys from all results
    for result in results:
        if isinstance(result, dict):
            all_keys.update(result.keys())

    # For each key, concatenate if values are lists
    for key in all_keys:
        list_values = []
        has_non_list = False

        for result in results:
            if isinstance(result, dict) and key in result:
                value = result[key]
                if isinstance(value, list):
                    list_values.append(value)
                else:
                    has_non_list = True
                    # Keep the last non-list value
                    concatenated_dict[key] = value

        # If all values for this key were lists, concatenate them
        if list_values and not has_non_list:
            concatenated_list = []
            for value_list in list_values:
                concatenated_list.extend(value_list)
            concatenated_dict[key] = concatenated_list

    # If no dict structure found, return aggregated results
    if not concatenated_dict:
        return {"results": results}

    return concatenated_dict


# ============================================================
# RESOURCE EXTRACTION POST-PROCESSING
# ============================================================
# Keys that LLM/configs may use for extraction output; we accept any of these so format changes don't break the pipeline.
RESOURCE_EXTRACTION_CONTAINER_KEYS = (
    "extracted_resources",  # BBQS format: {"1": [{...}], "2": [...]}
    "resources",  # list of resource objects
    "resource",  # single resource object
)


def _looks_like_resource(d: Any) -> bool:
    """True if d is a dict that looks like a resource (has at least one core field)."""
    if not isinstance(d, dict):
        return False
    return bool(d.get("name") or d.get("resource_name") or d.get("description") or d.get("type") or d.get("category") or d.get("target"))


def _collect_resources_from_value(value: Any) -> List[Dict[str, Any]]:
    """Recursively collect resource-like dicts from a value (list, dict, or single dict)."""
    out: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        if _looks_like_resource(value):
            out.append(value)
        else:
            for _k, v in value.items():
                out.extend(_collect_resources_from_value(v))
    elif isinstance(value, list):
        for item in value:
            out.extend(_collect_resources_from_value(item))
    return out


def _extract_resources_from_raw(raw_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a flat list of resource-like dicts from raw LLM output.

    Tries known container keys first (``extracted_resources``, ``resources``,
    ``resource``); for each, delegates to :func:`_collect_resources_from_value`
    so any nesting shape is handled:

    - ``extracted_resources: {"1": [{...}]}``   — dict-of-lists  ✓
    - ``extracted_resources: [{...}]``           — bare list      ✓
    - ``extracted_resources: {...}``             — bare resource  ✓
    - ``resources: [{...}]``                     — list           ✓
    - ``resources: {"1": [{...}]}``              — dict-of-lists  ✓
    - ``resource: {...}``                        — single dict    ✓

    Falls back to a full heuristic scan when no known key matches.
    """
    if not isinstance(raw_result, dict):
        return []

    # Known keys first — _collect_resources_from_value handles any shape
    for key in RESOURCE_EXTRACTION_CONTAINER_KEYS:
        val = raw_result.get(key)
        if val is None:
            continue
        items = _collect_resources_from_value(val)
        if items:
            return items

    # Heuristic fallback: scan the whole result for resource-like dicts
    return _collect_resources_from_value(raw_result)


def _scalar_str(val: Any) -> str:
    """Normalize to a single string for scalar fields; handle list from agent output."""
    if val is None:
        return ""
    if isinstance(val, list):
        return str(val[0]).strip() if val else ""
    return str(val).strip()


def _normalize_resource_for_merge(res: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single resource dict from chunk output (name/mentions) to a common shape.
    Includes downstream fields (judge_score, judge_rationale, etc.) when present.
    """
    name = _scalar_str(res.get("name") or res.get("resource_name"))
    out = {
        "name": name,
        "description": res.get("description") or "",
        "type": _scalar_str(res.get("type")),
        "category": res.get("category") or "",
        "target": res.get("target") or "",
        "specific_target": res.get("specific_target") or "",
        "url": res.get("url"),
        "key_features": res.get("key_features") or [],
        "performance": res.get("performance") or "",
        "model_architecture": res.get("model_architecture") or "",
        "mapped_target_concept": res.get("mapped_target_concept") or [],
        "mapped_specific_target_concept": res.get("mapped_specific_target_concept") or [],
        "mentions": res.get("mentions") or {},
    }
    # Downstream agent-added fields (preserve when present)
    if res.get("judge_score") is not None:
        out["judge_score"] = res["judge_score"]
    if res.get("judge_rationale") is not None:
        out["judge_rationale"] = res["judge_rationale"]
    if "provenance" in res and res["provenance"]:
        out["provenance"] = dict(res["provenance"]) if isinstance(res["provenance"], dict) else res["provenance"]
    return out


def _normalize_str_for_key(val: Any) -> str:
    """Normalize a value to a string for grouping; handle list/non-string from agent output."""
    if val is None:
        return ""
    if isinstance(val, list):
        if not val:
            return ""
        return " ".join(str(x).strip() for x in val if x is not None).strip().lower()
    return str(val).strip().lower()


def _resource_group_key(res: Dict[str, Any]) -> tuple:
    """Key for grouping same resource across chunks (normalized name + type)."""
    name = _normalize_str_for_key(res.get("name")) or "unknown"
    rtype = _normalize_str_for_key(res.get("type")) or "unknown"
    return (name, rtype)


def _build_mentions_dict(
    datasets: List[str],
    models: List[str],
    papers: List[str],
    tools: List[str],
    benchmarks: List[str],
) -> Dict[str, Any]:
    """Build mentions dict with only non-empty lists (omit null/empty)."""
    out: Dict[str, Any] = {}
    if datasets:
        out["datasets"] = datasets
    if models:
        out["related_models"] = models
    if papers:
        out["related_papers"] = papers
    if tools:
        out["tools"] = tools
    if benchmarks:
        out["benchmarks"] = benchmarks
    return out


def _merge_mention_lists(*lists: Any) -> List[str]:
    """Dedupe and sort non-empty string items from multiple lists; preserve order."""
    seen = set()
    out = []
    for lst in lists:
        if not isinstance(lst, list):
            continue
        for x in lst:
            if isinstance(x, str):
                s = x.strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def _merge_resources_into_one(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a group of normalized resource dicts into one final-schema resource (scalar fields)."""
    # Prefer longest/most complete scalar fields
    best = max(group, key=lambda r: (len(r.get("description") or ""), len(r.get("name") or "")))
    mentions_all = [r.get("mentions") or {} for r in group]

    # Merge mention lists; support both input keys (models, papers) and output keys (related_models, related_papers)
    def get_mention_list(m: Dict[str, Any], *keys: str) -> List[str]:
        for k in keys:
            v = m.get(k)
            if isinstance(v, list):
                return v
            if v is not None and not isinstance(v, list):
                return [str(v)] if str(v).strip() else []
        return []

    datasets = _merge_mention_lists(*(get_mention_list(m, "datasets") for m in mentions_all))
    models = _merge_mention_lists(*(get_mention_list(m, "related_models", "models") for m in mentions_all))
    papers = _merge_mention_lists(*(get_mention_list(m, "related_papers", "papers") for m in mentions_all))
    tools = _merge_mention_lists(*(get_mention_list(m, "tools") for m in mentions_all))
    benchmarks = _merge_mention_lists(*(get_mention_list(m, "benchmarks") for m in mentions_all))

    # First non-empty type/category/target/specific_target/url/performance/model_architecture
    def first_non_empty(*vals: Any) -> Any:
        for v in vals:
            if v is not None and str(v).strip():
                return v
        return None

    type_ = _scalar_str(first_non_empty(*(r.get("type") for r in group)) or "")
    category = _scalar_str(first_non_empty(*(r.get("category") for r in group)) or "")
    target = _scalar_str(first_non_empty(*(r.get("target") for r in group)) or "")
    specific_target = _scalar_str(first_non_empty(*(r.get("specific_target") for r in group)) or "N/A")
    url = first_non_empty(*(r.get("url") for r in group))
    performance = _scalar_str(first_non_empty(*(r.get("performance") for r in group)) or "")
    model_architecture = _scalar_str(first_non_empty(*(r.get("model_architecture") for r in group)) or "")

    # Longest description (handle list from agent)
    description = max((_scalar_str(r.get("description")) or "" for r in group), key=len)
    name = _scalar_str(best.get("name"))

    # Merge key_features and mapped_* (dedupe; prefer tool-backed entries with real ontology IDs over N/A)
    def _has_real_id(d: Any) -> bool:
        if not isinstance(d, dict):
            return False
        i = d.get("id") or d.get("mapped_target_concept", {}).get("id") if isinstance(d.get("mapped_target_concept"), dict) else None
        return bool(i and str(i).strip().startswith(("http://", "https://")))

    key_features = _merge_mention_lists(*(r.get("key_features") for r in group))
    # mapped_target_concept: by (id or label) keep best (prefer real IRI over N/A)
    target_by_key: Dict[str, Dict[str, Any]] = {}
    for r in group:
        for item in r.get("mapped_target_concept") or []:
            if not isinstance(item, dict):
                continue
            uid = (item.get("id") or item.get("label") or str(item) or "").strip()
            if not uid:
                uid = str(item)
            if uid not in target_by_key or (_has_real_id(item) and not _has_real_id(target_by_key[uid])):
                target_by_key[uid] = item
    mapped_target = list(target_by_key.values())

    # mapped_specific_target_concept: by normalized specific_target keep best (prefer inner real IRI over N/A)
    specific_by_key: Dict[str, Dict[str, Any]] = {}
    for r in group:
        for item in r.get("mapped_specific_target_concept") or []:
            if not isinstance(item, dict):
                continue
            st = (item.get("specific_target") or "").strip().lower()
            if not st:
                st = str(item.get("mapped_target_concept") or item)
            existing = specific_by_key.get(st)
            if existing is None or (_has_real_id(item) and not _has_real_id(existing)):
                specific_by_key[st] = item
    mapped_specific = list(specific_by_key.values())

    return {
        "resource_name": name,
        "description": description,
        "type": type_,
        "category": category,
        "target": target,
        "specific_target": specific_target,
        "mapped_target_concept": mapped_target,
        "mapped_specific_target_concept": mapped_specific,
        "key_features": key_features,
        "performance": performance,
        "url": url,
        "model_architecture": model_architecture,
        "mentions": _build_mentions_dict(datasets, models, papers, tools, benchmarks),
    }


def _aggregate_resources_into_one(all_resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge all resources into a single resource object where each field is a list.
    Use when the desired output is one aggregated resource with list-valued fields.
    """
    if not all_resources:
        return {
            "resource_name": [],
            "description": [],
            "type": [],
            "category": [],
            "target": [],
            "specific_target": [],
            "mapped_target_concept": [],
            "mapped_specific_target_concept": [],
            "key_features": [],
            "performance": [],
            "url": [],
            "model_architecture": [],
            "mentions": {},
        }

    def get_mention_list(m: Dict[str, Any], *keys: str) -> List[str]:
        for k in keys:
            v = m.get(k)
            if isinstance(v, list):
                return v
            if v is not None and not isinstance(v, list):
                return [str(v)] if str(v).strip() else []
        return []

    mentions_all = [r.get("mentions") or {} for r in all_resources]
    datasets = _merge_mention_lists(*(get_mention_list(m, "datasets") for m in mentions_all))
    models = _merge_mention_lists(*(get_mention_list(m, "related_models", "models") for m in mentions_all))
    papers = _merge_mention_lists(*(get_mention_list(m, "related_papers", "papers") for m in mentions_all))
    tools = _merge_mention_lists(*(get_mention_list(m, "tools") for m in mentions_all))
    benchmarks = _merge_mention_lists(*(get_mention_list(m, "benchmarks") for m in mentions_all))

    # Scalar fields as lists (one entry per resource, preserve order)
    resource_name = [r.get("name") or "" for r in all_resources]
    description = [r.get("description") or "" for r in all_resources]
    type_list = [r.get("type") or "" for r in all_resources]
    category_list = [r.get("category") or "" for r in all_resources]
    target_list = [r.get("target") or "" for r in all_resources]
    specific_target_list = [r.get("specific_target") or "" for r in all_resources]
    url_list = [r.get("url") for r in all_resources if r.get("url") is not None]
    performance_list = [r.get("performance") or "" for r in all_resources if (r.get("performance") or "").strip()]
    model_architecture_list = [r.get("model_architecture") or "" for r in all_resources if (r.get("model_architecture") or "").strip()]

    key_features = _merge_mention_lists(*(r.get("key_features") for r in all_resources))
    mapped_target = []
    seen_target = set()
    for r in all_resources:
        for item in r.get("mapped_target_concept") or []:
            if isinstance(item, dict):
                uid = item.get("id") or item.get("label") or str(item)
                if uid not in seen_target:
                    seen_target.add(uid)
                    mapped_target.append(item)
            elif item not in seen_target:
                seen_target.add(item)
                mapped_target.append(item)
    mapped_specific = []
    seen_specific = set()
    for r in all_resources:
        for item in r.get("mapped_specific_target_concept") or []:
            if isinstance(item, dict):
                uid = item.get("id") or item.get("label") or str(item)
                if uid not in seen_specific:
                    seen_specific.add(uid)
                    mapped_specific.append(item)
            elif item not in seen_specific:
                seen_specific.add(item)
                mapped_specific.append(item)

    return {
        "resource_name": resource_name,
        "description": description,
        "type": type_list,
        "category": category_list,
        "target": target_list,
        "specific_target": specific_target_list,
        "mapped_target_concept": mapped_target,
        "mapped_specific_target_concept": mapped_specific,
        "key_features": key_features,
        "performance": performance_list if performance_list else [],
        "url": url_list,
        "model_architecture": model_architecture_list if model_architecture_list else [],
        "mentions": _build_mentions_dict(datasets, models, papers, tools, benchmarks),
    }


def resource_extraction_post_process(
    full_text: str,
    full_doc: Any,
    chunk: Dict[str, Any],
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize chunk output for resource extraction.

    Accepts any raw_result shape: known keys (extracted_resources, resources, resource)
    or a heuristic scan for resource-like dicts (name, description, type, category, etc.).
    Always returns {"resources": [...]} for the merger so downstream never receives empty
    content due to format variation.
    """
    items = _extract_resources_from_raw(raw_result)
    if not items:
        return {"resources": []}
    normalized = [_normalize_resource_for_merge(r) for r in items]
    if len(normalized) != len(items):
        logger.debug(
            "resource_extraction_post_process: extracted %d items, normalized %d",
            len(items),
            len(normalized),
        )
    return {"resources": normalized}


def merge_resource_results(
    results: List[Dict[str, Any]],
    full_text: str,
) -> Dict[str, Any]:
    """
    Merge resource extraction results from multiple chunks into one aggregated resource.

    All resources are merged into a single resource object where each field is a list:
    resource_name, description, type, category, target, specific_target, url,
    performance, model_architecture (each a list); key_features, mapped_target_concept,
    mapped_specific_target_concept (merged deduped lists); mentions (datasets,
    related_models, related_papers, tools, benchmarks) as merged deduped lists.
    Judge fields (judge_score, judge_rationale, feedback_response) are left for later agents.
    """
    all_resources: List[Dict[str, Any]] = []
    for processed in results:
        if not isinstance(processed, dict):
            continue
        resources = processed.get("resources")
        if isinstance(resources, list):
            all_resources.extend(resources)
        # Also accept chunk that returned a single "resource" (post_process normalizes to "resources")
        if "resource" in processed and isinstance(processed["resource"], dict):
            all_resources.append(_normalize_resource_for_merge(processed["resource"]))

    if not all_resources:
        return {"resources": []}

    # Normalize so all items have same shape
    normalized = [_normalize_resource_for_merge(res) for res in all_resources]

    # Merge all into one resource with list-valued fields
    aggregated = _aggregate_resources_into_one(normalized)

    logger.info(f"Resource merge: {len(all_resources)} raw resources -> 1 aggregated resource (list-valued fields)")

    return {"resources": [aggregated]}


# ============================================================
# DOWNSTREAM MERGE WITH PROVENANCE
# ============================================================
# Which fields each agent typically adds/updates (for provenance tracking).
# Used for both NER and resource (and other) tasks; each item only gets fields it actually has.
PROVENANCE_AGENT_FIELDS: Dict[str, List[str]] = {
    "extractor_agent": [
        # NER fields
        "entity",
        "label",
        "sentence",
        "start",
        "end",
        "paper_location",
        "paper_title",
        "doi",
        # Resource fields
        "name",
        "description",
        "type",
        "category",
        "target",
        "specific_target",
        "url",
        "key_features",
        "performance",
        "model_architecture",
        "mentions",
        # Generic / shared
        "remarks",
    ],
    "alignment_agent": [
        # Concept mapping — flat (written by apply_concept_mapping_to_result)
        "ontology_id",
        "ontology_label",
        "ontology",
        "concept_mapping_provenance",
        # Structured mapped concepts — resource
        "mapped_name_concept",
        "mapped_target_concept",
        "mapped_specific_target_concept",
        "mapped_type_concept",
        "mapped_category_concept",
        # Flat per-field prefixed variants (target_, specific_target_, type_, category_)
        "target_ontology_id",
        "target_ontology_label",
        "target_ontology",
        "target_concept_mapping_provenance",
        "specific_target_ontology_id",
        "specific_target_ontology_label",
        "specific_target_ontology",
        "specific_target_concept_mapping_provenance",
        "type_ontology_id",
        "type_ontology_label",
        "type_ontology",
        "type_concept_mapping_provenance",
        "category_ontology_id",
        "category_ontology_label",
        "category_ontology",
        "category_concept_mapping_provenance",
        # NER label ontology
        "label_ontology_id",
        "label_ontology_label",
        "label_ontology",
        # Mention-level ontology (mentions_with_ontology is built here)
        "mentions_with_ontology",
    ],
    "judge_agent": ["judge_score", "judge_rationale", "remarks"],
    "humanfeedback_agent": ["judge_score", "judge_rationale", "user_feedback_applied", "remarks", "feedback_response"],
}

# Fields that are internal bookkeeping and should never appear in provenance field lists.
_PROVENANCE_SKIP_FIELDS = frozenset(
    {
        "provenance",
        "concept_mapping_provenance",
        "target_concept_mapping_provenance",
        "specific_target_concept_mapping_provenance",
        "type_concept_mapping_provenance",
        "category_concept_mapping_provenance",
        "_extraction_chunk_count",
    }
)


def _merge_single_resource_group_with_provenance(group: List[Dict[str, Any]], agent_key: str) -> Dict[str, Any]:
    """Merge a group of resource dicts (same resource across chunks) and add provenance for this agent."""
    if not group:
        return {}
    # Use existing merge for base + mapped + mentions; then add judge_score and any extra keys
    merged = _merge_resources_into_one([_normalize_resource_for_merge(r) for r in group])
    # Restore scalar name (merge may use resource_name)
    if "resource_name" in merged and "name" not in merged:
        merged["name"] = merged["resource_name"]
    # Merge judge_score, judge_rationale: take first non-empty
    for key in ("judge_score", "judge_rationale"):
        for r in group:
            if r.get(key) is not None:
                merged[key] = r[key]
                break
    # Merge provenance from any item and add this agent's contribution
    existing = merged.get("provenance") or {}
    if isinstance(existing, dict):
        merged["provenance"] = dict(existing)
    else:
        merged["provenance"] = {}
    fields_this_agent = PROVENANCE_AGENT_FIELDS.get(agent_key, [])
    contributed = [f for f in fields_this_agent if f in merged and merged[f] is not None]
    if contributed:
        merged["provenance"][agent_key] = contributed
    return merged


def _flatten_resource_items(container: Any) -> List[Dict[str, Any]]:
    """Flatten container to a list of resource dicts; handle dict-of-lists, list, or nested lists from agent output."""
    out: List[Dict[str, Any]] = []
    if container is None:
        return out
    if isinstance(container, dict):
        for _k, v in container.items():
            out.extend(_flatten_resource_items(v))
        return out
    if isinstance(container, list):
        for x in container:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, list):
                out.extend(_flatten_resource_items(x))
        return out
    return out


def merge_downstream_chunk_results_with_provenance(
    chunk_results: List[Dict[str, Any]],
    container_key: str,
    agent_key: str,
) -> Dict[str, Any]:
    """
    Merge downstream chunk results using the same resource merge logic and add provenance.

    Each chunk result is a dict with a container key (e.g. aligned_resources, judge_resource)
    whose value is either a dict (id -> list of items) or a list of items. We flatten all
    items, group by resource (name+type), merge each group with _merge_resources_into_one
    style and tag which agent contributed which fields.
    """
    all_items: List[Dict[str, Any]] = []
    for result in chunk_results:
        if not isinstance(result, dict):
            continue
        container = result.get(container_key)
        all_items.extend(_flatten_resource_items(container))

    if not all_items:
        first = chunk_results[0] if chunk_results else {}
        return {container_key: {} if isinstance(first.get(container_key), dict) else []}

    # Group by resource identity (name + type); skip items that break grouping
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        if not isinstance(item, dict):
            continue
        try:
            key = _resource_group_key(item)
            groups[key].append(item)
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning("Skipping resource item for grouping due to unexpected shape: %s", e)
            continue

    # Merge each group and add provenance
    merged_list = [_merge_single_resource_group_with_provenance(g, agent_key) for g in groups.values()]
    # Preserve dict-of-lists shape if original was dict (use index as key)
    first_container = chunk_results[0].get(container_key) if chunk_results else None
    if isinstance(first_container, dict):
        out_container = {str(i + 1): [m] for i, m in enumerate(merged_list)}
    else:
        out_container = merged_list
    return {container_key: out_container}


# ---------------------------------------------------------------------------
# Ontology consistency pass for parallel downstream chunking
# ---------------------------------------------------------------------------


def _ontology_score(ent: Dict[str, Any]) -> int:
    """Score an entity's ontology mapping quality. Higher = better."""
    s = 0
    if ent.get("concept_mapping_provenance") == "tool":
        s += 100
    oid = str(ent.get("ontology_id") or "").strip().lower()
    if oid and oid not in ("n/a", "none", "null", ""):
        s += 50
    if str(ent.get("ontology_label") or "").strip().lower() not in ("n/a", "none", "null", ""):
        s += 10
    return s


def unify_ontology_across_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """After parallel chunk alignment/judge, unify ontology fields across all entities.

    When the same entity text is processed in different parallel chunks, each LLM call
    may produce a different ontology ID for it.  This pass:

    1. Finds the best ontology mapping per (entity_text, label) pair — tool-backed
       concept mapping beats llm_knowledge; a real IRI beats N/A.
    2. Applies that best mapping to *every* occurrence of that entity text so that
       all individual instances (different sentences/positions) share one consistent
       ontology assignment.

    Individual entity instances are preserved — nothing is deduplicated.  Only the
    ontology fields are unified.
    """
    _ONTOLOGY_FIELDS = ("ontology_id", "ontology_label", "ontology", "concept_mapping_provenance")

    # Step 1: determine best mapping per (entity_text, label)
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        key = (
            str(ent.get("entity") or ent.get("term") or ent.get("name") or "").lower().strip(),
            str(ent.get("label") or "").lower().strip(),
        )
        if not key[0]:
            continue
        score = _ontology_score(ent)
        if key not in best or score > _ontology_score(best[key]):
            best[key] = {f: ent[f] for f in _ONTOLOGY_FIELDS if f in ent}

    # Step 2: apply best mapping to every occurrence
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        key = (
            str(ent.get("entity") or ent.get("term") or ent.get("name") or "").lower().strip(),
            str(ent.get("label") or "").lower().strip(),
        )
        if key in best:
            ent.update(best[key])

    return entities


def add_provenance_to_result(result_dict: Dict[str, Any], container_key: str, agent_key: str) -> Dict[str, Any]:
    """Add provenance to every item under *container_key*.

    For known agent keys (extractor_agent, alignment_agent, …) the contributed
    field list is intersected with ``PROVENANCE_AGENT_FIELDS[agent_key]`` so
    only fields that agent is responsible for are recorded.

    For unknown agent keys *or* when the known-field intersection is empty
    (generic/custom tasks with non-standard field names), falls back to
    recording all non-internal fields present on the item — so provenance is
    never silently empty for any task type.
    """
    if not result_dict or container_key not in result_dict:
        return result_dict
    container = result_dict[container_key]
    known_fields = PROVENANCE_AGENT_FIELDS.get(agent_key)  # None when agent_key unknown

    def add_to_item(item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        existing = item.get("provenance") or {}
        if not isinstance(existing, dict):
            existing = {}

        if known_fields is not None:
            contributed = [f for f in known_fields if f in item and item[f] is not None]
        else:
            contributed = []

        if not contributed:
            # Generic fallback: record every field the item has that is not
            # internal bookkeeping (provenance itself, concept_mapping_provenance, etc.)
            contributed = [f for f in item if f not in _PROVENANCE_SKIP_FIELDS and item[f] is not None]

        if contributed:
            existing[agent_key] = contributed
            item["provenance"] = existing

    if isinstance(container, dict):
        for _k, v in container.items():
            if isinstance(v, list):
                for it in v:
                    add_to_item(it)
            elif isinstance(v, dict):
                add_to_item(v)
    elif isinstance(container, list):
        for it in container:
            add_to_item(it)
    return result_dict


# ============================================================
# TASK REGISTRY
# ============================================================
_TASK_POST_PROCESSORS: Dict[str, Callable] = {
    "ner": ner_post_process,
    "extraction": generic_extraction_post_process,
    "resource": resource_extraction_post_process,
}

_TASK_MERGERS: Dict[str, Callable] = {
    "ner": merge_ner_results,
    "extraction": merge_generic_results,
    "resource": merge_resource_results,
}


def get_post_processor(task_type: str) -> Callable:
    """Return the post-processing function for a given task type.

    The post-processor is applied to each chunk's raw output before merging.
    Unknown task types fall back to :func:`generic_extraction_post_process`.

    Parameters
    ----------
    task_type : str
        Task type identifier (e.g. ``ner``, ``extraction``, ``resource``).
        Must match a key in the internal registry (see :func:`register_task_type`).

    Returns
    -------
    callable
        Post-processing function with signature suitable for chunk output.

    Note
    ----
    To add a new task type, use :func:`register_task_type` before running the pipeline.
    """
    if task_type not in _TASK_POST_PROCESSORS:
        logger.warning(f"Unknown task type '{task_type}', using generic post-processor")
        return generic_extraction_post_process
    return _TASK_POST_PROCESSORS[task_type]


def get_result_merger(task_type: str) -> Callable:
    """Return the result merging function for a given task type.

    The merger combines per-chunk results (e.g. list of entity lists) into
    a single structure (e.g. deduplicated entities with provenance).
    Unknown task types fall back to :func:`merge_generic_results`.

    Parameters
    ----------
    task_type : str
        Task type identifier (e.g. ``ner``, ``extraction``, ``resource``).

    Returns
    -------
    callable
        Merging function; typically (list of chunk results, full_text) → merged dict.

    See Also
    --------
    register_task_type : Register a custom merger for a new task type.
    """
    if task_type not in _TASK_MERGERS:
        logger.warning(f"Unknown task type '{task_type}', using generic merger")
        return merge_generic_results
    return _TASK_MERGERS[task_type]


def register_task_type(
    task_type: str,
    post_processor: Callable,
    result_merger: Callable,
) -> None:
    """Register a new task type with its post-processor and result merger.

    Call this before running the pipeline when adding a new use case (e.g. a new
    extractor task type) so that :func:`get_post_processor` and :func:`get_result_merger`
    return your functions for this task type.

    Parameters
    ----------
    task_type : str
        Task type identifier (e.g. ``my_custom_extraction``). Should match the
        type returned by :func:`task_detection.detect_task_type` for your task config.
    post_processor : callable
        Function to post-process each chunk's raw output (e.g. normalize, filter).
    result_merger : callable
        Function to merge a list of chunk results into one structure (e.g. with
        provenance). Signature typically (results: list, full_text: str) -> dict.
    """
    _TASK_POST_PROCESSORS[task_type] = post_processor
    _TASK_MERGERS[task_type] = result_merger
    logger.info(f"Registered task type '{task_type}' with custom post-processing")


def get_registered_task_types() -> List[str]:
    """Return the list of all registered task types.

    Returns
    -------
    list of str
        Keys for which :func:`get_post_processor` and :func:`get_result_merger`
        return custom functions (e.g. ``ner``, ``resource``, ``extraction``).
    """
    return list(_TASK_POST_PROCESSORS.keys())


# ============================================================
# POST-MERGE VERIFIER (ensure text, sentences, entities present in source)
# ============================================================
def verify_ner_result(merged_result: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """
    Verify NER merged result against full_text: keep only entities whose text
    and sentences are present in the source; key_terms must appear in text.
    Attaches a "verification" dict with counts and any dropped items.
    """
    if not full_text or not isinstance(merged_result, dict):
        return merged_result
    text_lower = full_text.lower()
    text_stripped = full_text.strip()

    entities = merged_result.get("entities") or []
    key_terms = merged_result.get("key_terms", [])
    metadata = dict(merged_result.get("metadata", {}))

    verified_entities: List[Dict[str, Any]] = []
    entities_dropped: List[Dict[str, Any]] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        ent_text = (entity.get("entity") or "").strip()
        if not ent_text:
            entities_dropped.append({**entity, "reason": "empty_text"})
            continue
        # Check entity text appears in full_text (at span if we have it, or anywhere)
        gs, ge = entity.get("global_start"), entity.get("global_end")
        if gs is not None and ge is not None and 0 <= gs < ge <= len(full_text):
            span_text = full_text[gs:ge].strip()
            if _normalize_span_for_compare(span_text) == _normalize_span_for_compare(ent_text):
                # Optionally verify occurrence sentences
                keep = True
                for occ in entity.get("occurrences", []):
                    if isinstance(occ, dict) and occ.get("sentence"):
                        sent = (occ.get("sentence") or "").strip()
                        if sent and sent.lower() not in text_lower:
                            keep = False
                            break
                if keep:
                    verified_entities.append(entity)
                else:
                    entities_dropped.append({**entity, "reason": "sentence_not_in_text"})
                continue
        # Fallback: entity text appears anywhere in full_text
        if ent_text.lower() in text_lower:
            verified_entities.append(entity)
        else:
            entities_dropped.append({**entity, "reason": "text_not_in_source"})

    # Key terms: keep only those present in full_text (re-verify)
    key_terms_valid = [t for t in key_terms if isinstance(t, str) and t.strip() and t.strip().lower() in text_lower]
    key_terms_dropped = [t for t in key_terms if isinstance(t, str) and t not in key_terms_valid]

    verification = {
        "entities_present": len(verified_entities),
        "entities_dropped": len(entities_dropped),
        "entities_dropped_detail": entities_dropped,
        "key_terms_present": len(key_terms_valid),
        "key_terms_dropped": len(key_terms_dropped),
        "all_entities_present_in_text": len(entities_dropped) == 0,
        "all_key_terms_present_in_text": len(key_terms_dropped) == 0,
    }
    if entities_dropped:
        logger.info(f"Verifier: dropped {len(entities_dropped)} entities not present in source text; " f"kept {len(verified_entities)}")
    if key_terms_dropped:
        logger.info(f"Verifier: dropped {len(key_terms_dropped)} key_terms not present in source text; " f"kept {len(key_terms_valid)}")

    out = {
        **merged_result,
        "entities": verified_entities,
        "key_terms": sorted(set(key_terms_valid)),
        "metadata": {**metadata, "verification": verification},
        "verification": verification,
    }
    return out


def _normalize_span_for_compare(s: str) -> str:
    """Normalize string for span comparison (collapse whitespace, strip)."""
    if not s:
        return ""
    return " ".join(s.split()).strip().lower()


def verify_resource_result(merged_result: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """Verify resource extraction: check that each resource name is grounded in the source text.

    Resources whose name cannot be found in the source are dropped and recorded in
    ``verification.resources_dropped_detail``.  Mention items (datasets, models, tools, etc.)
    inside each kept resource are also filtered to those whose name appears in the source.
    """
    if not full_text or not isinstance(merged_result, dict):
        return merged_result
    text_lower = full_text.lower()
    resources = merged_result.get("resources", [])
    if not resources:
        merged_result["verification"] = {
            "resources_checked": 0,
            "resources_present": 0,
            "resources_dropped": 0,
            "resources_dropped_detail": [],
            "all_resources_present_in_text": True,
        }
        return merged_result

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for res in resources:
        if not isinstance(res, dict):
            continue
        raw_name = res.get("name") or res.get("resource_name")
        if isinstance(raw_name, list):
            candidates = [n.strip() for n in raw_name if n and n.strip()]
        elif raw_name:
            candidates = [raw_name.strip()]
        else:
            candidates = []

        if not candidates:
            # No name — cannot ground; drop
            dropped.append({**res, "reason": "no_name"})
            continue

        # Use the first candidate name that appears in the source text
        name = next((n for n in candidates if n.lower() in text_lower), None)
        if name is None:
            dropped.append({**res, "reason": "name_not_in_source"})
            logger.info(
                "Resource verifier: dropped %r (and %d alt names) — none found in source text",
                candidates[0],
                len(candidates) - 1,
            )
            continue

        # Name is grounded — also filter mention lists to items present in source
        cleaned = dict(res)
        mentions = res.get("mentions") or {}
        if isinstance(mentions, dict):
            filtered_mentions: Dict[str, Any] = {}
            for cat, items in mentions.items():
                if isinstance(items, list):
                    # Items can be plain strings or dicts with a "name" key
                    grounded_items = []
                    for item in items:
                        if isinstance(item, dict):
                            item_name = (item.get("name") or "").strip()
                        else:
                            item_name = str(item).strip()
                        if item_name and item_name.lower() in text_lower:
                            grounded_items.append(item)
                        elif not item_name:
                            grounded_items.append(item)  # keep if no text to check
                    filtered_mentions[cat] = grounded_items
                else:
                    filtered_mentions[cat] = items
            cleaned["mentions"] = filtered_mentions
        kept.append(cleaned)

    verification = {
        "resources_checked": len(kept) + len(dropped),
        "resources_present": len(kept),
        "resources_dropped": len(dropped),
        "resources_dropped_detail": dropped,
        "all_resources_present_in_text": len(dropped) == 0,
    }
    if dropped:
        logger.info(
            "Resource verifier: dropped %d resource(s) not grounded in source text; kept %d",
            len(dropped),
            len(kept),
        )

    out = {**merged_result, "resources": kept, "verification": verification}
    return out


# Candidate field names (in priority order) used to find the text of a generic extracted item.
_GENERIC_TEXT_FIELDS = ("text", "term", "phrase", "keyword", "name", "value", "label", "entity")


def _generic_item_text(item: Any) -> str:
    """Return the best text representation of a generic extracted item."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for field in _GENERIC_TEXT_FIELDS:
            val = item.get(field)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def verify_generic_result(merged_result: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """Verify generic extraction: drop items whose text is not grounded in the source.

    Walks every list value in the result (the container key is task-defined).
    Items with no detectable text field are kept (cannot verify).
    """
    if not full_text or not isinstance(merged_result, dict):
        if isinstance(merged_result, dict):
            merged_result.setdefault("verification", {"all_present": True})
        return merged_result

    text_lower = full_text.lower()
    out = dict(merged_result)
    total_checked = total_kept = total_dropped = 0
    all_dropped_detail: List[Dict[str, Any]] = []

    for key, val in merged_result.items():
        if key in {"errors", "task_type", "elapsed_time", "verification", "metadata", "human_feedback_skipped"}:
            continue
        if not isinstance(val, list):
            continue

        kept: List[Any] = []
        for item in val:
            item_text = _generic_item_text(item)
            if not item_text:
                kept.append(item)  # no text to check — keep
                continue
            total_checked += 1
            if item_text.lower() in text_lower:
                kept.append(item)
                total_kept += 1
            else:
                total_dropped += 1
                detail = {"container": key, "reason": "text_not_in_source"}
                if isinstance(item, dict):
                    detail.update(item)
                else:
                    detail["text"] = item_text
                all_dropped_detail.append(detail)

        out[key] = kept

    if total_dropped:
        logger.info(
            "Generic verifier: dropped %d item(s) not grounded in source text; kept %d",
            total_dropped,
            total_kept,
        )

    out["verification"] = {
        "items_checked": total_checked,
        "items_present": total_kept,
        "items_dropped": total_dropped,
        "items_dropped_detail": all_dropped_detail,
        "all_present_in_text": total_dropped == 0,
    }
    return out


def verify_merged_result(merged_result: Dict[str, Any], full_text: str, task_type: str) -> Dict[str, Any]:
    """
    Run the appropriate verifier for the task type so that all text, sentences,
    and entities are present in the source. Returns the merged result with
    only valid items and a "verification" key with counts and dropped details.
    """
    if not full_text or not isinstance(merged_result, dict):
        return merged_result
    if task_type == "ner":
        return verify_ner_result(merged_result, full_text)
    if task_type in ("resource", "structured_extraction"):
        return verify_resource_result(merged_result, full_text)
    return verify_generic_result(merged_result, full_text)


# Keys that are intermediate agent outputs; stripped from final result so output is task-specific only (like NER).
INTERMEDIATE_CONTAINER_KEYS_NER = ("extracted_terms", "aligned_ner_terms", "judge_ner_terms")
INTERMEDIATE_CONTAINER_KEYS_RESOURCE = ("extracted_resources", "aligned_resources", "judge_resource")

# ---------------------------------------------------------------------------
# Stage-output key normalization
# ---------------------------------------------------------------------------
# Problem addressed:
#   Each pipeline stage (extraction, alignment, judge, human-feedback) may write
#   its output under a different dict key depending on the LLM response and the
#   task config (e.g. "judge_ner_terms", "aligned_ner_terms", "extracted_terms"
#   for NER; "judge_resource", "aligned_resources", "extracted_resources" for
#   resources).  Downstream code and the final result builder all expect the
#   canonical keys ("entities"/"key_terms" for NER, "resources" for resource).
#   Without normalization, missing canonical keys produce empty results even
#   when the LLM returned perfectly valid data under a stage-specific key.
#
# Solution:
#   `promote_stage_output_to_canonical` is called once after EVERY stage in the
#   pipeline loop (app.py).  It inspects the output in a priority order
#   (most-refined first: judge > alignment > extraction) and promotes any
#   stage-specific key to the canonical one.  After this call, all downstream
#   code — verify_ner_result, normalize_final_result_for_output,
#   apply_concept_mapping_to_result — can safely assume canonical keys exist.

# Priority order: most refined → least refined (judge > alignment > extraction)
_NER_STAGE_KEYS = ("judge_ner_terms", "aligned_ner_terms", "extracted_terms")
_RESOURCE_STAGE_KEYS = ("judge_resource", "aligned_resources", "extracted_resources", "resources")
# Pipeline placeholder wrapper keys: the LLM sometimes nests its output under
# these schema keys instead of placing entities/resources at the top level.
_UNWRAP_KEYS = (
    "judged_structured_information_with_human_feedback",
    "aligned_structured_information",
    "extracted_structured_information",
)


def promote_stage_output_to_canonical(result: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Normalize any pipeline stage output so that canonical keys are always at the
    top level, regardless of which key the LLM chose to use.

    What it fixes:
    - NER judge returns {"judge_ner_terms": {...}} instead of {"entities": [...]}
      → promotes judge_ner_terms (dict-of-lists keyed by entity ID) to entities list
    - NER alignment returns {"aligned_ner_terms": {...}} → promotes to entities
    - Resource judge returns {"judge_resource": [...]} → promotes to resources
    - Resource alignment returns {"aligned_resources": [...]} → promotes to resources
    - Any stage wraps output in {"extracted_structured_information": {...}} etc.
      → unwraps the inner dict first, then promotes

    Priority order (most refined wins): judge > alignment > extraction.
    Only promotes when the canonical key is absent or empty — never overwrites
    existing data.

    Called in app.py after every pipeline stage so all downstream logic
    (verify_ner_result, normalize_final_result_for_output, concept mapping)
    can rely on canonical keys being present.

    Operates in-place and returns the result.

    Example — NER judge stage output (BEFORE this call)::

        {
            "judge_ner_terms": {
                "1": [{"entity": "scRNA-seq", "label": "Technique", ...}],
                "2": [{"entity": "CRISPR",    "label": "Technique", ...}]
            },
            "key_terms": ["single-cell", "genome editing"]
        }

    After this call the same dict also contains::

        {
            "entities": [
                {"entity": "scRNA-seq", "label": "Technique", ...},
                {"entity": "CRISPR",    "label": "Technique", ...}
            ],
            "key_terms": ["single-cell", "genome editing"],
            "judge_ner_terms": { ... }   # original key kept; popped later by normalize_final_result_for_output
        }

    Example — resource alignment stage output (BEFORE this call)::

        {"aligned_resources": [{"name": "BERT", "type": "Model", ...}]}

    After this call::

        {"aligned_resources": [...], "resources": [{"name": "BERT", "type": "Model", ...}]}
    """
    if not isinstance(result, dict):
        return result
    t = (task_type or "").strip().lower()

    if t == "ner":
        # Step 1 — Unwrap nested pipeline-placeholder containers.
        # The LLM sometimes wraps its NER output inside a schema key like
        # "extracted_structured_information": {"entities": [...]} instead of
        # placing entities at the top level.
        if not result.get("entities"):
            for uk in _UNWRAP_KEYS:
                inner = result.get(uk)
                if isinstance(inner, dict) and (inner.get("entities") or inner.get("key_terms")):
                    result["entities"] = inner.get("entities") or []
                    result.setdefault("key_terms", inner.get("key_terms") or [])
                    break

        # Step 2 — Promote from stage-specific NER keys (judge > alignment > extraction).
        # _flatten_container_to_list handles both list-of-dicts and dict-of-lists
        # (the latter being the judge's typical format: {"1": [{entity...}], "2": [...]}）.
        #
        # DESIGN PRINCIPLE — entities are never dropped between stages:
        #   Alignment and judge only ENRICH entities (add ontology fields, validate).
        #   If the LLM wrote entities:[5] alongside aligned_ner_terms with 322 items,
        #   the 322 is the authoritative output and must win.
        #
        # OLD behaviour (bug): `if not result.get("entities")` — skipped this block
        #   entirely when entities was already non-empty, even if it only had 2 items
        #   while extracted_terms / aligned_ner_terms had hundreds.
        #
        # NEW behaviour: always scan ALL stage-specific keys and use whichever has
        #   the MOST entities.  If no stage key beats the existing entities list,
        #   keep what's there.
        _best_list = result.get("entities") or []
        _best_kts = None
        for key in _NER_STAGE_KEYS:
            container = result.get(key)
            if container:
                promoted = _flatten_container_to_list(container)
                if len(promoted) > len(_best_list):
                    _best_list = promoted
                    _best_kts = container.get("key_terms") if isinstance(container, dict) else None
        if _best_list:
            result["entities"] = _best_list
            if _best_kts:
                result.setdefault("key_terms", _best_kts)

        # Guarantee canonical keys always exist (empty list is better than KeyError).
        result.setdefault("entities", [])
        result.setdefault("key_terms", [])

    elif t in ("resource", "structured_extraction"):
        # Step 1 — Unwrap nested pipeline-placeholder containers.
        if not result.get("resources"):
            for uk in _UNWRAP_KEYS:
                inner = result.get(uk)
                if isinstance(inner, dict) and inner.get("resources"):
                    result["resources"] = inner["resources"]
                    break

        # Step 2 — Promote from stage-specific resource keys (judge > alignment > extraction).
        # Same "take largest" principle as the NER path above.
        _best_resources = result.get("resources") or []
        for key in _RESOURCE_STAGE_KEYS:
            container = result.get(key)
            if container:
                flat = _flatten_container_to_list(container)
                if len(flat) > len(_best_resources):
                    _best_resources = flat
        if _best_resources:
            result["resources"] = _best_resources

        result.setdefault("resources", [])

    return result


def inject_alignment_concept_mapping_into_ner_entities(
    entities: List[Dict[str, Any]],
    session_outputs: list,
) -> int:
    """
    Enrich NER entity dicts with concept mapping fields from alignment agent tool calls.

    What it fixes:
    - After the alignment stage, the alignment agent has called ConceptMappingLocalTool
      (or ConceptMappingTool) for each entity.  Those tool outputs are stored in
      _ALIGNMENT_TOOL_OUTPUTS.  Without this function, the ontology info (ontology_id,
      ontology_label, ontology) stays in the tool outputs and never reaches the
      entity dicts the caller actually returns — entities have no ontology fields even
      though the alignment agent resolved them successfully.
    - This function bridges that gap: for each entity in `entities` it looks up the
      entity text in the tool outputs and injects the three canonical fields.

    Example — entity dict BEFORE injection::

        {
            "entity": "scRNA-seq",
            "label": "Technique",
            "sentence": "We used scRNA-seq to profile cells.",
            "start": [10],
            "end": [18]
        }

    Entity dict AFTER injection::

        {
            "entity": "scRNA-seq",
            "label": "Technique",
            "sentence": "We used scRNA-seq to profile cells.",
            "start": [10],
            "end": [18],
            "ontology_id": "http://purl.obolibrary.org/obo/OBI_0002631",
            "ontology_label": "single cell RNA sequencing assay",
            "ontology": "OBI"
        }

    Input formats handled (from get_alignment_tool_outputs()):
    - Batch local tool output (ConceptMappingLocalTool, is_batch=True)::

        {
            "input": "<batch>",
            "output": {
                "scRNA-seq": {"ontology_id": "...", "ontology_label": "...", "ontology": "..."},
                "CRISPR":    {"ontology_id": "...", "ontology_label": "...", "ontology": "..."}
            }
        }

    - Single local/BioPortal tool output::

        {"input": "scRNA-seq", "output": {"ontology_id": "...", "ontology_label": "...", "ontology": "..."}}

    Only non-None values are injected so existing entity fields are never overwritten with None.

    Args:
        entities: List of entity dicts (modified in place).
        session_outputs: Output of get_alignment_tool_outputs() — list of
            {"input": str, "output": {ontology_id / ontology_label / ontology / ...}}.

    Returns:
        Number of entities that received at least one concept mapping field.

    Called in:
        app.py — after the alignment stage when task_type == "ner".
    """
    if not entities or not session_outputs:
        return 0

    # Fields to copy into entity dicts.
    MAPPING_FIELDS = ("ontology_id", "ontology_label", "ontology")

    # Build term → {ontology_id, ontology_label, ontology} lookup (case-insensitive).
    # Handles both batch format (output is dict-of-dicts keyed by term) and single-term format.
    term_to_mapping: Dict[str, Dict[str, Any]] = {}
    for item in session_outputs:
        if not isinstance(item, dict):
            continue
        inp = item.get("input") or ""
        out = item.get("output")
        if not isinstance(out, dict) or "error" in out:
            continue

        # Determine format: batch if output contains keys other than the standard field names.
        _known = {*MAPPING_FIELDS, "error"}
        is_batch = any(k not in _known for k in out)
        if is_batch:
            # Batch format: {term: {ontology_id, ontology_label, ontology}, ...}
            for term, mapping in out.items():
                if isinstance(mapping, dict) and "error" not in mapping and term:
                    term_to_mapping[term.lower()] = {f: mapping.get(f) for f in MAPPING_FIELDS}
        else:
            # Single-term format: output IS the mapping dict
            if inp:
                term_to_mapping[inp.lower()] = {f: out.get(f) for f in MAPPING_FIELDS}

    if not term_to_mapping:
        return 0

    enriched = 0
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ent_text = (ent.get("entity") or "").lower().strip()
        if not ent_text:
            continue
        mapping = term_to_mapping.get(ent_text)
        if mapping:
            # Inject only non-None values so existing fields are not overwritten with None.
            for field, value in mapping.items():
                if value is not None:
                    ent[field] = value
            enriched += 1

    return enriched


def inject_alignment_concept_mapping_into_resources(
    resources: List[Dict[str, Any]],
    session_outputs: list,
) -> int:
    """
    Enrich resource dicts with concept mapping fields from alignment agent tool calls.

    Mirrors inject_alignment_concept_mapping_into_ner_entities but operates on
    resource dicts (which use "name" as their text key, not "entity").

    Resource dict BEFORE injection::

        {"name": "STRING", "type": "Database", "url": "https://string-db.org"}

    Resource dict AFTER injection::

        {"name": "STRING", "type": "Database", "url": "https://string-db.org",
         "ontology_id": "SCR:006272",
         "ontology_label": "STRING",
         "ontology": "SciCrunch"}

    The concept mapping tool is called with resource names during the fast-alignment
    bypass so that no LLM call is needed for the alignment stage on resource tasks.

    Args:
        resources: List of resource dicts (modified in place).
        session_outputs: Output of get_alignment_tool_outputs() — list of
            {"input": str, "output": {ontology_id / ontology_label / ontology / ...}}.

    Returns:
        Number of resources that received at least one concept mapping field.
    """
    if not resources or not session_outputs:
        return 0

    MAPPING_FIELDS = ("ontology_id", "ontology_label", "ontology")

    # Build name → {ontology fields} lookup — same logic as NER injection.
    term_to_mapping: Dict[str, Dict[str, Any]] = {}
    for item in session_outputs:
        if not isinstance(item, dict):
            continue
        inp = item.get("input") or ""
        out = item.get("output")
        if not isinstance(out, dict) or "error" in out:
            continue
        _known = {*MAPPING_FIELDS, "error"}
        is_batch = any(k not in _known for k in out)
        if is_batch:
            for term, mapping in out.items():
                if isinstance(mapping, dict) and "error" not in mapping and term:
                    term_to_mapping[term.lower()] = {f: mapping.get(f) for f in MAPPING_FIELDS}
        else:
            if inp:
                term_to_mapping[inp.lower()] = {f: out.get(f) for f in MAPPING_FIELDS}

    if not term_to_mapping:
        return 0

    enriched = 0
    for res in resources:
        if not isinstance(res, dict):
            continue
        # Resources use "name" as primary text; fall back to "resource_name".
        res_name = (res.get("name") or res.get("resource_name") or "").lower().strip()
        if not res_name:
            continue
        mapping = term_to_mapping.get(res_name)
        if mapping:
            for field, value in mapping.items():
                if value is not None:
                    res[field] = value
            enriched += 1

    return enriched


def _resource_has_rich_mapped_concepts(res: Dict[str, Any]) -> bool:
    """True if resource has multiple mapped_specific_target_concept or any concept with non-N/A id."""
    if not isinstance(res, dict):
        return False
    mst = res.get("mapped_specific_target_concept") or []
    if len(mst) > 1:
        return True
    mtc = res.get("mapped_target_concept") or []
    for c in mtc:
        if isinstance(c, dict) and c.get("id") and str(c.get("id", "")).upper() not in ("N/A", "NA", "NULL", ""):
            return True
    for item in mst:
        inner = item.get("mapped_target_concept") if isinstance(item, dict) else None
        if isinstance(inner, dict) and inner.get("id") and str(inner.get("id", "")).upper() not in ("N/A", "NA", "NULL", ""):
            return True
    return False


def promote_canonical_resources_for_resource_task(final: Dict[str, Any]) -> None:
    """
    For resource/structured_extraction: set final["resources"] to the refined list from
    judge_resource or aligned_resources when present, so concept mapping and final output
    use the same list (with mapped_target_concept etc. from alignment/judge), not the
    extraction-merge list which has empty mapped_*.
    Prefers aligned_resources when they have richer mapped_* (multiple specific_target entries
    or real ontology IDs); otherwise uses judge. Merges judge_score from judge into the
    chosen list when possible.
    """
    if not isinstance(final, dict):
        return
    judge = final.get("judge_resource")
    aligned = final.get("aligned_resources")
    resources = final.get("resources") or []
    judge_flat = _flatten_container_to_list(judge) if judge is not None else []
    aligned_flat = _flatten_container_to_list(aligned) if aligned is not None else []
    flat: List[Dict[str, Any]] = []
    if aligned_flat and any(_resource_has_rich_mapped_concepts(r) for r in aligned_flat):
        flat = aligned_flat
    elif judge_flat:
        flat = judge_flat
    elif aligned_flat:
        flat = aligned_flat
    elif isinstance(resources, list) and resources:
        flat = resources
    else:
        flat = _flatten_container_to_list(resources) if isinstance(resources, dict) else []
    if flat and judge_flat:
        name_to_judge = {}
        for jr in judge_flat:
            if isinstance(jr, dict):
                n = (jr.get("name") or jr.get("resource_name") or "").strip()
                if n:
                    name_to_judge[n] = jr
        for res in flat:
            if not isinstance(res, dict):
                continue
            n = (res.get("name") or res.get("resource_name") or "").strip()
            judge_res = name_to_judge.get(n) if n else (judge_flat[0] if judge_flat and isinstance(judge_flat[0], dict) else None)
            if judge_res and "judge_score" in judge_res:
                res["judge_score"] = judge_res.get("judge_score")
                # Ensure provenance shows judge_agent (rich output like NER)
                res.setdefault("provenance", {})
                if isinstance(res["provenance"], dict):
                    jprovenance = judge_res.get("provenance") or {}
                    res["provenance"]["judge_agent"] = (jprovenance.get("judge_agent") if isinstance(jprovenance, dict) else None) or [
                        "judge_score"
                    ]
    if flat:
        for res in flat:
            if isinstance(res, dict):
                res.pop("judge_resource", None)
                res.pop("aligned_resources", None)
        final["resources"] = flat


def _flatten_container_to_list(container: Any) -> List[Dict[str, Any]]:
    """Turn dict-of-lists (e.g. {'1': [...], '2': [...]}) or list into a single list of items."""
    if isinstance(container, list):
        return [x for x in container if isinstance(x, dict)]
    if isinstance(container, dict):
        out: List[Dict[str, Any]] = []
        for _k, v in container.items():
            if isinstance(v, list):
                out.extend(x for x in v if isinstance(x, dict))
            elif isinstance(v, dict):
                out.append(v)
        return out
    return []


def normalize_final_result_for_output(final: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Keep only the task-specific canonical output and metadata; remove intermediate
    agent containers so the JSON matches NER style (no separate aligned/judge keys).

    - NER: keep entities, key_terms, verification, errors, task_type, elapsed_time, metadata.
    - Resource: keep resources, verification, errors, task_type, elapsed_time.
    """
    if not isinstance(final, dict):
        return final
    task_type = (task_type or "").strip().lower()

    allowed = {"errors", "task_type", "elapsed_time", "verification", "metadata", "human_feedback_skipped"}
    if task_type == "ner":
        allowed |= {"entities", "key_terms"}
        # By this point promote_stage_output_to_canonical() has already promoted
        # judge_ner_terms / aligned_ner_terms / extracted_terms → entities after every
        # pipeline stage, so entities should be at the top level. The promotion below
        # is a last-resort safety net for any edge cases that bypass the pipeline loop.
        for key in INTERMEDIATE_CONTAINER_KEYS_NER:
            final.pop(key, None)
        # Safety-net promotion: if entities still missing, check any remaining containers.
        if not final.get("entities") and not final.get("key_terms"):
            jn = None  # already popped above; promotion handled by promote_stage_output_to_canonical
            if jn and isinstance(jn, dict):
                entities = _flatten_container_to_list(jn)
                final["entities"] = entities
                final["key_terms"] = final.get("key_terms") or []
            else:
                hardcoded = (
                    "judged_structured_information_with_human_feedback",
                    "aligned_structured_information",
                    "extracted_structured_information",
                )
                # Also check any root-level key whose value is a dict with entities/key_terms (not only hardcoded names)
                container_keys = list(hardcoded)
                for k, v in final.items():
                    if k not in hardcoded and isinstance(v, dict) and ("entities" in v or "key_terms" in v):
                        container_keys.append(k)
                for container_key in container_keys:
                    container = final.get(container_key)
                    if container is None:
                        continue
                    if isinstance(container, dict) and ("entities" in container or "key_terms" in container):
                        final["entities"] = container.get("entities") or []
                        final["key_terms"] = container.get("key_terms") or final.get("key_terms") or []
                        break
                    flattened = _flatten_container_to_list(container)
                    if flattened:
                        final["entities"] = flattened
                        final["key_terms"] = final.get("key_terms") or (container.get("key_terms") if isinstance(container, dict) else [])
                        break
        final.pop("judge_ner_terms", None)
        for key in list(final):
            if key not in allowed:
                final.pop(key, None)
        return final
    elif task_type == "resource":
        # Keep canonical resources list (concept mapping runs on final["resources"]); drop intermediate keys only.
        resources = final.get("resources") or []
        if not isinstance(resources, list):
            resources = _flatten_container_to_list(resources) if isinstance(resources, dict) else []
        if not resources and isinstance(final.get("resource"), dict) and _looks_like_resource(final["resource"]):
            resources = [final["resource"]]
        cleaned_resources = [_clean_resource_for_export(r) for r in resources]
        out = {
            "errors": final.get("errors", []),
            "task_type": final.get("task_type", task_type),
            "elapsed_time": final.get("elapsed_time"),
            "resources": cleaned_resources,
            "verification": final.get("verification", {}),
        }
        if final.get("human_feedback_skipped") is not None:
            out["human_feedback_skipped"] = final["human_feedback_skipped"]
        return out
    else:
        # generic: drop intermediate keys only
        for key in list(INTERMEDIATE_CONTAINER_KEYS_NER) + list(INTERMEDIATE_CONTAINER_KEYS_RESOURCE):
            final.pop(key, None)
        return final


# ============================================================
# CONCEPT MAPPING (ONTOLOGY_ID, ONTOLOGY_LABEL, ONTOLOGY)
# ============================================================


def _resource_organize_mapped_concepts(res: Dict[str, Any]) -> None:
    """
    Populate structured mapped_* concept fields from flat ontology_* fields.
    Each concept object follows the NER-like shape: id, label, ontology, and
    concept_mapping_provenance ("tool" | "llm_knowledge") so provenance is clear
    per concept, same as entity-level ontology_id/ontology_label/concept_mapping_provenance in NER.
    When flat fields were filled by the Concept Mapping Tool (provenance "tool"), prefer them
    over alignment-provided mapped_* so that tool usage is reflected in the final output.
    """
    if not isinstance(res, dict):
        return

    def _provenance(res: Dict[str, Any], key: str, default: str = "tool") -> str:
        v = res.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return default

    def _has_flat_tool(prefix: str) -> bool:
        """True if flat ontology fields for this slot were set by the tool (provenance "tool")."""
        prov_key = (prefix + "concept_mapping_provenance") if prefix else "concept_mapping_provenance"
        return _provenance(res, prov_key, "") == "tool"

    def _first(val: Any) -> Any:
        if isinstance(val, list) and val:
            return val[0]
        return val

    # mapped_name_concept from top-level ontology_id / ontology_label / ontology (resource name)
    oid = res.get("ontology_id")
    olabel = res.get("ontology_label")
    oont = res.get("ontology")
    use_name_flat = oid is not None or olabel is not None or oont is not None
    if use_name_flat and (not res.get("mapped_name_concept") or _has_flat_tool("")):
        oid, olabel, oont = _first(oid), _first(olabel), _first(oont)
        prov = _provenance(res, "concept_mapping_provenance")
        res["mapped_name_concept"] = [{"id": oid, "label": olabel, "ontology": oont, "concept_mapping_provenance": prov}]

    # mapped_target_concept from target_ontology_* (prefer flat when tool wrote them so provenance shows "tool")
    tid = res.get("target_ontology_id")
    tlabel = res.get("target_ontology_label")
    tont = res.get("target_ontology")
    use_target_flat = tid is not None or tlabel is not None or tont is not None
    if use_target_flat and (not res.get("mapped_target_concept") or _has_flat_tool("target_")):
        tid, tlabel, tont = _first(tid), _first(tlabel), _first(tont)
        prov = _provenance(res, "target_concept_mapping_provenance")
        res["mapped_target_concept"] = [{"id": tid, "label": tlabel, "ontology": tont, "concept_mapping_provenance": prov}]

    # mapped_specific_target_concept from specific_target + specific_target_ontology_*
    # Do not replace when resource already has multiple entries (from alignment); only set provenance.
    sid = res.get("specific_target_ontology_id")
    slabel = res.get("specific_target_ontology_label")
    sont = res.get("specific_target_ontology")
    st = res.get("specific_target")
    existing_mst = res.get("mapped_specific_target_concept")
    use_specific_flat = st is not None or sid is not None or slabel is not None or sont is not None
    replace_mst = use_specific_flat and (not existing_mst or (len(existing_mst) <= 1 and _has_flat_tool("specific_target_")))
    if replace_mst:
        sid, slabel, sont = _first(sid), _first(slabel), _first(sont)
        if isinstance(st, list):
            st = st[0] if st else (str(st) if st else "")
        if st is None:
            st = ""
        prov = _provenance(res, "specific_target_concept_mapping_provenance")
        mc = {"id": sid, "label": slabel, "ontology": sont, "concept_mapping_provenance": prov}
        res["mapped_specific_target_concept"] = [{"specific_target": str(st) if st else "", "mapped_target_concept": mc}]

    # mapped_type_concept from type_ontology_*
    type_id = res.get("type_ontology_id")
    type_label = res.get("type_ontology_label")
    type_ont = res.get("type_ontology")
    use_type_flat = type_id is not None or type_label is not None or type_ont is not None
    if use_type_flat and (not res.get("mapped_type_concept") or _has_flat_tool("type_")):
        type_id = _first(type_id)
        type_label = _first(type_label)
        type_ont = _first(type_ont)
        prov = _provenance(res, "type_concept_mapping_provenance")
        res["mapped_type_concept"] = [{"id": type_id, "label": type_label, "ontology": type_ont, "concept_mapping_provenance": prov}]

    # mapped_category_concept from category_ontology_*
    cat_id = res.get("category_ontology_id")
    cat_label = res.get("category_ontology_label")
    cat_ont = res.get("category_ontology")
    use_cat_flat = cat_id is not None or cat_label is not None or cat_ont is not None
    if use_cat_flat and (not res.get("mapped_category_concept") or _has_flat_tool("category_")):
        cat_id = _first(cat_id)
        cat_label = _first(cat_label)
        cat_ont = _first(cat_ont)
        prov = _provenance(res, "category_concept_mapping_provenance")
        res["mapped_category_concept"] = [{"id": cat_id, "label": cat_label, "ontology": cat_ont, "concept_mapping_provenance": prov}]

    # Map structured key -> flat provenance key prefix (so we know if the tool was used for this slot)
    _key_to_flat_prefix = {
        "mapped_target_concept": "target_",
        "mapped_specific_target_concept": "specific_target_",
        "mapped_name_concept": "",
        "mapped_type_concept": "type_",
        "mapped_category_concept": "category_",
    }
    # Ensure existing mapped_* have concept_mapping_provenance. If the tool was used for this slot
    # (flat provenance "tool"), use "tool"; otherwise "llm_knowledge". Keep provenance only on the
    # concept itself (inner mapped_target_concept for specific_target), not duplicated on the wrapper.
    for key in (
        "mapped_target_concept",
        "mapped_specific_target_concept",
        "mapped_name_concept",
        "mapped_type_concept",
        "mapped_category_concept",
    ):
        val = res.get(key)
        if not isinstance(val, list):
            continue
        prov_for_slot = "tool" if _has_flat_tool(_key_to_flat_prefix[key]) else "llm_knowledge"
        for item in val:
            if not isinstance(item, dict):
                continue
            # Set provenance on the concept (inner) for mapped_specific_target_concept; on the item for others
            inner = item.get("mapped_target_concept")
            if isinstance(inner, dict):
                if "concept_mapping_provenance" not in inner or not inner["concept_mapping_provenance"]:
                    inner["concept_mapping_provenance"] = prov_for_slot
                # Single provenance: keep only on inner concept, remove from outer wrapper
                item.pop("concept_mapping_provenance", None)
            else:
                if "concept_mapping_provenance" not in item or not item["concept_mapping_provenance"]:
                    item["concept_mapping_provenance"] = prov_for_slot


# Flat concept-mapping keys to remove from resources so output only has structured mapped_* fields.
_RESOURCE_FLAT_ONTOLOGY_KEYS = (
    "ontology_id",
    "ontology_label",
    "ontology",
    "concept_mapping_provenance",
    "target_ontology_id",
    "target_ontology_label",
    "target_ontology",
    "target_concept_mapping_provenance",
    "specific_target_ontology_id",
    "specific_target_ontology_label",
    "specific_target_ontology",
    "specific_target_concept_mapping_provenance",
    "type_ontology_id",
    "type_ontology_label",
    "type_ontology",
    "type_concept_mapping_provenance",
    "category_ontology_id",
    "category_ontology_label",
    "category_ontology",
    "category_concept_mapping_provenance",
)


def _resource_strip_flat_ontology_fields(res: Dict[str, Any]) -> None:
    """Remove flat ontology_* fields from a resource so output only has structured mapped_* and mentions_with_ontology."""
    if not isinstance(res, dict):
        return
    for key in _RESOURCE_FLAT_ONTOLOGY_KEYS:
        res.pop(key, None)


def _concept_has_real_iri(d: Dict[str, Any]) -> bool:
    """True if concept dict has an id that is a real IRI (http/https), not N/A."""
    if not isinstance(d, dict):
        return False
    i = d.get("id") or (d.get("mapped_target_concept") or {}).get("id") if isinstance(d.get("mapped_target_concept"), dict) else None
    if not i or not isinstance(i, str):
        return False
    i = i.strip().upper()
    if i in ("N/A", "NA", "NULL", ""):
        return False
    return i.startswith(("HTTP://", "HTTPS://"))


def ensure_resource_mapped_concepts_provenance(resources: List[Dict[str, Any]]) -> None:
    """
    Ensure every mapped_* concept on each resource has concept_mapping_provenance
    ("tool" or "llm_knowledge"). Use "tool" when the concept has a real ontology IRI
    (from alignment's Concept Mapping Tool); otherwise "llm_knowledge". Call when
    concept mapping is skipped (e.g. user approved) so output still has provenance.
    Accepts a list of resource dicts, or a single resource dict (normalized to list of one).
    """
    if isinstance(resources, dict) and _looks_like_resource(resources):
        resources = [resources]
    if not isinstance(resources, list):
        return
    for res in resources:
        if not isinstance(res, dict):
            continue
        for key in (
            "mapped_target_concept",
            "mapped_specific_target_concept",
            "mapped_name_concept",
            "mapped_type_concept",
            "mapped_category_concept",
        ):
            val = res.get(key)
            if not isinstance(val, list):
                continue
            for item in val:
                if not isinstance(item, dict):
                    continue
                inner = item.get("mapped_target_concept")
                if isinstance(inner, dict):
                    # Always set from IRI: real IRI => "tool" (from Concept Mapping Tool), else "llm_knowledge"
                    inner["concept_mapping_provenance"] = "tool" if _concept_has_real_iri(inner) else "llm_knowledge"
                    item.pop("concept_mapping_provenance", None)
                else:
                    item["concept_mapping_provenance"] = "tool" if _concept_has_real_iri(item) else "llm_knowledge"


def _concept_map_one_term(text: str, tool: Any) -> Dict[str, Any]:  # noqa: ANN401
    """
    Map a single term using ConceptMappingTool. Returns dict with
    ontology_id, ontology_label, ontology (or None if no match/error).
    """
    if not (text and str(text).strip()):
        return {"ontology_id": None, "ontology_label": None, "ontology": None}
    text = str(text).strip()
    try:
        out = tool._run(text=text, max_results=1, ontologies=None)
        if isinstance(out, dict) and "error" not in out:
            return {
                "ontology_id": out.get("ontology_id"),
                "ontology_label": out.get("ontology_label"),
                "ontology": out.get("ontology"),
            }
    except Exception as e:
        logger.debug("Concept mapping failed for %r: %s", text, e)
    return {"ontology_id": None, "ontology_label": None, "ontology": None}


# Keys that are metadata or already handled by task-specific concept mapping (do not collect as "generic" terms)
_GENERIC_SKIP_TOP_KEYS = frozenset(
    {
        "errors",
        "elapsed_time",
        "task_type",
        "verification",
        "metadata",
        "human_feedback_skipped",
        "pipeline_stages",
        "stage_timings",
        "token_usage",
        "context_management",
    }
)
# Keys under which we already do task-specific mapping; when task is ner/resource, skip these to avoid duplicate work
_TASK_SPECIFIC_CONTAINER_KEYS = frozenset({"entities", "resources", "key_terms", "judge_ner_terms"})
# String values that are not meaningful terms for concept mapping
_GENERIC_SKIP_VALUES = frozenset({"true", "false", "null", "yes", "no", "n/a", "na", ""})

# ---------------------------------------------------------------------------
# Concept-term guard
# ---------------------------------------------------------------------------
# Maximum character length for a valid ontology term.
# Real terms (SNOMED, MESH, UBERON, EDAM …) are noun phrases, typically
# 1–8 words.  Anything longer is almost certainly a sentence, a remark,
# or a serialised dict that the LLM accidentally put in the wrong field.
_CONCEPT_TERM_MAX_LEN: int = 120
# Maximum number of whitespace-delimited tokens.
# "feature selection" = 2, "vasoactive intestinal polypeptide-expressing inhibitory neurons" = 5.
# A sentence easily exceeds 10.
_CONCEPT_TERM_MAX_WORDS: int = 10


def _is_valid_concept_term(text: str) -> bool:
    """Return True only when *text* looks like a short ontology / concept term.

    Rejects strings that the LLM commonly produces by mistake:

    1. **Empty / too-short** — blank, single-char, or pure whitespace.
    2. **Too long** — > ``_CONCEPT_TERM_MAX_LEN`` chars (sentences, paragraphs,
       remarks, rationale text).
    3. **Too many words** — > ``_CONCEPT_TERM_MAX_WORDS`` tokens (sentence-like
       phrases, bullet-point summaries).
    4. **Serialised structures** — contains ``{``, ``}``, ``[``, ``]``
       (the LLM sometimes passes a Python dict or JSON blob as a term, e.g.
       ``str({"key_term": "...", "remarks": "...long remarks..."})``)
    5. **Known non-term values** — "true", "false", "null", "yes", "no", …
       (the LLM fills optional fields with these placeholders).
    6. **Numeric-only / version strings** — purely numeric text or version
       patterns like "v2.03.9", "1.3", "4.0" that appeared in the wild
       and map to nonsense ontology entries.

    This function is the **single gate** for every term collected by
    ``apply_concept_mapping_to_result``.  Add new rejection rules here; the
    rest of the function does not need to change.
    """
    if not text or not text.strip():
        return False

    s = text.strip()

    # Rule 2: length cap
    if len(s) > _CONCEPT_TERM_MAX_LEN:
        return False

    # Rule 3: word count cap
    if len(s.split()) > _CONCEPT_TERM_MAX_WORDS:
        return False

    # Rule 4: serialised structures (dicts, lists, JSON fragments)
    if any(ch in s for ch in "{}[]"):
        return False

    # Rule 5: known non-term placeholder values
    if s.lower() in _GENERIC_SKIP_VALUES:
        return False

    # Rule 6: numeric-only or pure version strings (e.g. "1.3", "v4.0", "2.0v3.9")
    import re as _re

    if _re.fullmatch(r"v?\d[\d.\-v]*", s, _re.IGNORECASE):
        return False

    # Rule 1 (secondary): must have at least 2 characters of non-whitespace content
    if len(s.replace(" ", "")) < 2:
        return False

    return True


def _collect_terms_from_result_generic(
    payload: Any,
    path: str = "",
    sanitize_fn: Optional[Callable[[str], str]] = None,
    skip_top_keys: Optional[frozenset] = None,
    max_depth: int = 12,
    min_len: int = 2,
    max_len: int = 600,
    _depth: int = 0,
) -> List[Tuple[str, str]]:
    """
    Recursively collect term-like string values from any nested structure (dict/list).
    Returns list of (sanitized_term, path) for concept mapping. Path is a readable path like
    "judged_terms.refined_metadata.activity.description" or "aligned_terms.items[0].question".
    Used when task type is unknown or extraction so concept mapping works without hardcoded shapes.
    """
    out: List[Tuple[str, str]] = []
    skip_top = skip_top_keys or frozenset()
    sanitize = sanitize_fn or (lambda s: (s or "").strip())

    def is_term_like(s: str) -> bool:
        if not s or len(s) < min_len or len(s) > max_len:
            return False
        s_lower = s.strip().lower()
        if s_lower in _GENERIC_SKIP_VALUES:
            return False
        if s.strip().startswith(("http://", "https://")):
            return False
        if s.strip().replace(".", "").replace("-", "").isdigit():
            return False
        return True

    if _depth > max_depth:
        return out

    if isinstance(payload, dict):
        for k, v in payload.items():
            if path == "" and k in skip_top:
                continue
            next_path = f"{path}.{k}" if path else k
            if isinstance(v, str):
                if is_term_like(v):
                    term = sanitize(v)
                    if term:
                        out.append((term, next_path))
            elif isinstance(v, (dict, list)):
                out.extend(_collect_terms_from_result_generic(v, next_path, sanitize_fn, None, max_depth, min_len, max_len, _depth + 1))
            # skip numbers, bools, None
        return out

    if isinstance(payload, list):
        for i, item in enumerate(payload):
            next_path = f"{path}[{i}]"
            if isinstance(item, str):
                if is_term_like(item):
                    term = sanitize(item)
                    if term:
                        out.append((term, next_path))
            elif isinstance(item, (dict, list)):
                out.extend(_collect_terms_from_result_generic(item, next_path, sanitize_fn, None, max_depth, min_len, max_len, _depth + 1))
        return out

    return out


def apply_concept_mapping_to_result(
    result: Dict[str, Any],
    task_type: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Concept-map terms from the extractor output and add ontology_id, ontology_label,
    ontology to each item. Task-aware:
    - NER: entities (entity text and label) and key_terms.
    - Resource / structured_extraction: resources — name, type, category, target,
      specific_target, and all mention lists (related_models, datasets, tools,
      related_papers, benchmarks); key_terms.
    - Extraction or unknown task_type: no hardcoded shape. Terms are collected
      generically by recursively walking the result and collecting term-like string
      values (with path tracking). result["concept_mapping"] is set so concept
      mapping works for any pipeline output without code changes.

    Expects result to contain some of: entities, resources, key_terms; or any nested
    structure for extraction/unknown task. Also handles judge_ner_terms. Updates
    result in place and returns it.

    Speed: Results are deduplicated by term (one API call per unique term). Optional
    env CONCEPT_MAPPING_MAX_TERMS caps how many unique terms are mapped (rest get null).
    ConceptMappingTool uses an in-memory cache and BIOPORTAL_REQUEST_INTERVAL (lower = faster, risk 429).
    """
    # Prefer local concept mapping service when LOCAL_CONCEPT_MAPPING_URL is set.
    from .conceptmappingtool import _sanitize_text as _sanitize_term  # always available, no API key needed

    tool = None
    local_url = os.getenv("LOCAL_CONCEPT_MAPPING_URL", "http://localhost:8000").strip()
    if local_url:
        try:
            from .conceptmappinglocal import ConceptMappingLocalTool

            tool = ConceptMappingLocalTool()
            logger.info("Concept mapping: using local service (%s)", local_url)
        except (ImportError, Exception) as e:
            logger.warning("Local concept mapping tool unavailable (%s); falling back to BioPortal.", e)
    if tool is None:
        try:
            from .conceptmappingtool import ConceptMappingTool

            tool = ConceptMappingTool()
        except (ValueError, ImportError) as e:
            logger.warning("Concept mapping skipped (no tool available): %s", e)
            t = (task_type or "").strip().lower()
            if t in ("resource", "structured_extraction"):
                resources_list = result.get("resources")
                if (
                    not isinstance(resources_list, list)
                    and isinstance(result.get("resource"), dict)
                    and _looks_like_resource(result["resource"])
                ):
                    resources_list = [result["resource"]]
                    result["resources"] = resources_list
                if isinstance(resources_list, list):
                    ensure_resource_mapped_concepts_provenance(resources_list)
            return result

    # ---- Task-aware: what to map from extractor output ----
    t = (task_type or "").strip().lower()

    # ---- Collect (term_key, target_container, target_key_or_index, suffix) ----
    # term_key: unique string to map (sanitized)
    # target_container: list or dict we'll update
    # target_key_or_index: index (for list) or key (for dict)
    # suffix: entity|label|name|type|category|target|specific_target (for apply-back)
    tasks: List[Tuple[str, Any, Any, Optional[str]]] = []  # (term, container, index_or_key, suffix)
    # For resource mentions: (term, resources_list, resource_index, mention_key, item_index)
    mention_tasks: List[Tuple[str, List[Dict[str, Any]], int, str, int]] = []

    def add_term(term: str, container: Any, index_or_key: Any, suffix: Optional[str] = None) -> None:
        cleaned = _sanitize_term(term) if term is not None else ""
        # _is_valid_concept_term rejects sentences, serialised dicts, version strings,
        # placeholders, and anything else the LLM should not have put in a term field.
        if _is_valid_concept_term(cleaned):
            tasks.append((cleaned, container, index_or_key, suffix))

    # Entities: map entity text; for NER also map entity label
    entities = result.get("entities", [])
    if isinstance(entities, list):
        for i, ent in enumerate(entities):
            if isinstance(ent, dict):
                add_term(ent.get("entity") or "", entities, i, "entity")
                if t == "ner":
                    add_term(ent.get("label") or "", entities, i, "label")

    # Resources: name, target, specific_target always; for resource/structured_extraction add type, category, and mention lists (models, datasets, tools, etc.)
    # Normalize single "resource" (dict) to "resources" (list of one) so the same logic runs for both
    resources = result.get("resources", [])
    if (
        not (isinstance(resources, list) and resources)
        and isinstance(result.get("resource"), dict)
        and _looks_like_resource(result["resource"])
    ):
        result["resources"] = [result["resource"]]
        resources = result["resources"]
    if isinstance(resources, list):
        mention_keys = ("related_models", "models", "datasets", "tools", "related_papers", "papers", "benchmarks")
        for i, res in enumerate(resources):
            if not isinstance(res, dict):
                continue

            # Scalar resource fields may still be arrays from parallel chunk extraction at this point
            # (clean_resource_for_export runs later). Extract a single representative value so
            # _is_valid_concept_term doesn't reject them for containing '[' / ']'.
            def _res_scalar(val: Any) -> str:
                if isinstance(val, list):
                    return _mode_of(val) or (str(val[0]).strip() if val else "")
                return str(val).strip() if val else ""

            add_term(_res_scalar(res.get("name") or res.get("resource_name")), resources, i, "name")
            add_term(_res_scalar(res.get("target")), resources, i, "target")
            add_term(_res_scalar(res.get("specific_target")), resources, i, "specific_target")
            if t in ("resource", "structured_extraction"):
                add_term(_res_scalar(res.get("type")), resources, i, "type")
                add_term(_res_scalar(res.get("category")), resources, i, "category")
                mentions = res.get("mentions") or {}
                if isinstance(mentions, dict):
                    for mk in mention_keys:
                        lst = mentions.get(mk)
                        if isinstance(lst, list):
                            for j, val in enumerate(lst):
                                if isinstance(val, str) and val.strip():
                                    term_clean = _sanitize_term(val)
                                    if _is_valid_concept_term(term_clean):
                                        mention_tasks.append((term_clean, resources, i, mk, j))

    # key_terms: list of strings or dicts -> will become list of {term, ontology_id, ontology_label, ontology}
    #
    # key_terms: list of strings or dicts -> will become list of {term, ontology_id, ontology_label, ontology}
    #
    # WHY WE CHECK MULTIPLE DICT KEYS:
    #   The judge stage converts plain key_term strings into dicts:
    #     {"key_term": "feature selection", "judge_score": 1.0, "remarks": "...long remarks..."}
    #   We explicitly check "key_term" | "term" | "name" to extract the short term text.
    #   All candidates are then validated by _is_valid_concept_term, which rejects sentences,
    #   serialised dicts, version strings, and anything else the LLM should not have put there.
    key_terms_raw = result.get("key_terms", [])
    key_terms_tasks: List[Tuple[str, int]] = []  # (term, index for new list)
    if isinstance(key_terms_raw, list):
        for j, t in enumerate(key_terms_raw):
            if isinstance(t, str):
                term = t
            elif isinstance(t, dict):
                # Try common key names used by extractor / judge outputs
                term = t.get("term") or t.get("key_term") or t.get("name") or None
            else:
                term = None  # skip anything that is not a str or dict
            cleaned = _sanitize_term(term) if term is not None else ""
            if _is_valid_concept_term(cleaned):
                key_terms_tasks.append((cleaned, j))

    # judge_ner_terms: dict id -> list of entity dicts; add ontology_* to each entity
    judge_ner = result.get("judge_ner_terms", {})
    judge_entity_tasks: List[Tuple[Any, str, int, int]] = []  # (list_ref, id, list_idx, entity_idx)
    if isinstance(judge_ner, dict):
        for rid, elist in judge_ner.items():
            if isinstance(elist, list):
                for eidx, ent in enumerate(elist):
                    if isinstance(ent, dict) and (ent.get("entity") or "").strip():
                        judge_entity_tasks.append((elist, rid, eidx, eidx))

    # Generic term collection when task is extraction or unknown: walk result and collect term-like
    # strings so concept mapping works for any output shape without hardcoded task logic.
    generic_term_tasks: List[Tuple[str, str]] = []  # (term, path) for concept_mapping list
    use_generic_collector = t == "extraction" or t not in ("ner", "resource", "structured_extraction")
    if use_generic_collector:
        skip_top = _GENERIC_SKIP_TOP_KEYS | _TASK_SPECIFIC_CONTAINER_KEYS
        raw = _collect_terms_from_result_generic(result, path="", sanitize_fn=_sanitize_term, skip_top_keys=skip_top)
        # Dedupe by term (keep first path) so each unique term is mapped once
        seen = set()
        for term, src in raw:
            if term and term not in seen:
                seen.add(term)
                generic_term_tasks.append((term, src))

    # ---- Build unique term -> mapping result ----
    unique_terms: Dict[str, Dict[str, Any]] = {}
    terms_order: List[str] = []
    for term, container, index_or_key, suffix in tasks:
        if term not in unique_terms:
            terms_order.append(term)
    for term, _ in key_terms_tasks:
        if term not in unique_terms:
            terms_order.append(term)
    for term, _res_list, _res_i, _mk, _j in mention_tasks:
        if term not in unique_terms:
            terms_order.append(term)
    for list_ref, rid, _, eidx in judge_entity_tasks:
        ent = list_ref[eidx] if eidx < len(list_ref) else {}
        if isinstance(ent, dict):
            term = _sanitize_term(ent.get("entity") or "")
            if term and term not in unique_terms:
                terms_order.append(term)
    for term, _ in generic_term_tasks:
        if term and term not in unique_terms:
            terms_order.append(term)

    # Optional cap to speed up large results: only map first N unique terms (env CONCEPT_MAPPING_MAX_TERMS)
    max_terms_to_map: Optional[int] = None
    try:
        n = os.getenv("CONCEPT_MAPPING_MAX_TERMS")
        if n is not None:
            max_terms_to_map = max(1, int(n))
    except (TypeError, ValueError):
        pass
    if max_terms_to_map is not None and len(terms_order) > max_terms_to_map:
        skipped_count = len(terms_order) - max_terms_to_map
        for term in terms_order[max_terms_to_map:]:
            unique_terms[term] = {"ontology_id": None, "ontology_label": None, "ontology": None}
        terms_order = terms_order[:max_terms_to_map]
        logger.info(
            "Concept mapping capped to %s terms (CONCEPT_MAPPING_MAX_TERMS); %s terms skipped (no API call)",
            max_terms_to_map,
            skipped_count,
        )

    # Run concept mapping — single batch call for local tool, per-term parallel for BioPortal
    if hasattr(tool, "_map_terms") and terms_order:
        # ConceptMappingLocalTool: send all terms in one batch (handles internal sub-batching up to 4000/batch)
        logger.info("Concept mapping: batch mode (%d unique terms, task_type=%s)", len(terms_order), task_type)
        term_objects = [{"text": term, "context": None} for term in terms_order]
        try:
            batch_result = tool._map_terms(term_objects, max_results=1)
            for term in terms_order:
                mapping = batch_result.get(term, {})
                if isinstance(mapping, dict) and "error" not in mapping:
                    unique_terms[term] = {
                        "ontology_id": mapping.get("ontology_id"),
                        "ontology_label": mapping.get("ontology_label"),
                        "ontology": mapping.get("ontology"),
                    }
                else:
                    unique_terms[term] = {"ontology_id": None, "ontology_label": None, "ontology": None}
        except Exception as e:
            logger.warning("Batch concept mapping failed (%s); falling back to per-term parallel", e)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_term = {executor.submit(_concept_map_one_term, term, tool): term for term in terms_order}
                for future in as_completed(future_to_term):
                    term = future_to_term[future]
                    try:
                        unique_terms[term] = future.result()
                    except Exception as exc:
                        logger.debug("Concept mapping task failed for %r: %s", term, exc)
                        unique_terms[term] = {"ontology_id": None, "ontology_label": None, "ontology": None}
    else:
        # BioPortal or other tools: parallel per-term
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_term = {executor.submit(_concept_map_one_term, term, tool): term for term in terms_order}
            for future in as_completed(future_to_term):
                term = future_to_term[future]
                try:
                    unique_terms[term] = future.result()
                except Exception as e:
                    logger.debug("Concept mapping task failed for %r: %s", term, e)
                    unique_terms[term] = {"ontology_id": None, "ontology_label": None, "ontology": None}

    # ---- Apply mappings back (provenance = "tool" when mapping comes from Concept Mapping Tool) ----
    def _top1(val: Any) -> Any:
        """Alignment result = top 1 only: single value or first element of list."""
        if isinstance(val, list) and val:
            return val[0]
        return val

    def set_ontology(item: Dict[str, Any], mapping: Dict[str, Any], prefix: str = "", provenance: str = "tool") -> None:
        pre = prefix or ""
        item[pre + "ontology_id"] = _top1(mapping.get("ontology_id"))
        item[pre + "ontology_label"] = _top1(mapping.get("ontology_label"))
        item[pre + "ontology"] = _top1(mapping.get("ontology"))
        item[pre + "concept_mapping_provenance"] = provenance

    for term, container, index_or_key, suffix in tasks:
        mapping = unique_terms.get(term, {})
        if isinstance(container, list) and isinstance(index_or_key, int) and 0 <= index_or_key < len(container):
            item = container[index_or_key]
            if isinstance(item, dict):
                if suffix == "name":
                    set_ontology(item, mapping, provenance="tool")
                elif suffix == "target":
                    set_ontology(item, mapping, "target_", provenance="tool")
                elif suffix == "specific_target":
                    set_ontology(item, mapping, "specific_target_", provenance="tool")
                elif suffix == "type":
                    set_ontology(item, mapping, "type_", provenance="tool")
                elif suffix == "category":
                    set_ontology(item, mapping, "category_", provenance="tool")
                elif suffix == "label":
                    set_ontology(item, mapping, "label_", provenance="tool")
                else:
                    set_ontology(item, mapping, provenance="tool")

    # Resource mentions: build mentions_with_ontology per resource (related_models, datasets, tools, etc.)
    for term, res_list, res_i, mk, j in mention_tasks:
        mapping = unique_terms.get(term, {})
        if res_i >= len(res_list) or not isinstance(res_list[res_i], dict):
            continue
        res = res_list[res_i]
        mentions = res.get("mentions") or {}
        lst = mentions.get(mk) if isinstance(mentions, dict) else None
        if not isinstance(lst, list) or j >= len(lst):
            continue
        if "mentions_with_ontology" not in res:
            res["mentions_with_ontology"] = {}
        if mk not in res["mentions_with_ontology"]:
            res["mentions_with_ontology"][mk] = [{} for _ in range(len(lst))]
        if j < len(res["mentions_with_ontology"][mk]):
            res["mentions_with_ontology"][mk][j] = {
                "name": term,
                "ontology_id": _top1(mapping.get("ontology_id")),
                "ontology_label": _top1(mapping.get("ontology_label")),
                "ontology": _top1(mapping.get("ontology")),
                "concept_mapping_provenance": "tool",
            }

    for list_ref, rid, _, eidx in judge_entity_tasks:
        if not isinstance(list_ref, list) or eidx >= len(list_ref):
            continue
        ent = list_ref[eidx]
        if isinstance(ent, dict):
            term = _sanitize_term(ent.get("entity") or "")
            set_ontology(ent, unique_terms.get(term, {}), prefix="", provenance="tool")

    # key_terms: replace with list of {term, ontology_id, ontology_label, ontology, concept_mapping_provenance}
    if key_terms_tasks:
        new_key_terms = []
        for term, j in key_terms_tasks:
            m = unique_terms.get(term, {})
            new_key_terms.append(
                {
                    "term": term,
                    "ontology_id": _top1(m.get("ontology_id")),
                    "ontology_label": _top1(m.get("ontology_label")),
                    "ontology": _top1(m.get("ontology")),
                    "concept_mapping_provenance": "tool",
                }
            )
        result["key_terms"] = new_key_terms

    # Generic / extraction: add concept_mapping list so output reflects tool usage for any task shape
    if use_generic_collector and generic_term_tasks:
        result["concept_mapping"] = []
        for term, source in generic_term_tasks:
            m = unique_terms.get(term, {})
            result["concept_mapping"].append(
                {
                    "term": term,
                    "source": source,
                    "ontology_id": _top1(m.get("ontology_id")),
                    "ontology_label": _top1(m.get("ontology_label")),
                    "ontology": _top1(m.get("ontology")),
                    "concept_mapping_provenance": "tool",
                }
            )

    # Resources: ensure flat provenance is set so _resource_organize_mapped_concepts sees "tool" for each slot we process (handles both "resources" list and single "resource" normalized above)
    resources_list = result.get("resources")
    if not isinstance(resources_list, list) and isinstance(result.get("resource"), dict) and _looks_like_resource(result["resource"]):
        result["resources"] = [result["resource"]]
        resources_list = result["resources"]
    if isinstance(resources_list, list):
        for res in resources_list:
            if not isinstance(res, dict):
                continue
            prov = res.get("provenance") or {}
            if isinstance(prov, dict) and prov.get("concept_mapping_provenance") == "tool":
                res["target_concept_mapping_provenance"] = res.get("target_concept_mapping_provenance") or "tool"
            if res.get("mapped_target_concept_provenance") == "tool":
                res["target_concept_mapping_provenance"] = "tool"
            mst_provens = res.get("mapped_specific_target_concept_provenances")
            if isinstance(mst_provens, list) and any(isinstance(p, dict) and p.get("provenance") == "tool" for p in mst_provens):
                res["specific_target_concept_mapping_provenance"] = res.get("specific_target_concept_mapping_provenance") or "tool"
            # If any mapped concept has a real IRI (tool output), mark as tool
            for item in res.get("mapped_specific_target_concept") or []:
                inner = item.get("mapped_target_concept") if isinstance(item, dict) else None
                if isinstance(inner, dict) and inner.get("id") and str(inner.get("id", "")).startswith(("http://", "https://")):
                    res["specific_target_concept_mapping_provenance"] = res.get("specific_target_concept_mapping_provenance") or "tool"
                    break
            mtc_list = res.get("mapped_target_concept") or []
            if isinstance(mtc_list, list):
                for c in mtc_list:
                    if isinstance(c, dict) and c.get("id") and str(c.get("id", "")).startswith(("http://", "https://")):
                        res["target_concept_mapping_provenance"] = res.get("target_concept_mapping_provenance") or "tool"
                        break
            # Only set flat provenance from alignment/real IRI above; do not force "tool" for every field (would mislabel llm_knowledge as tool)
        for res in resources_list:
            if isinstance(res, dict):
                _resource_organize_mapped_concepts(res)
                _resource_strip_flat_ontology_fields(res)

    return result


# ============================================================
# RESOURCE CLEAN OUTPUT FOR EXPORT
# ============================================================


def _mode_of(vals: List[Any]) -> Optional[str]:
    """Return the most common non-empty string value from a list, or None."""
    flat = [s for v in vals if v is not None for s in (str(v).strip(),) if s]
    if not flat:
        return None
    return Counter(flat).most_common(1)[0][0]


def _best_description(vals: Any) -> str:
    """Pick the longest unique non-empty description from an array (passthrough if string)."""
    if isinstance(vals, str):
        return vals.strip()
    if not isinstance(vals, list):
        return str(vals).strip() if vals else ""
    unique = list(dict.fromkeys(s for v in vals if v for s in (str(v).strip(),) if s))
    return max(unique, key=len) if unique else ""


def _first_url(vals: List[Any]) -> Optional[str]:
    """Return the first URL-looking value from a list, or None."""
    items: List[Any] = list(vals)
    for v in items:
        s = str(v).strip() if v is not None else ""
        if s.startswith(("http://", "https://")):
            return s
    for v in items:
        s = str(v).strip() if v is not None else ""
        if s:
            return s
    return None


def _clean_resource_for_export(res: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a clean, deduplicated resource dict for final output.

    Parallel extraction chunks produce array-valued scalar fields (name, type,
    category, description, target, specific_target, performance, url,
    model_architecture).  This function collapses each array to a single
    canonical value, merges ``mentions_with_ontology`` as the canonical
    ``mentions``, deduplicates ``key_features``, removes empty list fields,
    and strips internal provenance bookkeeping from the exported record.
    """
    if not isinstance(res, dict):
        return res

    out = dict(res)  # shallow copy — do not mutate caller's dict

    # resource_name → name (resource_name is the chunk-level key; name is the canonical export key)
    if "resource_name" in out and "name" not in out:
        out["name"] = out.pop("resource_name")
    else:
        out.pop("resource_name", None)

    name_val = out.get("name")
    if isinstance(name_val, list):
        unique_names = list(dict.fromkeys(s for v in name_val if v for s in (str(v).strip(),) if s))
        out["name"] = unique_names[0] if unique_names else ""

    # description: longest unique value
    desc_val = out.get("description")
    if isinstance(desc_val, list):
        out["description"] = _best_description(desc_val)

    # mode-picked categorical fields
    for field in ("type", "category", "target", "specific_target"):
        val = out.get(field)
        if isinstance(val, list):
            picked = _mode_of(val)
            if picked is not None:
                out[field] = picked
            else:
                out.pop(field, None)

    # performance / model_architecture: longest non-empty value from list
    for field in ("performance", "model_architecture"):
        val = out.get(field)
        if isinstance(val, list):
            best = _best_description(val)
            if best:
                out[field] = best
            else:
                out.pop(field, None)

    # url: first URL-looking value from list
    url_val = out.get("url")
    if isinstance(url_val, list):
        picked_url = _first_url(url_val)
        if picked_url:
            out["url"] = picked_url
        else:
            out.pop("url", None)

    # key_features: deduplicate preserving order
    kf = out.get("key_features")
    if isinstance(kf, list):
        kf_items: List[Any] = cast(List[Any], kf)
        kf_strs: List[str] = [str(item).strip() for item in kf_items if item is not None and str(item).strip()]
        deduped = list(dict.fromkeys(kf_strs))
        if deduped:
            out["key_features"] = deduped
        else:
            out.pop("key_features", None)

    # mentions_with_ontology is the richer canonical form (name + ontology fields per mention);
    # replace the flat-string mentions dict with it when available.
    mwo = out.get("mentions_with_ontology")
    if isinstance(mwo, dict) and mwo:
        out["mentions"] = mwo
    out.pop("mentions_with_ontology", None)

    # Remove any remaining empty list fields
    for key in list(out.keys()):
        if isinstance(out[key], list) and not out[key]:
            out.pop(key)

    return out


def clean_resource_output_for_export(result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply :func:`_clean_resource_for_export` to every resource in *result*.

    Returns a new dict with cleaned resources; the original is not modified.
    Safe to call on any result shape — non-resource results pass through.
    """
    if not isinstance(result, dict):
        return result
    resources = result.get("resources")
    if not isinstance(resources, list):
        return result
    out = dict(result)
    out["resources"] = [_clean_resource_for_export(r) for r in resources]
    return out


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def update_model_weights(custom_weights: Dict[str, float]) -> None:
    """
    Update model weights for weighted majority voting.

    Args:
        custom_weights: Dictionary mapping model names to weight values
    """
    MODEL_WEIGHTS.update(custom_weights)
    logger.info(f"Updated model weights: {MODEL_WEIGHTS}")


def add_generic_labels_to_filter(labels: List[str]) -> None:
    """
    Add additional labels to the generic filter list.

    Args:
        labels: List of label names to filter out
    """
    GENERIC_LABELS_TO_REMOVE.update(label.upper() for label in labels)
    logger.info(f"Generic labels to filter: {GENERIC_LABELS_TO_REMOVE}")
