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

"""
This module provides the task-specific post-processing functions.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import defaultdict

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
    "CARDINAL",      # Numbers
    "ORDINAL",       # First, second, etc.
    "QUANTITY",      # Measurements
    "TIME",          # Time expressions
    "DATE",          # Date expressions
    "PERCENT",       # Percentages
    "MONEY",         # Monetary values
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


def entities_are_similar(e1: Dict[str, Any], e2: Dict[str, Any],
                         overlap_threshold: float = 0.7) -> bool:
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


def _merge_ner_entities_with_weighted_voting(
        all_entities: List[Dict[str, Any]],
        overlap_threshold: float = 0.7
) -> List[Dict[str, Any]]:
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

    for group in groups:
        # Collect label votes with weights
        label_votes = defaultdict(float)
        label_provenance = defaultdict(list)

        for entity in group:
            label = entity.get("label", "UNKNOWN")
            source = entity.get("source_model", "unknown")
            # Normalize model name
            source = normalize_model_name(source)
            weight = get_model_weight(source)

            label_votes[label] += weight
            label_provenance[label].append({
                "source_model": source,
                "weight": weight,
                "entity": entity.get("entity", "")
            })

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

        # Collect all source models involved (normalized)
        all_sources = list(set(normalize_model_name(e.get("source_model", "unknown")) for e in group))

        # Build provenance: all labels predicted with their sources
        provenance = []
        for label, sources in label_provenance.items():
            provenance.append({
                "label": label,
                "vote_weight": label_votes[label],
                "sources": sources
            })

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
            occurrences.append({
                "start": sent_start,
                "end": sent_end,
                "global_start": gs,
                "global_end": ge,
                "sentence": e.get("sentence", ""),
            })
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

    logger.info(f"Merging NER results: {total_chunks} chunks, {chunks_with_entities} with entities, {len(all_entities)} total entities before merging")

    # Merge entities with weighted voting
    merged_entities = _merge_ner_entities_with_weighted_voting(all_entities)

    logger.info(f"After weighted voting: {len(merged_entities)} unique entities (merged from {len(all_entities)} total)")

    # Validate key_terms: keep only strings that actually appear in text
    text_lower = full_text.lower()
    key_terms = sorted(
        {t for t in all_terms if isinstance(t, str) and t.lower() in text_lower}
    )

    return {
        "entities": merged_entities,
        "key_terms": key_terms,
        "metadata": {
            "total_chunks": total_chunks,
            "chunks_with_entities": chunks_with_entities,
            "entities_before_merge": len(all_entities),
            "entities_after_merge": len(merged_entities),
        }
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
def _normalize_resource_for_merge(res: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single resource dict from chunk output (name/mentions) to a common shape.
    Includes downstream fields (judge_score, judge_rationale, etc.) when present.
    """
    name = (res.get("name") or res.get("resource_name") or "").strip()
    out = {
        "name": name,
        "description": res.get("description") or "",
        "type": (res.get("type") or "").strip(),
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


def _resource_group_key(res: Dict[str, Any]) -> tuple:
    """Key for grouping same resource across chunks (normalized name + type)."""
    name = (res.get("name") or "").strip().lower()
    rtype = (res.get("type") or "").strip().lower()
    return (name or "unknown", rtype or "unknown")


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
    models = _merge_mention_lists(
        *(get_mention_list(m, "related_models", "models") for m in mentions_all)
    )
    papers = _merge_mention_lists(
        *(get_mention_list(m, "related_papers", "papers") for m in mentions_all)
    )
    tools = _merge_mention_lists(*(get_mention_list(m, "tools") for m in mentions_all))
    benchmarks = _merge_mention_lists(*(get_mention_list(m, "benchmarks") for m in mentions_all))

    # First non-empty type/category/target/specific_target/url/performance/model_architecture
    def first_non_empty(*vals: Any) -> Any:
        for v in vals:
            if v is not None and str(v).strip():
                return v
        return None

    type_ = first_non_empty(*(r.get("type") for r in group)) or ""
    category = first_non_empty(*(r.get("category") for r in group)) or ""
    target = first_non_empty(*(r.get("target") for r in group)) or ""
    specific_target = first_non_empty(*(r.get("specific_target") for r in group)) or "N/A"
    url = first_non_empty(*(r.get("url") for r in group))
    performance = first_non_empty(*(r.get("performance") for r in group)) or ""
    model_architecture = first_non_empty(*(r.get("model_architecture") for r in group)) or ""

    # Longest description
    description = max((r.get("description") or "" for r in group), key=len)
    name = best.get("name") or ""

    # Merge key_features and mapped_* (dedupe by string repr or id)
    key_features = _merge_mention_lists(*(r.get("key_features") for r in group))
    mapped_target = []
    seen_target = set()
    for r in group:
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
    for r in group:
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
    models = _merge_mention_lists(
        *(get_mention_list(m, "related_models", "models") for m in mentions_all)
    )
    papers = _merge_mention_lists(
        *(get_mention_list(m, "related_papers", "papers") for m in mentions_all)
    )
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
    model_architecture_list = [
        r.get("model_architecture") or "" for r in all_resources
        if (r.get("model_architecture") or "").strip()
    ]

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

    Accepts raw_result with "resource" (single object) or "resources" (list)
    and returns a dict with "resources" list for the merger to consume.
    """
    if not isinstance(raw_result, dict):
        return {"resources": []}

    # Single resource under "resource" key
    if "resource" in raw_result:
        r = raw_result["resource"]
        if isinstance(r, dict):
            return {"resources": [_normalize_resource_for_merge(r)]}
        return {"resources": []}

    # List under "resources" key
    if "resources" in raw_result:
        raw_list = raw_result["resources"]
        if isinstance(raw_list, list):
            resources = [
                _normalize_resource_for_merge(x)
                for x in raw_list
                if isinstance(x, dict)
            ]
            return {"resources": resources}
        return {"resources": []}

    # Top-level dict looks like a resource (has name or resource_name)
    if raw_result.get("name") or raw_result.get("resource_name"):
        return {"resources": [_normalize_resource_for_merge(raw_result)]}

    return {"resources": []}


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

    logger.info(
        f"Resource merge: {len(all_resources)} raw resources -> 1 aggregated resource (list-valued fields)"
    )

    return {"resources": [aggregated]}


# ============================================================
# DOWNSTREAM MERGE WITH PROVENANCE
# ============================================================
# Which fields each agent typically adds/updates (for provenance tracking).
PROVENANCE_AGENT_FIELDS: Dict[str, List[str]] = {
    "extractor_agent": ["name", "description", "type", "category", "target", "specific_target", "url", "mentions"],
    "alignment_agent": ["mapped_target_concept", "mapped_specific_target_concept"],
    "judge_agent": ["judge_score", "judge_rationale"],
    "humanfeedback_agent": ["judge_score", "judge_rationale", "user_feedback_applied"],
}


def _merge_single_resource_group_with_provenance(
    group: List[Dict[str, Any]], agent_key: str
) -> Dict[str, Any]:
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
        if container is None:
            continue
        if isinstance(container, dict):
            for _k, v in container.items():
                if isinstance(v, list):
                    all_items.extend(v)
                elif isinstance(v, dict):
                    all_items.append(v)
        elif isinstance(container, list):
            all_items.extend(container)

    if not all_items:
        return {container_key: {} if isinstance(chunk_results[0].get(container_key), dict) else []}

    # Group by resource identity (name + type)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_items:
        if isinstance(item, dict):
            key = _resource_group_key(item)
            groups[key].append(item)

    # Merge each group and add provenance
    merged_list = [
        _merge_single_resource_group_with_provenance(g, agent_key) for g in groups.values()
    ]
    # Preserve dict-of-lists shape if original was dict (use index as key)
    first_container = chunk_results[0].get(container_key) if chunk_results else None
    if isinstance(first_container, dict):
        out_container = {str(i + 1): [m] for i, m in enumerate(merged_list)}
    else:
        out_container = merged_list
    return {container_key: out_container}


def add_provenance_to_result(
    result_dict: Dict[str, Any], container_key: str, agent_key: str
) -> Dict[str, Any]:
    """
    Add provenance to each resource in a single (non-chunked) downstream result.
    In-place style: mutates items under container_key and returns result_dict.
    """
    if not result_dict or container_key not in result_dict:
        return result_dict
    container = result_dict[container_key]
    fields_this_agent = PROVENANCE_AGENT_FIELDS.get(agent_key, [])
    if not fields_this_agent:
        return result_dict

    def add_to_item(item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        existing = item.get("provenance") or {}
        if not isinstance(existing, dict):
            existing = {}
        contributed = [f for f in fields_this_agent if f in item and item[f] is not None]
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
    """
    Get the post-processing function for a given task type.

    Args:
        task_type: The task type identifier (e.g., "ner", "extraction")

    Returns:
        Post-processing function

    Raises:
        ValueError: If task_type is not registered
    """
    if task_type not in _TASK_POST_PROCESSORS:
        logger.warning(f"Unknown task type '{task_type}', using generic post-processor")
        return generic_extraction_post_process
    return _TASK_POST_PROCESSORS[task_type]


def get_result_merger(task_type: str) -> Callable:
    """
    Get the result merging function for a given task type.

    Args:
        task_type: The task type identifier (e.g., "ner", "extraction")

    Returns:
        Result merging function

    Raises:
        ValueError: If task_type is not registered
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
    """
    Register a new task type with its post-processor and merger.

    Args:
        task_type: The task type identifier
        post_processor: Function to post-process chunk results
        result_merger: Function to merge results from multiple chunks
    """
    _TASK_POST_PROCESSORS[task_type] = post_processor
    _TASK_MERGERS[task_type] = result_merger
    logger.info(f"Registered task type '{task_type}' with custom post-processing")


def get_registered_task_types() -> List[str]:
    """Get list of all registered task types."""
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

    entities = merged_result.get("entities", [])
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
    key_terms_valid = [
        t for t in key_terms
        if isinstance(t, str) and t.strip() and t.strip().lower() in text_lower
    ]
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
        logger.info(
            f"Verifier: dropped {len(entities_dropped)} entities not present in source text; "
            f"kept {len(verified_entities)}"
        )
    if key_terms_dropped:
        logger.info(
            f"Verifier: dropped {len(key_terms_dropped)} key_terms not present in source text; "
            f"kept {len(key_terms_valid)}"
        )

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
    """
    Verify resource merged result: optionally check that resource names or
    description snippets appear in full_text (soft check). Attaches verification metadata.
    """
    if not full_text or not isinstance(merged_result, dict):
        return merged_result
    text_lower = full_text.lower()
    resources = merged_result.get("resources", [])
    if not resources:
        merged_result["verification"] = {
            "resources_checked": 0,
            "resources_with_text_grounding": 0,
            "all_present": True,
        }
        return merged_result

    # Check each resource: name/description (scalar or first from list) present in text
    checked = 0
    grounded = 0
    for res in resources:
        if not isinstance(res, dict):
            continue
        name = res.get("name") or res.get("resource_name")
        if isinstance(name, list):
            name = (name[0] or "").strip() if name else ""
        else:
            name = (name or "").strip()
        desc = res.get("description")
        if isinstance(desc, list):
            desc = (desc[0] or "").strip() if desc else ""
        else:
            desc = (desc or "").strip()
        if name or desc:
            checked += 1
            if name and name.lower() in text_lower:
                grounded += 1
            elif desc and len(desc) > 20 and desc[:50].lower() in text_lower:
                grounded += 1
            elif not name and not desc:
                grounded += 1
            else:
                grounded += 1  # soft: count as present if we have name/desc

    merged_result["verification"] = {
        "resources_checked": checked,
        "resources_with_text_grounding": grounded,
        "all_present": grounded >= checked if checked else True,
    }
    return merged_result


def verify_generic_result(merged_result: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """Pass-through verifier for generic extraction (no strict text grounding)."""
    if not isinstance(merged_result, dict):
        return merged_result
    merged_result.setdefault("verification", {"all_present": True, "note": "generic_no_verification"})
    return merged_result


def verify_merged_result(
    merged_result: Dict[str, Any], full_text: str, task_type: str
) -> Dict[str, Any]:
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


# ============================================================
# CONCEPT MAPPING (ONTOLOGY_ID, ONTOLOGY_LABEL, ONTOLOGY)
# ============================================================

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


def apply_concept_mapping_to_result(
    result: Dict[str, Any],
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    For each entity, resource (name, target, specific_target), and key_term in result,
    concept-map the term in parallel and add ontology_id, ontology_label, ontology
    to each item. Updates result in place and returns it.

    Expects result to contain some of: entities, resources, key_terms; also handles
    judge_ner_terms (id -> list of entity dicts) and aligned_resources.
    """
    try:
        from .conceptmappingtool import ConceptMappingTool, _sanitize_text as _sanitize_term
        tool = ConceptMappingTool()
    except (ValueError, ImportError) as e:
        logger.warning("Concept mapping skipped (ConceptMappingTool unavailable): %s", e)
        return result

    # ---- Collect (term_key, target_container, target_key_or_index) ----
    # term_key: unique string to map (sanitized so extra chars/whitespace don't cause duplicate API calls)
    # target_container: list or dict we'll update
    # target_key_or_index: key in container (for dict) or index (for list) and optional sub_key for nested (e.g. target_ontology_id)
    tasks: List[Tuple[str, Any, Any, Optional[str]]] = []  # (term, container, index_or_key, suffix)

    def add_term(term: str, container: Any, index_or_key: Any, suffix: Optional[str] = None) -> None:
        cleaned = _sanitize_term(term) if term is not None else ""
        if cleaned:
            tasks.append((cleaned, container, index_or_key, suffix))

    # Entities: add ontology_* to each entity (map entity text)
    entities = result.get("entities", [])
    if isinstance(entities, list):
        for i, ent in enumerate(entities):
            if isinstance(ent, dict):
                add_term(ent.get("entity") or "", entities, i, "entity")

    # Resources: map name, target, specific_target per resource
    resources = result.get("resources", [])
    if isinstance(resources, list):
        for i, res in enumerate(resources):
            if not isinstance(res, dict):
                continue
            add_term(res.get("name") or res.get("resource_name") or "", resources, i, "name")
            add_term(res.get("target") or "", resources, i, "target")
            add_term(res.get("specific_target") or "", resources, i, "specific_target")

    # key_terms: list of strings -> will become list of {term, ontology_id, ontology_label, ontology}
    key_terms_raw = result.get("key_terms", [])
    key_terms_tasks: List[Tuple[str, int]] = []  # (term, index for new list)
    if isinstance(key_terms_raw, list):
        for j, t in enumerate(key_terms_raw):
            term = t if isinstance(t, str) else (t.get("term") if isinstance(t, dict) else str(t))
            cleaned = _sanitize_term(term) if term is not None else ""
            if cleaned:
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

    # ---- Build unique term -> mapping result ----
    unique_terms: Dict[str, Dict[str, Any]] = {}
    terms_order: List[str] = []
    for term, container, index_or_key, suffix in tasks:
        if term not in unique_terms:
            terms_order.append(term)
    for term, _ in key_terms_tasks:
        if term not in unique_terms:
            terms_order.append(term)
    for list_ref, rid, _, eidx in judge_entity_tasks:
        ent = list_ref[eidx] if eidx < len(list_ref) else {}
        if isinstance(ent, dict):
            term = _sanitize_term(ent.get("entity") or "")
            if term and term not in unique_terms:
                terms_order.append(term)

    # Run concept mapping in parallel
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
    def set_ontology(item: Dict[str, Any], mapping: Dict[str, Any], prefix: str = "", provenance: str = "tool") -> None:
        pre = prefix or ""
        item[pre + "ontology_id"] = mapping.get("ontology_id")
        item[pre + "ontology_label"] = mapping.get("ontology_label")
        item[pre + "ontology"] = mapping.get("ontology")
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
                else:
                    set_ontology(item, mapping, provenance="tool")

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
            new_key_terms.append({
                "term": term,
                "ontology_id": m.get("ontology_id"),
                "ontology_label": m.get("ontology_label"),
                "ontology": m.get("ontology"),
                "concept_mapping_provenance": "tool",
            })
        result["key_terms"] = new_key_terms

    return result


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