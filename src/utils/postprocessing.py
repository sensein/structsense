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
from typing import Dict, Any, List, Callable, Optional
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
    "mobashgr/BC5CDR-chem-WLT-384-BioELECTRA-Pubmed-ENS-20-5": 3.0,
    "mobashgr/NCBI-disease-WLT-256-SciBERT-13INS": 2.0,
    "en_core_web_sm": 1.0,
}

# Mapping for model name variations to canonical names
MODEL_NAME_MAPPING = {
    "spacy (en_core_web_sm)": "en_core_web_sm",
    "spacy": "en_core_web_sm",
    "en_core_web_sm": "en_core_web_sm",
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
    text1 = e1.get("text", "").lower().strip()
    text2 = e2.get("text", "").lower().strip()

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
                "text": entity.get("text", "")
            })

        # Select winning label (highest weighted vote)
        winning_label = max(label_votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(label_votes.values())
        weighted_score = label_votes[winning_label] / total_weight if total_weight > 0 else 0.0

        # Use the longest/most complete text for the winning label
        winning_entities = [e for e in group if e.get("label") == winning_label]
        best_entity = max(winning_entities, key=lambda e: len(e.get("text", "")))

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
            "text": best_entity.get("text", ""),
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
        chunk: Dictionary with "text" and "start" keys
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
        chunk: Dictionary with "text" and "start" keys
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
# TASK REGISTRY
# ============================================================
_TASK_POST_PROCESSORS: Dict[str, Callable] = {
    "ner": ner_post_process,
    "extraction": generic_extraction_post_process,
}

_TASK_MERGERS: Dict[str, Callable] = {
    "ner": merge_ner_results,
    "extraction": merge_generic_results,
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