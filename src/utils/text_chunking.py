import logging
import time
from typing import List, Optional, Union, Dict, Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _chunk_doc_by_sentences(doc, max_chars: int) -> List[Dict[str, Any]]:
    """
    Split a document into text chunks based on sentence boundaries, ensuring
    each chunk does not exceed a maximum character length.

    Sentences are accumulated sequentially until adding another sentence would
    exceed `max_chars`. Each chunk records both the extracted text and its starting
    character offset in the original document.

    Args:
        doc: A spaCy-like document object with `.text` and `.sents` attributes.
        max_chars (int): Maximum number of characters allowed per chunk.
            If `max_chars` is less than or equal to 0, or the document text is
            shorter than `max_chars`, the entire document is returned as a
            single chunk.

    Returns:
        List[Dict[str, Any]]: A list of chunks, where each chunk is a dictionary
        containing:
            - "text": The chunk's text.
            - "start": The starting character index of the chunk in `doc.text`.
    """
    text = doc.text
    if max_chars <= 0 or len(text) <= max_chars:
        return [{"text": text, "start": 0}]

    chunks: List[Dict[str, Any]] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for sent in doc.sents:
        s_start, s_end = sent.start_char, sent.end_char

        if cur_start is None:
            cur_start, cur_end = s_start, s_end
            continue

        if s_end - cur_start <= max_chars:
            cur_end = s_end
        else:
            chunks.append({"text": text[cur_start:cur_end], "start": cur_start})
            cur_start, cur_end = s_start, s_end

    if cur_start is not None:
        chunks.append({"text": text[cur_start:cur_end], "start": cur_start})

    return chunks


def _get_sentence_info_for_span(doc, start: int, end: int) -> Dict[str, Any]:
    """
    Given global start/end char offsets, return sentence information including
    sentence text, sentence start/end, and entity positions relative to sentence.

    Args:
        doc: A spaCy-like document object with `.text` and `.sents` attributes.
        start: Global start character offset
        end: Global end character offset

    Returns:
        Dict containing:
            "sentence": str,
            "sentence_start": int,  # Global start of sentence
            "sentence_end": int,     # Global end of sentence
            "sentence_start_offset": int,  # Entity start relative to sentence start
            "sentence_end_offset": int,    # Entity end relative to sentence start
    """
    for sent in doc.sents:
        if sent.start_char <= start < sent.end_char:
            sentence_text = doc.text[sent.start_char : sent.end_char]
            sentence_start_offset = start - sent.start_char
            sentence_end_offset = end - sent.start_char
            return {
                "sentence": sentence_text,
                "sentence_start": sent.start_char,
                "sentence_end": sent.end_char,
                "sentence_start_offset": sentence_start_offset,
                "sentence_end_offset": sentence_end_offset,
            }
    # Fallback if sentence not found
    sentence_text = doc.text[start:end]
    return {
        "sentence": sentence_text,
        "sentence_start": start,
        "sentence_end": end,
        "sentence_start_offset": 0,
        "sentence_end_offset": end - start,
    }


def _validate_text_presence(
    full_text: str,
    chunk: Dict[str, Any],
    text: str,
    local_start: int,
    local_end: int,
) -> Optional[tuple[int, int]]:
    """
    Validates that the given text exists at the specified position in full_text.
    If not found at the exact position, searches within the chunk region.

    Args:
        full_text: The complete text document
        chunk: Dictionary with "start" and "text" keys
        text: The text to validate
        local_start: Start offset relative to chunk start
        local_end: End offset relative to chunk start

    Returns:
        Tuple (global_start, global_end) if text is found, None otherwise
    """
    chunk_start = chunk["start"]
    chunk_text = chunk["text"]

    # Compute naive global offsets
    global_start = chunk_start + local_start
    global_end = chunk_start + local_end

    # Basic bounds check
    if global_start < 0 or global_end > len(full_text) or global_start >= global_end:
        return None

    # Check that text at that span matches the entity text
    slice_text = full_text[global_start:global_end]
    if slice_text == text:
        return (global_start, global_end)

    # Fallback: search within this chunk's region in the full text
    region_start = chunk_start
    region_end = min(chunk_start + len(chunk_text), len(full_text))
    region = full_text[region_start:region_end]
    rel_pos = region.find(text)
    if rel_pos == -1:
        # Drop this entity if we can't verify it exists in the original text
        return None

    global_start = region_start + rel_pos
    global_end = global_start + len(text)
    return (global_start, global_end)


def _globalize_entities(
    full_text: str,
    full_doc,
    chunk: Dict[str, Any],
    chunk_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert chunk-local entity offsets to global offsets, validate that
    the entity text is actually present in the original text, and attach
    the sentence containing it.

    Uses _validate_text_presence to ensure text exists at the specified positions.
    If entities don't have start/end positions, searches for the text in the chunk.
    Any malformed entities (missing text or label) are skipped.

    Args:
        full_text: The complete text document
        full_doc: spaCy document for the full text
        chunk: Dictionary with "start" and "text" keys
        chunk_entities: List of entities with "text", "label", optionally "start", "end" keys

    Returns:
        List of validated entities with global offsets and sentence context
    """
    results: List[Dict[str, Any]] = []
    chunk_start = chunk.get("start", 0)
    chunk_text = chunk.get("text", "")

    for ent in chunk_entities:
        # Defensive: entity must be a dict
        if not isinstance(ent, dict):
            continue

        # Accept both "entity" and "text" (task/LLM often use "entity", NER tool uses "entity")
        ent_text = ent.get("entity") or ent.get("text")
        if isinstance(ent_text, str):
            ent_text = ent_text.strip()
        else:
            ent_text = ""
        label = ent.get("label")
        local_start = ent.get("start")
        local_end = ent.get("end")

        # Skip malformed entities (must have text and label)
        if not ent_text or not isinstance(label, str):
            continue

        # Handle entities with or without positions
        if isinstance(local_start, int) and isinstance(local_end, int):
            # Entity has positions - validate and use them
            result = _validate_text_presence(full_text, chunk, ent_text, local_start, local_end)
            if result is None:
                # Position validation failed, try searching
                result = _find_text_in_chunk(full_text, chunk, ent_text)
        else:
            # Entity doesn't have positions - search for text in chunk
            result = _find_text_in_chunk(full_text, chunk, ent_text)

        if result is None:
            # Text not found in chunk, skip this entity
            logger.debug(f"Entity '{ent_text}' not found in chunk, skipping")
            continue

        global_start, global_end = result
        sentence_info = _get_sentence_info_for_span(full_doc, global_start, global_end)

        result_ent = {
            "entity": ent_text,  # Pipeline and verifier expect "entity"
            "text": ent_text,
            "label": label,
            "global_start": global_start,
            "global_end": global_end,
            "sentence": sentence_info["sentence"],
            "sentence_start": sentence_info["sentence_start_offset"],
            "sentence_end": sentence_info["sentence_end_offset"],
        }

        # Preserve source_model if present
        if "source_model" in ent:
            result_ent["source_model"] = ent["source_model"]

        results.append(result_ent)

    return results


def _find_text_in_chunk(
    full_text: str,
    chunk: Dict[str, Any],
    text: str,
) -> Optional[tuple[int, int]]:
    """
    Find text in the chunk region of full_text.

    Args:
        full_text: The complete text document
        chunk: Dictionary with "start" and "text" keys
        text: The text to find

    Returns:
        Tuple (global_start, global_end) if text is found, None otherwise
    """
    chunk_start = chunk.get("start", 0)
    chunk_text = chunk.get("text", "")

    # Search within this chunk's region in the full text
    region_start = chunk_start
    region_end = min(chunk_start + len(chunk_text), len(full_text))
    region = full_text[region_start:region_end]

    # Try exact match first (case-sensitive)
    rel_pos = region.find(text)
    if rel_pos == -1:
        # Try case-insensitive match
        region_lower = region.lower()
        text_lower = text.lower()
        rel_pos = region_lower.find(text_lower)
        if rel_pos == -1:
            # Try normalized whitespace matching
            region_normalized = " ".join(region.split())
            text_normalized = " ".join(text.split())
            rel_pos_normalized = region_normalized.find(text_normalized)
            if rel_pos_normalized == -1:
                return None
            # Map back to original position (approximate)
            # This is a fallback - may not be perfectly accurate
            rel_pos = region.find(text_normalized[:20]) if len(text_normalized) > 20 else -1
            if rel_pos == -1:
                return None

    global_start = region_start + rel_pos
    global_end = global_start + len(text)

    # Verify the found text matches
    if global_end > len(full_text):
        return None

    found_text = full_text[global_start:global_end]
    if found_text.lower() != text.lower():
        # Found position doesn't match exactly, return None
        return None

    return (global_start, global_end)


def _merge_ner_entities_with_occurrences(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge entities with the same (text, label) and accumulate all their locations.

    Input entities: {text, label, global_start, global_end, sentence, sentence_start, sentence_end, source_model?}
    Output:
      {
        "text": str,
        "label": str,
        "source_models": [str],       # List of all unique source models that detected this entity
        "occurrences": [
          {
            "start": int,             # Sentence-level position (relative to sentence start)
            "end": int,               # Sentence-level position (relative to sentence start)
            "global_start": int,      # Global position in full document
            "global_end": int,        # Global position in full document
            "sentence": str,           # Full sentence text
            "source_model": str,      # Source model for this specific occurrence (if available)
          },
          ...
        ]
      }

    Args:
        entities: List of entities with global and sentence-level positions, optionally including source_model

    Returns:
        List of merged entities with occurrences and source model information
    """
    merged: Dict[tuple, Dict[str, Any]] = {}

    for ent in entities:
        key = (ent["text"], ent["label"])
        # Use sentence-level positions for "start" and "end" (as user expects)
        # and include global positions separately
        occ = {
            "start": ent.get("sentence_start", 0),  # Sentence-level position
            "end": ent.get("sentence_end", 0),  # Sentence-level position
            "global_start": ent.get("global_start", ent.get("start", 0)),  # Global position
            "global_end": ent.get("global_end", ent.get("end", 0)),  # Global position
            "sentence": ent.get("sentence", ""),
        }

        # Preserve source_model in occurrence if present
        if "source_model" in ent:
            occ["source_model"] = ent["source_model"]

        if key not in merged:
            merged[key] = {
                "text": ent["text"],
                "label": ent["label"],
                "occurrences": [occ],
                "source_models": set(),  # Collect all unique source models
            }
            # Add source_model to the set if present
            if "source_model" in ent:
                merged[key]["source_models"].add(ent["source_model"])
        else:
            # Check for duplicate based on global positions
            if not any(
                o["global_start"] == occ["global_start"] and o["global_end"] == occ["global_end"] for o in merged[key]["occurrences"]
            ):
                merged[key]["occurrences"].append(occ)
                # Add source_model to the set if present
                if "source_model" in ent:
                    merged[key]["source_models"].add(ent["source_model"])

    # Convert sets to sorted lists for JSON serialization
    result = []
    for merged_ent in merged.values():
        merged_ent["source_models"] = sorted(list(merged_ent["source_models"])) if merged_ent["source_models"] else []
        result.append(merged_ent)

    return result
