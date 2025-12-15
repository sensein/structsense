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
            sentence_text = doc.text[sent.start_char:sent.end_char]
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
    if (
        global_start < 0
        or global_end > len(full_text)
        or global_start >= global_end
    ):
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
    Any malformed entities (missing keys, wrong types) are skipped.

    Args:
        full_text: The complete text document
        full_doc: spaCy document for the full text
        chunk: Dictionary with "start" and "text" keys
        chunk_entities: List of entities with "text", "label", "start", "end" keys

    Returns:
        List of validated entities with global offsets and sentence context
    """
    results: List[Dict[str, Any]] = []

    for ent in chunk_entities:
        # Defensive: entity must be a dict
        if not isinstance(ent, dict):
            continue

        ent_text = ent.get("text")
        label = ent.get("label")
        local_start = ent.get("start")
        local_end = ent.get("end")

        # Skip malformed entities
        if not isinstance(ent_text, str) or not isinstance(label, str):
            continue
        if not isinstance(local_start, int) or not isinstance(local_end, int):
            continue

        # Validate text presence and get global offsets
        result = _validate_text_presence(full_text, chunk, ent_text, local_start, local_end)
        if result is None:
            continue

        global_start, global_end = result
        sentence_info = _get_sentence_info_for_span(full_doc, global_start, global_end)

        results.append(
            {
                "text": ent_text,
                "label": label,
                "global_start": global_start,  # Global position in full document
                "global_end": global_end,      # Global position in full document
                "sentence": sentence_info["sentence"],
                "sentence_start": sentence_info["sentence_start_offset"],  # Relative to sentence start
                "sentence_end": sentence_info["sentence_end_offset"],      # Relative to sentence start
            }
        )

    return results


def _merge_ner_entities_with_occurrences(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge entities with the same (text, label) and accumulate all their locations.

    Input entities: {text, label, global_start, global_end, sentence, sentence_start, sentence_end}
    Output:
      {
        "text": str,
        "label": str,
        "occurrences": [
          {
            "start": int,             # Sentence-level position (relative to sentence start)
            "end": int,               # Sentence-level position (relative to sentence start)
            "global_start": int,      # Global position in full document
            "global_end": int,        # Global position in full document
            "sentence": str,           # Full sentence text
          },
          ...
        ]
      }

    Args:
        entities: List of entities with global and sentence-level positions

    Returns:
        List of merged entities with occurrences
    """
    merged: Dict[tuple, Dict[str, Any]] = {}

    for ent in entities:
        key = (ent["text"], ent["label"])
        # Use sentence-level positions for "start" and "end" (as user expects)
        # and include global positions separately
        occ = {
            "start": ent.get("sentence_start", 0),  # Sentence-level position
            "end": ent.get("sentence_end", 0),      # Sentence-level position
            "global_start": ent.get("global_start", ent.get("start", 0)),  # Global position
            "global_end": ent.get("global_end", ent.get("end", 0)),        # Global position
            "sentence": ent.get("sentence", ""),
        }

        if key not in merged:
            merged[key] = {
                "text": ent["text"],
                "label": ent["label"],
                "occurrences": [occ],
            }
        else:
            # Check for duplicate based on global positions
            if not any(
                o["global_start"] == occ["global_start"] and o["global_end"] == occ["global_end"]
                for o in merged[key]["occurrences"]
            ):
                merged[key]["occurrences"].append(occ)

    return list(merged.values())