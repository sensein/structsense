import logging
import time
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_text_into_chunks(
    text: Union[str, bytes, dict, list], chunk_size: int = 4000, overlap: int = 100, break_on: Optional[List[str]] = None
) -> List[str]:
    """Split text into overlapping chunks for parallel processing.

    Args:
        text (Union[str, bytes, dict, list]): The input text to split. Can be:
            - str: Direct text input
            - bytes: Binary text that will be decoded
            - dict: Dictionary containing text (will look for 'text' or 'content' key)
            - list: List of text items that will be joined
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        break_on (List[str], optional): List of characters to break on. Defaults to ['.', '\n']

    Returns:
        List[str]: List of text chunks
    """
    start_time = time.time()
    logger.info(f"Starting text chunking with chunk_size={chunk_size}, overlap={overlap}")

    # Convert input to string
    if text is None:
        logger.warning("Received None input, returning empty list")
        return []

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
            logger.info("Successfully decoded bytes input to string")
        except UnicodeDecodeError:
            logger.error("Failed to decode bytes input")
            return []

    elif isinstance(text, dict):
        # Try to find text content in common keys
        for key in ["text", "content", "data", "input"]:
            if key in text and isinstance(text[key], (str, bytes)):
                text = text[key] if isinstance(text[key], str) else text[key].decode("utf-8")
                logger.info(f"Found text content in dictionary key: {key}")
                break
        else:
            logger.error("Could not find text content in dictionary")
            return []

    elif isinstance(text, list):
        # Join list items with newlines
        text = "\n".join(str(item) for item in text)
        logger.info(f"Joined {len(text)} list items into single text")

    elif not isinstance(text, str):
        # Convert any other type to string
        text = str(text)
        logger.info("Converted non-string input to string")

    logger.info(f"Input text length: {len(text)} characters")

    # Default break characters if none provided
    if break_on is None:
        break_on = [".", "\n"]
    logger.info(f"Using break characters: {break_on}")

    chunks = []
    start = 0
    text_length = len(text)

    # Calculate approximate number of chunks
    estimated_chunks = (text_length + chunk_size - 1) // chunk_size
    logger.info(f"Estimated number of chunks: {estimated_chunks}")

    while start < text_length:
        # Calculate end position for this chunk
        end = min(start + chunk_size, text_length)

        # If this is not the last chunk, try to find a good break point
        if end < text_length:
            # Look for the last break character within the last 100 characters
            break_point = -1
            for break_char in break_on:
                last_break = text.rfind(break_char, start, end)
                if last_break > break_point:
                    break_point = last_break

            if break_point != -1:
                end = break_point + 1

        # Add the chunk
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position for next chunk, accounting for overlap
        start = end - overlap if end < text_length else text_length

        # Log progress
        chunk_num = len(chunks)
        logger.info(f"Created chunk {chunk_num}/{estimated_chunks}: {len(chunk)} characters")

    total_time = time.time() - start_time
    logger.info(f"Split text into {len(chunks)} chunks in {total_time:.2f} seconds")
    logger.info(
        f"Chunk sizes: min={min(len(c) for c in chunks)}, max={max(len(c) for c in chunks)}, avg={sum(len(c) for c in chunks)/len(chunks):.1f}"
    )
    return chunks


# def merge_chunk_results(chunk_results: List[dict], result_key: str = "terms") -> dict:
#     """Merge results from multiple chunks into a single result.
#
#     Args:
#         chunk_results (List[dict]): List of results from individual chunks
#         result_key (str): Key in the result dictionary containing the terms/items to merge
#
#     Returns:
#         dict: Combined result with merged terms/items
#     """
#     start_time = time.time()
#     logger.info(f"Starting to merge {len(chunk_results)} chunk results")
#
#     if not chunk_results:
#         logger.warning("No chunk results to merge")
#         return {result_key: []}
#
#     # First, detect the actual key being used across all chunk results
#     detected_key = None
#     possible_keys = [
#         'terms', 'extracted_terms', 'extracted_resources',
#         'extracted_structured_information', 'aligned_terms',
#         'judged_terms', 'resources', 'entities'
#     ]
#
#     # Count occurrences of each possible key
#     key_counts = {}
#     for result in chunk_results:
#         if isinstance(result, dict):
#             for key in possible_keys:
#                 if key in result:
#                     key_counts[key] = key_counts.get(key, 0) + 1
#
#     # Find the most common key
#     if key_counts:
#         detected_key = max(key_counts.items(), key=lambda x: x[1])[0]
#         logger.info(f"Detected key '{detected_key}' present in {key_counts[detected_key]}/{len(chunk_results)} chunk results")
#     else:
#         # Fallback to the provided result_key
#         detected_key = result_key
#         logger.warning(f"No common keys found, using fallback key '{detected_key}'")
#
#     # If the detected key is different from the provided one, use the detected one
#     if detected_key != result_key:
#         logger.info(f"Using detected key '{detected_key}' instead of provided key '{result_key}'")
#         result_key = detected_key
#
#     combined_result = {result_key: []}
#     total_items = 0
#
#     for i, result in enumerate(chunk_results, 1):
#         chunk_start_time = time.time()
#         if result_key in result:
#             items = result[result_key]
#             if isinstance(items, list):
#                 combined_result[result_key].extend(items)
#                 total_items += len(items)
#                 chunk_time = time.time() - chunk_start_time
#                 logger.info(f"Chunk {i}/{len(chunk_results)}: added {len(items)} items (took {chunk_time:.2f}s)")
#             else:
#                 logger.warning(f"Chunk {i}/{len(chunk_results)}: key '{result_key}' contains non-list data: {type(items)}")
#         else:
#             # Try to find any list data in this result
#             found_items = []
#             for key, value in result.items():
#                 if isinstance(value, list) and len(value) > 0:
#                     found_items.extend(value)
#                     logger.info(f"Chunk {i}/{len(chunk_results)}: found {len(value)} items in key '{key}'")
#
#             if found_items:
#                 combined_result[result_key].extend(found_items)
#                 total_items += len(found_items)
#                 chunk_time = time.time() - chunk_start_time
#                 logger.info(f"Chunk {i}/{len(chunk_results)}: added {len(found_items)} items from various keys (took {chunk_time:.2f}s)")
#             else:
#                 logger.warning(f"Chunk {i}/{len(chunk_results)}: missing key '{result_key}' and no list data found")
#
#     total_time = time.time() - start_time
#     logger.info(f"Merged {len(chunk_results)} chunk results into {total_items} total items in {total_time:.2f} seconds")
#     return combined_result

from collections.abc import Mapping
from copy import deepcopy


def merge_json_chunks(chunks):
    """
    Merges a list of JSON chunks (arbitrary nested dict/list structures).
    Handles structures like 'judge_ner_terms' where keys are strings of numbers mapping to lists.
    """

    def merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            result = deepcopy(a)
            for key, b_val in b.items():
                if key in result:
                    result[key] = merge(result[key], b_val)
                else:
                    result[key] = deepcopy(b_val)
            return result
        elif isinstance(a, list) and isinstance(b, list):
            return a + b
        elif a == b:
            return a
        else:
            return [a, b] if not isinstance(a, list) else (a + ([b] if b not in a else []))

    merged_result = {}
    for chunk in chunks:
        merged_result = merge(merged_result, chunk)
    return merged_result
