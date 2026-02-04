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
# @File    : context_window_manager.py
# @Software: PyCharm

"""
Context Window Manager for Multi-Agent Token Management.

Handles intelligent context compression, summarization, and token-aware
input preparation for downstream agents.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import tiktoken

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    Manages context windows and token limits for multi-agent pipelines.

    Features:
    - Token counting and validation
    - Intelligent context compression
    - Hierarchical summarization
    - Priority-based content selection
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        encoding_model: str = "cl100k_base",
        reserve_tokens: int = 2000
    ):
        """
        Initialize context window manager.

        Args:
            max_tokens: Maximum tokens for model context
            encoding_model: Tokenizer model name
            reserve_tokens: Tokens to reserve for prompts and responses
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_model)
            logger.info(f"Initialized tokenizer: {encoding_model}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {encoding_model}: {e}. Using character approximation.")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}. Using character approximation.")
                return len(text) // 4
        else:
            # Fallback: approximate 1 token = 4 characters
            return len(text) // 4

    def estimate_tokens(self, data: Any) -> int:
        """
        Estimate tokens for any data structure.

        Args:
            data: Data to estimate tokens for (dict, list, str, etc.)

        Returns:
            Estimated token count
        """
        if isinstance(data, str):
            return self.count_tokens(data)
        else:
            # Convert to JSON string and count
            try:
                json_str = json.dumps(data, default=str)
                return self.count_tokens(json_str)
            except Exception:
                # Fallback: rough estimate
                return len(str(data)) // 4

    def fits_in_context(self, text: str, extra_tokens: int = 0) -> Tuple[bool, int]:
        """
        Check if text fits in available context window.

        Args:
            text: Input text
            extra_tokens: Additional tokens to account for (prompts, etc.)

        Returns:
            Tuple of (fits: bool, token_count: int)
        """
        token_count = self.count_tokens(text)
        total_needed = token_count + extra_tokens
        fits = total_needed <= self.available_tokens

        if not fits:
            logger.warning(
                f"Content exceeds context window: {total_needed} tokens needed, "
                f"{self.available_tokens} available (max: {self.max_tokens}, reserve: {self.reserve_tokens})"
            )

        return fits, token_count

    def prepare_for_downstream_agent(
        self,
        results: Dict[str, Any],
        agent_key: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prepare results for downstream agent consumption with token limit.

        Args:
            results: Results from previous agent
            agent_key: Target agent key (for logging)
            max_tokens: Maximum tokens (uses self.available_tokens if None)

        Returns:
            Prepared results that fit within token limit
        """
        max_tokens = max_tokens or self.available_tokens
        current_tokens = self.estimate_tokens(results)

        if current_tokens <= max_tokens:
            logger.debug(f"Results for {agent_key} fit in context: {current_tokens}/{max_tokens} tokens")
            return results

        logger.warning(
            f"Results for {agent_key} exceed token limit: {current_tokens}/{max_tokens} tokens. "
            "Applying intelligent compression..."
        )

        # Apply compression strategies
        compressed = self._compress_results(results, max_tokens, agent_key)

        # Verify compression worked
        final_tokens = self.estimate_tokens(compressed)
        logger.info(
            f"Compressed results for {agent_key}: {current_tokens} -> {final_tokens} tokens "
            f"({(1 - final_tokens/current_tokens)*100:.1f}% reduction)"
        )

        return compressed

    def _compress_results(
        self,
        results: Dict[str, Any],
        max_tokens: int,
        agent_key: str
    ) -> Dict[str, Any]:
        """
        Intelligently compress results to fit token limit.

        Strategies (in order):
        1. Remove verbose descriptions and metadata
        2. Truncate lists to top-confidence items
        3. Create hierarchical summary
        4. Apply aggressive truncation

        Args:
            results: Results to compress
            max_tokens: Target token limit
            agent_key: Agent key for context-aware compression

        Returns:
            Compressed results
        """
        compressed = results.copy()

        # Strategy 1: Remove verbose fields
        compressed = self._remove_verbose_fields(compressed)
        if self.estimate_tokens(compressed) <= max_tokens:
            return compressed

        # Strategy 2: Truncate lists intelligently
        compressed = self._truncate_lists(compressed, max_tokens)
        if self.estimate_tokens(compressed) <= max_tokens:
            return compressed

        # Strategy 3: Create hierarchical summary
        compressed = self._create_summary(compressed, max_tokens, agent_key)
        if self.estimate_tokens(compressed) <= max_tokens:
            return compressed

        # Strategy 4: Aggressive truncation (keep only essentials)
        compressed = self._aggressive_truncation(compressed, max_tokens)

        return compressed

    def _remove_verbose_fields(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove verbose fields that don't affect downstream processing."""
        verbose_fields = [
            "raw_results",
            "errors",  # Keep critical info, remove error details
            "timing",
            "metadata",
            "_internal",
            "debug_info",
            "provenance_details",  # Keep summary, remove details
        ]

        cleaned = {}
        for key, value in results.items():
            if key in verbose_fields:
                # Skip verbose fields, but keep error count
                if key == "errors" and isinstance(value, list):
                    cleaned["error_count"] = len(value)
                continue

            # Recursively clean nested dicts
            if isinstance(value, dict):
                cleaned[key] = self._remove_verbose_fields(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                cleaned[key] = [self._remove_verbose_fields(item) for item in value]
            else:
                cleaned[key] = value

        return cleaned

    def _truncate_lists(self, results: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        Intelligently truncate lists to fit token limit.

        Prioritizes:
        - High confidence items
        - Unique entities
        - Recently processed items
        """
        truncated = {}
        current_tokens = 0

        # Priority order for keys (most important first)
        priority_keys = self._get_priority_keys(results)

        for key in priority_keys:
            if key not in results:
                continue

            value = results[key]

            if isinstance(value, list):
                # Truncate list based on importance
                truncated_list = []
                for item in value:
                    item_tokens = self.estimate_tokens(item)

                    if current_tokens + item_tokens <= max_tokens:
                        truncated_list.append(item)
                        current_tokens += item_tokens
                    else:
                        # Add metadata about truncation
                        logger.debug(f"Truncated '{key}' list: {len(truncated_list)}/{len(value)} items kept")
                        truncated[f"{key}_truncated"] = True
                        truncated[f"{key}_original_count"] = len(value)
                        break

                if truncated_list:
                    truncated[key] = truncated_list
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                value_tokens = self.estimate_tokens(value)
                if current_tokens + value_tokens <= max_tokens:
                    truncated[key] = value
                    current_tokens += value_tokens
                else:
                    # Try to compress nested dict
                    compressed_value = self._truncate_lists(value, max_tokens - current_tokens)
                    truncated[key] = compressed_value
                    current_tokens += self.estimate_tokens(compressed_value)
            else:
                # Simple value
                value_tokens = self.estimate_tokens(str(value))
                if current_tokens + value_tokens <= max_tokens:
                    truncated[key] = value
                    current_tokens += value_tokens

            # Stop if we're at the limit
            if current_tokens >= max_tokens * 0.95:  # 95% threshold
                break

        return truncated

    def _get_priority_keys(self, results: Dict[str, Any]) -> List[str]:
        """
        Get priority-ordered list of keys for result processing.

        Args:
            results: Results dictionary

        Returns:
            List of keys in priority order
        """
        # High priority: core extraction results
        high_priority = ["entities", "resources", "key_terms", "aligned_resources", "judge_resource"]

        # Medium priority: metadata and scores
        medium_priority = ["confidence", "scores", "alignments", "modifications"]

        # Low priority: auxiliary info
        low_priority = ["provenance", "chunk_info", "statistics"]

        # Build final list
        priority_list = []

        # Add high priority keys that exist
        for key in high_priority:
            if key in results:
                priority_list.append(key)

        # Add medium priority
        for key in medium_priority:
            if key in results:
                priority_list.append(key)

        # Add low priority
        for key in low_priority:
            if key in results:
                priority_list.append(key)

        # Add remaining keys
        for key in results.keys():
            if key not in priority_list:
                priority_list.append(key)

        return priority_list

    def _create_summary(
        self,
        results: Dict[str, Any],
        max_tokens: int,
        agent_key: str
    ) -> Dict[str, Any]:
        """
        Create a hierarchical summary of results.

        Args:
            results: Results to summarize
            max_tokens: Target token limit
            agent_key: Agent key for context

        Returns:
            Summarized results
        """
        summary = {
            "_summary_mode": True,
            "_original_token_count": self.estimate_tokens(results)
        }

        # Create counts and statistics
        for key, value in results.items():
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
                # Keep a small sample
                sample_size = min(5, len(value))
                summary[f"{key}_sample"] = value[:sample_size]
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())
            else:
                # Keep simple values
                summary[key] = value

        return summary

    def _aggressive_truncation(
        self,
        results: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Last resort: aggressively truncate to fit token limit.

        Keeps only the most essential information.

        Args:
            results: Results to truncate
            max_tokens: Target token limit

        Returns:
            Aggressively truncated results
        """
        essential = {}
        current_tokens = 0

        # Only keep absolutely essential keys
        essential_keys = ["entities", "resources", "key_terms", "aligned_resources", "judge_resource"]

        for key in essential_keys:
            if key not in results:
                continue

            value = results[key]
            value_str = json.dumps({key: value}, default=str)
            value_tokens = self.count_tokens(value_str)

            if current_tokens + value_tokens <= max_tokens:
                essential[key] = value
                current_tokens += value_tokens
            elif isinstance(value, list) and value:
                # Keep at least one item
                essential[key] = [value[0]]
                essential[f"{key}_truncated_to_1"] = True
                essential[f"{key}_original_count"] = len(value)
                break

        essential["_aggressively_truncated"] = True
        essential["_warning"] = "Results heavily truncated due to token limits"

        return essential

    def split_for_parallel_processing(
        self,
        text: str,
        max_chunk_tokens: int,
        overlap_tokens: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Split text into token-aware chunks for parallel processing.

        Args:
            text: Input text
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks

        Returns:
            List of chunk dictionaries with text, start, end, token_count
        """
        if self.encoding is None:
            # Fallback to character-based splitting
            return self._char_based_split(text, max_chunk_tokens * 4, overlap_tokens * 4)

        # Tokenize full text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= max_chunk_tokens:
            return [{
                "text": text,
                "start": 0,
                "end": len(text),
                "token_count": total_tokens,
                "chunk_id": 0
            }]

        chunks = []
        chunk_id = 0
        start_token = 0

        while start_token < total_tokens:
            end_token = min(start_token + max_chunk_tokens, total_tokens)

            # Decode chunk
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "start": start_token,
                "end": end_token,
                "token_count": len(chunk_tokens),
                "chunk_id": chunk_id
            })

            chunk_id += 1
            start_token = end_token - overlap_tokens  # Overlap for context continuity

        logger.info(f"Split text into {len(chunks)} chunks (max {max_chunk_tokens} tokens/chunk)")
        return chunks

    def _char_based_split(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int
    ) -> List[Dict[str, Any]]:
        """Fallback character-based splitting when tokenizer unavailable."""
        chunks = []
        chunk_id = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_chars, text_len)
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end,
                "token_count": self.count_tokens(chunk_text),
                "chunk_id": chunk_id
            })

            chunk_id += 1
            start = end - overlap_chars

        return chunks
