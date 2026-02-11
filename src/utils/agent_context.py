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
# @File    : agent_context.py
# @Software: PyCharm

"""
Agent Context Management for Multi-Agent Pipelines.

Provides context passing, token management, and iterative refinement
capabilities for sequential agent processing.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from threading import Lock
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Stores result from a single agent execution."""
    agent_key: str
    task_key: str
    chunk_id: Optional[int]
    result: Dict[str, Any]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


class AgentContext:
    """
    Manages context and communication between agents in a multi-agent pipeline.

    Features:
    - Context accumulation across agent chains
    - Confidence tracking
    - Provenance and metadata management
    - Token-aware context summarization
    """

    def __init__(self, max_tokens: int = 90000, encoding_model: str = "cl100k_base"):
        """
        Initialize agent context.

        Args:
            max_tokens: Maximum tokens for context window
            encoding_model: Tokenizer model name (cl100k_base for GPT-4)
        """
        self.agent_results: Dict[str, List[AgentResult]] = {}
        self.confidence_scores: Dict[str, float] = {}
        self.entity_mappings: Dict[str, Any] = {}
        self.chunk_relationships: Dict[int, Set[int]] = {}
        self.metadata: Dict[str, Any] = {}
        self.max_tokens = max_tokens

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {encoding_model}: {e}. Using character approximation.")
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate 1 token = 4 characters
            return len(text) // 4

    def add_agent_result(
        self,
        agent_key: str,
        task_key: str,
        result: Dict[str, Any],
        chunk_id: Optional[int] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add result from an agent execution.

        Args:
            agent_key: Key identifying the agent
            task_key: Key identifying the task
            result: Result dictionary from agent
            chunk_id: Optional chunk identifier
            confidence: Confidence score (0.0-1.0)
            metadata: Optional metadata dictionary
        """
        if agent_key not in self.agent_results:
            self.agent_results[agent_key] = []

        # Calculate token count
        result_str = json.dumps(result, default=str)
        token_count = self.count_tokens(result_str)

        agent_result = AgentResult(
            agent_key=agent_key,
            task_key=task_key,
            chunk_id=chunk_id,
            result=result,
            confidence=confidence,
            metadata=metadata or {},
            token_count=token_count
        )

        self.agent_results[agent_key].append(agent_result)
        self.confidence_scores[agent_key] = confidence

        logger.debug(f"Added result for {agent_key} (chunk {chunk_id}): {token_count} tokens")

    def get_agent_results(self, agent_key: str) -> List[AgentResult]:
        """Get all results for a specific agent."""
        return self.agent_results.get(agent_key, [])

    def get_latest_result(self, agent_key: str) -> Optional[AgentResult]:
        """Get the most recent result for an agent."""
        results = self.agent_results.get(agent_key, [])
        return results[-1] if results else None

    def get_merged_results(self, agent_key: str) -> Dict[str, Any]:
        """
        Get merged results for an agent across all chunks.

        Returns a single dictionary combining all chunk results.
        """
        results = self.agent_results.get(agent_key, [])
        if not results:
            return {}

        # Merge logic depends on result structure
        merged = {}
        for agent_result in results:
            result = agent_result.result
            for key, value in result.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, list) and isinstance(merged[key], list):
                    merged[key].extend(value)
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key].update(value)
                else:
                    # Use the latest value for simple types
                    merged[key] = value

        return merged

    def get_summary(
        self,
        agent_key: str,
        max_tokens: Optional[int] = None,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Get a token-aware summary of agent results.

        Args:
            agent_key: Agent to get summary for
            max_tokens: Maximum tokens for summary (uses self.max_tokens if None)
            include_confidence: Whether to include confidence scores

        Returns:
            Summarized results dictionary
        """
        max_tokens = max_tokens or self.max_tokens
        results = self.agent_results.get(agent_key, [])

        if not results:
            return {}

        # Start with merged results
        summary = self.get_merged_results(agent_key)

        if include_confidence:
            summary["_confidence"] = self.confidence_scores.get(agent_key, 0.0)

        # Check token count
        summary_str = json.dumps(summary, default=str)
        current_tokens = self.count_tokens(summary_str)

        if current_tokens <= max_tokens:
            return summary

        # If too large, intelligently truncate
        logger.warning(f"Summary for {agent_key} exceeds {max_tokens} tokens ({current_tokens}). Truncating...")
        return self._truncate_summary(summary, max_tokens)

    def _truncate_summary(self, summary: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        Intelligently truncate summary to fit token limit.

        Strategy:
        1. Keep high-confidence items
        2. Prioritize unique/important information
        3. Remove verbose descriptions
        """
        truncated = {}
        current_tokens = 0

        # Priority order: confidence, metadata, then content
        priority_keys = ["_confidence", "entities", "resources", "key_terms", "aligned_resources", "judge_resource"]

        for key in priority_keys:
            if key not in summary:
                continue

            value = summary[key]
            value_str = json.dumps({key: value}, default=str)
            value_tokens = self.count_tokens(value_str)

            if current_tokens + value_tokens <= max_tokens:
                truncated[key] = value
                current_tokens += value_tokens
            elif isinstance(value, list):
                # Truncate list to fit
                items_to_include = []
                for item in value:
                    item_str = json.dumps(item, default=str)
                    item_tokens = self.count_tokens(item_str)
                    if current_tokens + item_tokens <= max_tokens:
                        items_to_include.append(item)
                        current_tokens += item_tokens
                    else:
                        break
                if items_to_include:
                    truncated[key] = items_to_include
                    truncated[f"{key}_truncated"] = True
                    truncated[f"{key}_original_count"] = len(value)
                break  # Stop processing more keys if we're truncating lists
            else:
                # Can't include this key without exceeding limit
                logger.warning(f"Cannot include key '{key}' in summary (would exceed token limit)")
                break

        return truncated

    def get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for all agents."""
        return self.confidence_scores.copy()

    def get_entity_mappings(self) -> Dict[str, Any]:
        """Get entity mappings across chunks."""
        return self.entity_mappings.copy()

    def add_entity_mapping(self, entity_id: str, mapping: Any):
        """Add an entity mapping."""
        self.entity_mappings[entity_id] = mapping

    def add_chunk_relationship(self, chunk_id: int, related_chunk_id: int):
        """Record a relationship between chunks."""
        if chunk_id not in self.chunk_relationships:
            self.chunk_relationships[chunk_id] = set()
        self.chunk_relationships[chunk_id].add(related_chunk_id)

    def get_related_chunks(self, chunk_id: int) -> Set[int]:
        """Get chunks related to the given chunk."""
        return self.chunk_relationships.get(chunk_id, set())

    def get_total_tokens(self) -> int:
        """Calculate total tokens used across all agent results."""
        total = 0
        for results in self.agent_results.values():
            for result in results:
                total += result.token_count
        return total

    def clear_agent_results(self, agent_key: str):
        """Clear results for a specific agent (useful for refinement loops)."""
        if agent_key in self.agent_results:
            del self.agent_results[agent_key]
        if agent_key in self.confidence_scores:
            del self.confidence_scores[agent_key]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "agent_results": {
                key: [
                    {
                        "agent_key": r.agent_key,
                        "task_key": r.task_key,
                        "chunk_id": r.chunk_id,
                        "result": r.result,
                        "confidence": r.confidence,
                        "metadata": r.metadata,
                        "token_count": r.token_count
                    }
                    for r in results
                ]
                for key, results in self.agent_results.items()
            },
            "confidence_scores": self.confidence_scores,
            "entity_mappings": self.entity_mappings,
            "metadata": self.metadata,
            "total_tokens": self.get_total_tokens()
        }


class ThreadSafeMemory:
    """
    Thread-safe in-memory storage for agent communication in parallel execution.

    Replaces SQLite-based memory to avoid database locking in parallel mode.
    """

    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = Lock()
        self.access_count: Dict[str, int] = {}

    def write(self, key: str, value: Any) -> bool:
        """
        Write a value to memory (thread-safe).

        Args:
            key: Storage key
            value: Value to store

        Returns:
            True if successful
        """
        try:
            with self.lock:
                self.memory[key] = value
                self.access_count[key] = self.access_count.get(key, 0) + 1
            return True
        except Exception as e:
            logger.error(f"Failed to write to memory: {e}")
            return False

    def read(self, key: str, default: Any = None) -> Any:
        """
        Read a value from memory (thread-safe).

        Args:
            key: Storage key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        with self.lock:
            return self.memory.get(key, default)

    def read_all(self) -> Dict[str, Any]:
        """Get all memory contents (thread-safe)."""
        with self.lock:
            return self.memory.copy()

    def update(self, key: str, update_fn) -> Any:
        """
        Update a value atomically using a function.

        Args:
            key: Storage key
            update_fn: Function that takes current value and returns new value

        Returns:
            Updated value
        """
        with self.lock:
            current = self.memory.get(key)
            updated = update_fn(current)
            self.memory[key] = updated
            return updated

    def delete(self, key: str) -> bool:
        """Delete a key from memory."""
        try:
            with self.lock:
                if key in self.memory:
                    del self.memory[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete from memory: {e}")
            return False

    def clear(self):
        """Clear all memory."""
        with self.lock:
            self.memory.clear()
            self.access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            return {
                "total_keys": len(self.memory),
                "access_counts": self.access_count.copy(),
                "most_accessed": max(self.access_count.items(), key=lambda x: x[1]) if self.access_count else None
            }
