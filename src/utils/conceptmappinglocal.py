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
# @File    : conceptmappinglocal.py
# @Software: PyCharm

"""
CrewAI Tool for Ontology Concept Mapping using a local HTTP service.

Targets the ``POST /map/batch`` endpoint of the local Ontology Database
Concept Mapping service (hybrid BM25 + dense retrieval with re-ranking).

All calls — single term or multi-term — are sent as a single batch request
so the service can run them concurrently, which is much faster than one
request per term.

Response fields
---------------
The local service returns ``ResultItem`` objects with:
  - ``ontology_id``     → canonical IRI for the concept
  - ``ontology_label``  → human-readable label
  - ``ontology``        → ontology acronym (e.g. SNOMEDCT, MONDO)
  - ``final_score``     → composite re-ranking score

These are mapped to the canonical output format used by ``ConceptMappingTool``
so downstream pipeline stages need no changes:
  ``{"class_uri": <IRI>, "ontology_label": <label>, "ontology_id": <acronym>}``

Environment variables
---------------------
LOCAL_CONCEPT_MAPPING_URL
    Base URL of the local mapping service.
    Default: ``http://localhost:8000``

LOCAL_CONCEPT_MAPPING_API_KEY
    Optional OpenRouter API key forwarded to the service as
    ``openrouter_api_key`` for LLM-based re-ranking modes.
    Falls back to ``OPENROUTER_API_KEY`` if unset.
    Leave empty when using the default ``dual_late`` re-ranker (no LLM needed).

LOCAL_CONCEPT_MAPPING_MODEL
    Optional OpenRouter model name forwarded as ``openrouter_model``.
    Falls back to ``OPENROUTER_MODEL`` if unset.

LOCAL_CONCEPT_MAPPING_TIMEOUT
    Request timeout in seconds.  Default: ``30``

MAX_CONCEPT_MAPPING_RESULTS
    Max results per term (1-20).  Shared with the BioPortal tool.
    Default: ``1``
"""

import os
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Type

import requests
from pydantic import BaseModel
from crewai.tools import BaseTool

from .types import ConceptMappingInput
from .conceptmappingtool import (
    _ALIGNMENT_TOOL_OUTPUTS,
    _ALIGNMENT_TOOL_OUTPUTS_LOCK,
    _CONCEPT_MAPPING_CACHE,
    _CONCEPT_MAPPING_CACHE_LOCK,
    _concept_mapping_cache_max_size,
    _normalize_max_results,
    _sanitize_text,
    _sanitize_ontologies_raw,
)

logger = logging.getLogger("ConceptMappingLocalTool")


# ---------------------------------------------------------------------------
# Env-var helpers (read at call time so .env load order is respected)
# ---------------------------------------------------------------------------


def _local_base_url() -> str:
    return os.getenv("LOCAL_CONCEPT_MAPPING_URL", "http://localhost:8000").rstrip("/")


def _local_api_key() -> Optional[str]:
    v = os.getenv("LOCAL_CONCEPT_MAPPING_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    return v if v else None


def _local_model() -> Optional[str]:
    v = os.getenv("LOCAL_CONCEPT_MAPPING_MODEL", "").strip() or os.getenv("OPENROUTER_MODEL", "").strip()
    return v if v else None


def _local_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("LOCAL_CONCEPT_MAPPING_TIMEOUT", "30")))
    except (TypeError, ValueError):
        return 30.0


# ---------------------------------------------------------------------------
# ConceptMappingLocalTool
# ---------------------------------------------------------------------------


class ConceptMappingLocalTool(BaseTool):
    """
    CrewAI Tool for mapping concepts to ontology IRIs and labels via a
    local Ontology Concept Mapping service.

    Always uses ``POST /map/batch`` for efficiency — single terms are sent
    as a one-element batch so the service can parallelize internally.

    Output format is identical to ``ConceptMappingTool`` (BioPortal) so the
    rest of the pipeline needs no changes.

    Single concept result (max_results=1):
    ```json
    {"ontology_id": "http://purl.obolibrary.org/obo/MONDO_0005015",
     "ontology_label": "diabetes mellitus",
     "ontology": "MONDO"}
    ```

    Multiple concepts (comma-separated):
    ```json
    {
      "diabetes": {"ontology_id": "...", "ontology_label": "...", "ontology": "MONDO"},
      "cancer":   {"ontology_id": "...", "ontology_label": "...", "ontology": "NCIT"}
    }
    ```

    Multiple results per concept (max_results > 1):
    ```json
    {"ontology_id": ["http://...", "http://..."],
     "ontology_label": ["diabetes mellitus", "Diabetes"],
     "ontology": ["MONDO", "SNOMEDCT"]}
    ```
    """

    name: str = "Concept Mapping Tool"
    description: str = (
        "Maps biomedical/neuroscientific concepts to ontology identifiers (IRIs) and labels "
        "using a local service with hybrid BM25 + dense retrieval.\n\n"
        "IMPORTANT — call this tool ONCE with ALL terms, not once per term.\n\n"
        "PREFERRED — pass a list of dicts, one per entity, with its source sentence as context:\n"
        '  [{"text": "hippocampus", "context": "Neurons in CA1 were recorded."},\n'
        '   {"text": "cortex", "context": "Prefrontal cortex activity was measured."}]\n\n'
        "Also accepted:\n"
        "- List of strings: ['hippocampus', 'cortex', 'amygdala']\n"
        "- Single string or phrase: 'hippocampus'\n\n"
        "Returns ontology_id, ontology_label, and ontology acronym for each term."
    )
    args_schema: Type[BaseModel] = ConceptMappingInput

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_session", requests.Session())
        logger.info(
            "ConceptMappingLocalTool initialized (LOCAL_CONCEPT_MAPPING_URL=%s)",
            _local_base_url(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_batch(
        self,
        term_objects: list,
        max_results: int,
    ) -> Optional[dict]:
        """
        POST to ``/map/batch``.

        ``term_objects`` is a list of ``{"text": str, "context": str|None}`` dicts.
        Always sends dict format so the service keys results by the ``text`` field.
        The local service handles ontology selection internally; no ``ontologies``
        field is sent (it causes HTTP 400 on the batch endpoint).
        """
        url = f"{_local_base_url()}/map/batch"

        # Always send dict format {"text": ..., "context": ...} so the service
        # consistently keys results by the "text" field value.
        #
        # Context truncation: the local service may validate context length (e.g.
        # Field(max_length=200)).  Chunk-level contexts generated by the pipeline can
        # be 300-400+ characters, which triggers HTTP 400 on the batch endpoint.
        # Truncate to 200 chars so we always stay within server limits while still
        # providing enough context for re-ranking.
        _MAX_CONTEXT_LEN = 200
        text_payload = []
        for obj in term_objects:
            item: dict = {"text": obj["text"]}
            ctx = obj.get("context") or ""
            if ctx:
                item["context"] = ctx[:_MAX_CONTEXT_LEN]
            text_payload.append(item)

        # The local service handles ontology selection internally — do not send
        # the ontologies field, which causes HTTP 400 on the batch endpoint.
        payload: dict = {
            "text": text_payload,
            "max_results": max_results,
        }

        api_key = _local_api_key()
        if api_key:
            payload["openrouter_api_key"] = api_key

        model = _local_model()
        if model:
            payload["openrouter_model"] = model

        logger.info(
            "Sending batch to %s | %d term(s) | max_results=%d | payload text=%s",
            url,
            len(term_objects),
            max_results,
            text_payload,
        )

        try:
            resp = self._session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=_local_timeout(),
            )
        except Exception as exc:
            logger.error("Local concept mapping request to %s failed: %s", url, exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as exc:
                logger.error("Failed to parse JSON from %s: %s", url, exc)
                return None

        # Log the full response body so the exact server-side validation error is visible.
        # Without this, all we see is the HTTP status code and the 400/422 root cause
        # (e.g. context string too long, empty text after sanitization, batch size limit)
        # is invisible in the logs.
        try:
            body = resp.text[:500]  # truncate so we don't flood the log
        except Exception:
            body = "<unreadable>"
        logger.warning(
            "Local concept mapping service returned HTTP %s for %s | response: %s",
            resp.status_code,
            url,
            body,
        )
        return None

    @staticmethod
    def _result_items_to_mapping(items: list, max_results: int) -> dict:
        """
        Convert a list of ResultItem dicts (from the service response) into the
        canonical output dict.

        Service response fields (updated batch format):
          - ``ontology_id``    → class_uri  (IRI, e.g. "http://purl.obolibrary.org/obo/HP_0012622")
          - ``ontology_label`` → ontology_label
          - ``ontology``       → ontology_id (acronym, e.g. "HP", "MONDO")
        """
        valid = [it for it in items if isinstance(it, dict) and it.get("ontology_id") and it.get("ontology_label")][:max_results]

        if not valid:
            return {"error": "No mapping found", "ontology_id": None, "ontology_label": None, "ontology": None}

        if max_results == 1 or len(valid) == 1:
            item = valid[0]
            return {
                "ontology_id": item["ontology_id"],
                "ontology_label": item["ontology_label"],
                "ontology": item.get("ontology", ""),
            }

        return {
            "ontology_id": [it["ontology_id"] for it in valid],
            "ontology_label": [it["ontology_label"] for it in valid],
            "ontology": [it.get("ontology", "") for it in valid],
        }

    def _map_terms(
        self,
        term_objects: list,
        max_results: int,
        ontologies: Optional[str] = None,
    ) -> dict:
        """
        Map a list of ``{"text": str, "context": str|None}`` objects via /map/batch.

        Large batches are split into parallel sub-batches (env LOCAL_CONCEPT_MAPPING_BATCH_SIZE,
        default 4000) and sent concurrently (env LOCAL_CONCEPT_MAPPING_WORKERS, default 4).
        Cache is keyed by term text only.  Returns ``{term_text: mapping_dict, ...}``.
        """
        results: dict = {}
        uncached: list = []

        # Check cache for all terms at once under a single lock acquisition
        with _CONCEPT_MAPPING_CACHE_LOCK:
            for obj in term_objects:
                term = obj["text"]
                cache_key = f"local|{term}|{max_results}"
                if cache_key in _CONCEPT_MAPPING_CACHE:
                    results[term] = dict(_CONCEPT_MAPPING_CACHE[cache_key])
                else:
                    uncached.append(obj)

        if not uncached:
            logger.info(
                "ConceptMappingLocalTool: all %d term(s) served from in-memory cache — no HTTP request sent",
                len(results),
            )
            return results

        # Deduplicate uncached by text (same term may appear twice from different entities)
        seen_texts: set = set()
        deduped_uncached: list = []
        text_to_objs: dict = {}  # text → list of all objs sharing that text (to map result back)
        for obj in uncached:
            t = obj["text"]
            if t not in seen_texts:
                seen_texts.add(t)
                deduped_uncached.append(obj)
            text_to_objs.setdefault(t, []).append(obj)
        uncached = deduped_uncached

        logger.info(
            "ConceptMappingLocalTool: %d term(s) cached, %d term(s) need HTTP request",
            len(results),
            len(uncached),
        )

        # Determine sub-batch size and worker count from env
        # Default is 4000 — the service supports up to 4000 concepts per request.
        try:
            batch_size = max(1, int(os.getenv("LOCAL_CONCEPT_MAPPING_BATCH_SIZE", "4000")))
        except (TypeError, ValueError):
            batch_size = 4000
        try:
            max_workers = max(1, int(os.getenv("LOCAL_CONCEPT_MAPPING_WORKERS", "4")))
        except (TypeError, ValueError):
            max_workers = 4

        # Split uncached terms into sub-batches
        n_batches = math.ceil(len(uncached) / batch_size)
        sub_batches = [uncached[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]

        def _fetch_sub_batch(batch: list) -> dict:
            """POST one sub-batch; returns {term: [ResultItem, ...]}."""
            raw = self._post_batch(batch, max_results)
            if raw and isinstance(raw.get("results"), dict):
                return raw["results"]
            return {}

        # Send all sub-batches in parallel
        per_term: dict = {}
        if len(sub_batches) == 1:
            per_term = _fetch_sub_batch(sub_batches[0])
        else:
            logger.info(
                "ConceptMappingLocalTool: %d terms → %d parallel sub-batches (batch_size=%d, workers=%d)",
                len(uncached),
                len(sub_batches),
                batch_size,
                max_workers,
            )
            with ThreadPoolExecutor(max_workers=min(max_workers, len(sub_batches))) as executor:
                futures = {executor.submit(_fetch_sub_batch, sb): sb for sb in sub_batches}
                for future in as_completed(futures):
                    try:
                        per_term.update(future.result())
                    except Exception as exc:
                        logger.warning("Sub-batch request failed: %s", exc)

        # Store results and populate cache
        new_entries: dict = {}
        for obj in uncached:
            term = obj["text"]
            items = per_term.get(term)
            if isinstance(items, list):
                mapping = self._result_items_to_mapping(items, max_results)
            else:
                mapping = {
                    "error": f"No mapping returned for: {term}",
                    "ontology_id": None,
                    "ontology_label": None,
                    "ontology": None,
                }
            new_entries[f"local|{term}|{max_results}"] = mapping
            results[term] = mapping

        with _CONCEPT_MAPPING_CACHE_LOCK:
            max_size = _concept_mapping_cache_max_size()
            for cache_key, mapping in new_entries.items():
                if len(_CONCEPT_MAPPING_CACHE) >= max_size:
                    _CONCEPT_MAPPING_CACHE.pop(next(iter(_CONCEPT_MAPPING_CACHE)), None)
                _CONCEPT_MAPPING_CACHE[cache_key] = mapping

        return results

    # ------------------------------------------------------------------
    # CrewAI _run entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_input(text, shared_context: Optional[str]) -> list:
        """
        Parse the ``text`` field into a list of ``{"text": str, "context": str|None}``
        objects ready for ``_map_terms``.

        Accepted formats:

        1. PREFERRED — Python list of dicts with per-term context:
           ``[{"text": "diabetes", "context": "Type 2 with complications"}, ...]``

        2. Python list of strings:
           ``["hippocampus", "cortex", "amygdala"]``

        3. JSON array string (fallback when agent serialises the list to a string):
           ``'[{"text": "diabetes", "context": "..."}, ...]'``

        4. Single string concept or phrase:
           ``"hippocampus"``
        """
        import json as _json

        # Format 1 & 2: already a Python list — no string parsing needed
        if isinstance(text, list):
            result = []
            for item in text:
                if isinstance(item, dict) and item.get("text"):
                    result.append(
                        {
                            "text": _sanitize_text(item["text"]),
                            "context": (item.get("context") or "").strip() or None,
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    result.append({"text": _sanitize_text(item), "context": shared_context})
            return result

        # Remaining formats: text is a string
        stripped = text.strip() if text else ""

        # Format 3: JSON array string
        if stripped.startswith("["):
            try:
                parsed = _json.loads(stripped)
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, dict) and item.get("text"):
                            result.append(
                                {
                                    "text": _sanitize_text(item["text"]),
                                    "context": (item.get("context") or "").strip() or None,
                                }
                            )
                        elif isinstance(item, str) and item.strip():
                            result.append({"text": _sanitize_text(item), "context": shared_context})
                    if result:
                        return result
            except Exception:
                pass  # fall through to single-string handling

        # Format 4: single concept or phrase
        cleaned = _sanitize_text(stripped)
        return [{"text": cleaned, "context": shared_context}] if cleaned else []

    def _run(
        self,
        text,
        max_results: Optional[int] = None,
        ontologies: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict:
        max_results = _normalize_max_results(max_results)
        ontologies = _sanitize_ontologies_raw(ontologies)
        shared_context = (context or "").strip() or None

        if not text and text != 0:
            return {
                "error": "No valid text provided for concept mapping",
                "ontology_id": None,
                "ontology_label": None,
                "ontology": None,
            }

        # When text is a list pass it directly; when a string sanitize unless it
        # looks like a JSON array (sanitize would mangle brackets/quotes).
        if isinstance(text, list):
            raw = text
        else:
            stripped = text.strip() if text else ""
            if not stripped.startswith("["):
                stripped = _sanitize_text(stripped)
            if not stripped:
                return {
                    "error": "No valid text provided for concept mapping",
                    "ontology_id": None,
                    "ontology_label": None,
                    "ontology": None,
                }
            raw = stripped

        term_objects = self._parse_input(raw, shared_context)
        is_batch = len(term_objects) > 1

        logger.info(
            "ConceptMappingLocalTool: %s input, %d term(s), max_results=%d | terms=%s",
            "batch" if is_batch else "single",
            len(term_objects),
            max_results,
            [
                (o["text"], o["context"][:40] + "..." if o.get("context") and len(o["context"]) > 40 else o.get("context"))
                for o in term_objects
            ],
        )

        mapped = self._map_terms(term_objects, max_results, ontologies)

        if is_batch:
            out = mapped
            with _ALIGNMENT_TOOL_OUTPUTS_LOCK:
                for term, mapping in out.items():
                    if isinstance(mapping, dict) and "error" not in mapping:
                        _ALIGNMENT_TOOL_OUTPUTS.append({"input": term, "output": dict(mapping)})
        else:
            first_term = term_objects[0]["text"]
            out = mapped.get(
                first_term,
                {
                    "error": "No mapping returned",
                    "ontology_id": None,
                    "ontology_label": None,
                    "ontology": None,
                },
            )
            if isinstance(out, dict) and "error" not in out:
                with _ALIGNMENT_TOOL_OUTPUTS_LOCK:
                    _ALIGNMENT_TOOL_OUTPUTS.append({"input": first_term, "output": dict(out)})

        return out
