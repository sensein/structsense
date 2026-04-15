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
# @File    : conceptmappingtool.py
# @Software: PyCharm

"""
CrewAI Tool for Ontology Concept Mapping using BioPortal API
Automatically handles single or batch mapping based on input
"""

import os
import re
import time
import threading
import requests
import logging
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from .types import ConceptMappingInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConceptMappingTool")

# Max length for a single concept/sentence sent to the API (avoid oversized or malformed queries)
MAX_QUERY_LENGTH = 500

# Throttle BioPortal API to avoid 429 Rate Limit (min seconds between requests)
_bioportal_throttle_lock = threading.Lock()
_bioportal_last_request_time = 0.0


def _bioportal_interval() -> float:
    """Read at request time so .env is respected. Lower = faster, but may hit 429."""
    try:
        return max(0.2, float(os.getenv("BIOPORTAL_REQUEST_INTERVAL", "0.7")))
    except (TypeError, ValueError):
        return 0.7


def _bioportal_backoff() -> float:
    try:
        return max(0.5, float(os.getenv("BIOPORTAL_BACKOFF_AFTER_429", "2.0")))
    except (TypeError, ValueError):
        return 2.0


# In-memory cache: term -> mapping result. Speeds up large runs when same term appears often or across runs.
_CONCEPT_MAPPING_CACHE: dict = {}
_CONCEPT_MAPPING_CACHE_LOCK = threading.Lock()

# Session capture: when alignment (or other) agent uses the tool, we record input/output here
# so the pipeline can preserve concept_mapping in the final result without re-running at end.
_ALIGNMENT_TOOL_OUTPUTS: list = []
_ALIGNMENT_TOOL_OUTPUTS_LOCK = threading.Lock()


def clear_alignment_tool_outputs() -> None:
    """Clear recorded tool outputs. Call before running the alignment stage so we only capture alignment's tool calls."""
    with _ALIGNMENT_TOOL_OUTPUTS_LOCK:
        _ALIGNMENT_TOOL_OUTPUTS.clear()


def get_alignment_tool_outputs() -> list:
    """Return a copy of recorded tool outputs (input + output) from the current alignment stage."""
    with _ALIGNMENT_TOOL_OUTPUTS_LOCK:
        return list(_ALIGNMENT_TOOL_OUTPUTS)


def format_alignment_tool_outputs_as_concept_mapping(session_outputs: list) -> list:
    """
    Convert session_outputs from get_alignment_tool_outputs() into the concept_mapping list format:
    [{"term": str, "source": "alignment_agent_tool", "ontology_id", "ontology_label", "ontology", "concept_mapping_provenance": "tool"}, ...].
    Uses top-1 when ontology_id/ontology_label/ontology are lists.
    """
    result = []
    for item in session_outputs:
        if not isinstance(item, dict):
            continue
        inp = item.get("input") or ""
        out = item.get("output")
        if not isinstance(out, dict) or "error" in out:
            continue

        def _top1(v):
            if isinstance(v, list) and v:
                return v[0]
            return v

        # Batch: out = {term: {ontology_id, ontology_label, ontology}, ...}
        if any(k not in ("ontology_id", "ontology_label", "ontology", "error") for k in out):
            for term, mapping in out.items():
                if not isinstance(mapping, dict) or mapping.get("error"):
                    continue
                result.append(
                    {
                        "term": term,
                        "source": "alignment_agent_tool",
                        "ontology_id": _top1(mapping.get("ontology_id")),
                        "ontology_label": _top1(mapping.get("ontology_label")),
                        "ontology": _top1(mapping.get("ontology")),
                        "concept_mapping_provenance": "tool",
                    }
                )
        else:
            # Single concept: out = {ontology_id, ontology_label, ontology}
            result.append(
                {
                    "term": inp,
                    "source": "alignment_agent_tool",
                    "ontology_id": _top1(out.get("ontology_id")),
                    "ontology_label": _top1(out.get("ontology_label")),
                    "ontology": _top1(out.get("ontology")),
                    "concept_mapping_provenance": "tool",
                }
            )
    return result


def _concept_mapping_cache_max_size() -> int:
    """Max cache entries (read at use time so .env is respected)."""
    try:
        return max(100, int(os.getenv("CONCEPT_MAPPING_CACHE_SIZE", "2000")))
    except (TypeError, ValueError):
        return 2000


def _sanitize_text(text: Optional[str]) -> str:
    """
    Robust sanitization for tool input: coerce to str, strip, normalize whitespace,
    remove control characters, and truncate to a safe length.
    """
    if text is None:
        return ""
    s = str(text)
    # Strip leading/trailing whitespace and normalize line endings
    s = s.strip().replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # Collapse multiple spaces/tabs into one
    s = re.sub(r"[ \t]+", " ", s)
    # Remove control characters (C0/C1 and other non-printable)
    s = "".join(c for c in s if c.isprintable() or c.isspace())
    s = s.strip()
    if len(s) > MAX_QUERY_LENGTH:
        s = s[:MAX_QUERY_LENGTH].rstrip()
    return s


def _sanitize_ontologies_raw(ontologies: Optional[str]) -> Optional[str]:
    """Strip and normalize ontologies string; return None if empty."""
    if ontologies is None:
        return None
    s = str(ontologies).strip()
    return s if s else None


def _env_max_results() -> int:
    """Read at runtime so load_dotenv() in app is respected. Default 1 = top-1 alignment result."""
    try:
        n = int(os.getenv("MAX_CONCEPT_MAPPING_RESULTS", "1"))
        return max(1, min(n, 50))
    except (TypeError, ValueError):
        return 1


def _normalize_max_results(value, default: Optional[int] = None) -> int:
    """Ensure max_results is an int in [1, 20]. Handles None, list, or bad types from agent input."""
    if default is None:
        default = _env_max_results()
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        n = int(value)
        return max(1, min(n, 20))
    except (TypeError, ValueError):
        return default


class ConceptMappingTool(BaseTool):
    """
    CrewAI Tool for mapping concepts to ontology IRIs and labels.
    Automatically handles single or batch mapping based on input.
    Uses BioPortal Recommender API for automatic ontology detection.

    Features:
    - Single concept: 'diabetes' → maps one concept
    - Multiple concepts: 'diabetes,cancer,asthma' → batch mapping
    - Sentence: 'diabetic kidney disease' → maps as phrase
    - Auto-detects relevant ontologies

    Sample response from the tool depending on how it's called

    ================================================================================
    Example 1: Single Concept (Top 1 Result - Default)
    ================================================================================
    {
      "ontology_id": "http://www.semanticweb.org/mypc/ontologies/2022/11/USBirthOnto-22#Diabetes",
      "ontology_label": "Diabetes",
      "ontology": "BIRTHONTO"
    }

    ================================================================================
    Example 2: Single Concept (Top 3 Results)
    ================================================================================
    {
      "ontology_id": [
        "http://www.semanticweb.org/danielhier/ontologies/2019/3/untitled-ontology-57#diabetes",
        "http://www.semanticweb.org/mypc/ontologies/2022/11/USBirthOnto-22#Diabetes",
        "http://www.semanticweb.org/diwaleva/ontologies/2019/9/fcc-ontology#Diabetes"
      ],
      "ontology_label": [
        "diabetes",
        "Diabetes",
        "Diabetes"
      ],
      "ontology": [
        "NEO",
        "BIRTHONTO",
        "FCC1"
      ]
    }

    ================================================================================
    Example 3: Batch Concepts (Top 1 Each)
    ================================================================================
    {
      "diabetes": {
        "ontology_id": "http://www.semanticweb.org/mypc/ontologies/2022/11/USBirthOnto-22#Diabetes",
        "ontology_label": "Diabetes",
        "ontology": "BIRTHONTO"
      },
      "cancer": {
        "ontology_id": "http://purl.bioontology.org/ontology/LNC/LA10524-9",
        "ontology_label": "Cancer",
        "ontology": "LOINC"
      },
      "asthma": {
        "ontology_id": "http://purl.bioontology.org/ontology/CST/ASTHMA",
        "ontology_label": "ASTHMA",
        "ontology": "COSTART"
      }
    }

    ================================================================================
    Example 4: Batch Concepts (Top 2 Each)
    ================================================================================
    {
      "diabetes": {
        "ontology_id": [
          "http://www.semanticweb.org/mypc/ontologies/2022/11/USBirthOnto-22#Diabetes",
          "http://www.semanticweb.org/diwaleva/ontologies/2019/9/fcc-ontology#Diabetes"
        ],
        "ontology_label": [
          "Diabetes",
          "Diabetes"
        ],
        "ontology": [
          "BIRTHONTO",
          "FCC1"
        ]
      },
      "cancer": {
        "ontology_id": [
          "http://purl.bioontology.org/ontology/LNC/LA10524-9",
          "http://purl.bioontology.org/ontology/LNC/LP7106-0"
        ],
        "ontology_label": [
          "Cancer",
          "Cancer"
        ],
        "ontology": [
          "LOINC",
          "LOINC"
        ]
      }
    }

    ================================================================================
    Example 5: Sentence/Phrase
    ================================================================================
    {
      "ontology_id": "http://purl.bioontology.org/ontology/MEDDRA/10084917",
      "ontology_label": "Diabetic kidney disease",
      "ontology": "MEDDRA"
    }

    ================================================================================
    Example 6: With Specific Ontologies
    ================================================================================
    {
      "ontology_id": [
        "http://purl.obolibrary.org/obo/MONDO_0007254",
        "http://purl.obolibrary.org/obo/MONDO_0002054"
      ],
      "ontology_label": [
        "breast cancer",
        "obsolete breast cancer"
      ],
      "ontology": [
        "MONDO",
        "MONDO"
      ]
    }

    ================================================================================
    Example 7: Hippocampus (Top 1)
    ================================================================================
    {
      "ontology_id": "http://www.radlex.org/RID/RID6529",
      "ontology_label": "hippocampus",
      "ontology": "RADLEX"
    }
    """

    name: str = "Concept Mapping Tool"
    description: str = (
        "Maps biomedical/neuroscientific text or concepts to ontology identifiers (IRIs) and labels. "
        "Supports single concepts, multiple concepts (comma-separated), or sentences. "
        "Useful for diseases, genes, proteins, chemicals, anatomical structures, etc. "
        "Automatically detects relevant ontologies. "
        "Examples:\n"
        "- Single: 'diabetes'\n"
        "- Multiple: 'diabetes,cancer,asthma'\n"
        "- Sentence: 'diabetic kidney disease'\n"
        "Returns: IRI and label for each matched concept with ontology source."
    )
    args_schema: Type[BaseModel] = ConceptMappingInput

    api_key: str = Field(default_factory=lambda: os.getenv("BIOPORTAL_API_KEY", ""))
    base_url: str = Field(default="https://data.bioontology.org")

    # Allow extra fields for session object
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the tool with BioPortal API key

        Args:
            api_key: BioPortal API key (optional - will read from BIOPORTAL_API_KEY env var if not provided)
        """
        # Read from env if not provided
        if api_key is None:
            api_key = os.getenv("BIOPORTAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "BioPortal API key not found. Either:\n"
                    "1. Pass api_key parameter: ConceptMappingTool(api_key='your-key')\n"
                    "2. Set environment variable: export BIOPORTAL_API_KEY='your-key'\n"
                    "Get API key from: https://bioportal.bioontology.org/account"
                )

        super().__init__(api_key=api_key, **kwargs)

        # Now we can set session after super().__init__
        object.__setattr__(self, "session", requests.Session())
        self.session.headers.update({"Authorization": f"apikey token={api_key}"})
        logger.info("ConceptMappingTool initialized (BIOPORTAL_API_KEY is set; " "requests are throttled to avoid 429 rate limit)")

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make API request with throttling and error handling. Uses BIOPORTAL_API_KEY in Authorization header."""
        global _bioportal_last_request_time
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        interval = _bioportal_interval()
        with _bioportal_throttle_lock:
            now = time.monotonic()
            wait = interval - (now - _bioportal_last_request_time)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, params=params, timeout=10)
                _bioportal_last_request_time = time.monotonic()
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None

        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            backoff = _bioportal_backoff()
            logger.warning(
                "BioPortal API rate limit (429). Retrying once after %.1fs; "
                "set BIOPORTAL_REQUEST_INTERVAL=1.0 or BIOPORTAL_BACKOFF_AFTER_429=3 to reduce 429s.",
                backoff,
            )
            time.sleep(backoff)
            with _bioportal_throttle_lock:
                try:
                    response = self.session.get(url, params=params, timeout=10)
                    _bioportal_last_request_time = time.monotonic()
                except Exception as e:
                    logger.error(f"Retry request failed: {e}")
                    return None
            if response.status_code == 200:
                return response.json()
            with _bioportal_throttle_lock:
                _bioportal_last_request_time = time.monotonic() + backoff
        else:
            logger.warning(f"API error {response.status_code}")
        return None

    def _recommend_ontologies(self, text: str, max_ontologies: int = 10) -> list:
        """Use Recommender API to auto-detect relevant ontologies"""
        text = _sanitize_text(text)
        if not text:
            logger.info("Using fallback ontologies (empty input)")
            return ["SNOMEDCT", "MONDO", "NCIT", "GO", "HP", "CHEBI"]
        params = {"input": text, "input_type": 1, "output_type": 1}

        result = self._make_request("/recommender", params)

        if not result:
            # Fallback when recommender fails (e.g. 429 rate limit or network)
            logger.debug("Using fallback ontologies (recommender unavailable or 429)")
            return ["SNOMEDCT", "MONDO", "NCIT", "GO", "HP", "CHEBI"]

        if not isinstance(result, list):
            logger.debug("Using fallback ontologies (recommender did not return list)")
            return ["SNOMEDCT", "MONDO", "NCIT", "GO", "HP", "CHEBI"]

        ontologies = []
        for item in result[:max_ontologies]:
            if not isinstance(item, dict):
                continue
            ont_list = item.get("ontologies")
            if not isinstance(ont_list, list) or not ont_list:
                continue
            ontology_info = ont_list[0] if isinstance(ont_list[0], dict) else {}
            acronym = (ontology_info.get("acronym") or "").strip()
            score = item.get("evaluationScore")
            try:
                score = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            if acronym and score > 0.1:
                ontologies.append(acronym)

        logger.info(f"Auto-detected ontologies: {ontologies}")
        return ontologies

    def _is_batch_input(self, text: str) -> bool:
        """
        Determine if input is batch (multiple concepts) or single
        Heuristic: Check if contains comma AND multiple distinct terms
        """
        if "," not in text:
            return False

        # Split by comma and check if we have multiple non-empty terms
        terms = [t.strip() for t in text.split(",") if t.strip()]

        # If 2+ short terms (likely list), treat as batch
        # If 1 term with commas (likely sentence), treat as single
        if len(terms) >= 2 and all(len(t.split()) <= 3 for t in terms):
            return True

        return False

    def _map_single_concept(
        self,
        text: str,
        ontology_list: Optional[list],
        max_results: Optional[int] = None,
    ) -> dict:
        """Map a single concept to ontology IRIs and labels. Uses in-memory cache to avoid duplicate API calls."""
        max_results = _normalize_max_results(max_results)
        query = _sanitize_text(text) if text else ""
        if not query:
            return {"error": f"No valid text to map: {text!r}", "ontology_id": None, "ontology_label": None}
        # Cache lookup (same term = one API call across entities and runs)
        cache_key = f"{query}|{max_results}"
        with _CONCEPT_MAPPING_CACHE_LOCK:
            if cache_key in _CONCEPT_MAPPING_CACHE:
                return dict(_CONCEPT_MAPPING_CACHE[cache_key])
        # Auto-detect ontologies if not specified
        if ontology_list is None:
            ontology_list = self._recommend_ontologies(text)

        params = {"q": query, "pagesize": min(max_results, 20), "also_search_obsolete": "false"}
        if ontology_list and isinstance(ontology_list, list):
            acrons = [str(o).strip() for o in ontology_list if str(o).strip()]
            if acrons:
                params["ontologies"] = ",".join(acrons)

        try:
            result = self._make_request("/search", params)
        except Exception as e:
            logger.debug("Concept mapping request failed for %r: %s", query[:80] if query else "", e)
            return {"error": "Concept mapping failed", "ontology_id": None, "ontology_label": None}

        if not result or "collection" not in result:
            return {"error": f"No ontology matches found for: {text}", "ontology_id": None, "ontology_label": None}

        collection = result.get("collection")
        if not isinstance(collection, list):
            return {"error": f"No ontology matches found for: {text}", "ontology_id": None, "ontology_label": None}

        # Extract IRI and label pairs (guard against non-dict items or missing fields)
        matches = []
        for item in collection[:max_results]:
            if not isinstance(item, dict):
                continue
            iri = item.get("@id") or item.get("iri") or ""
            label = item.get("prefLabel") or item.get("label") or ""
            links = item.get("links")
            if isinstance(links, dict):
                ont_url = links.get("ontology")
                ontology = (ont_url.split("/")[-1]) if isinstance(ont_url, str) and ont_url else ""
            else:
                ontology = ""

            iri_str = str(iri).strip() if iri else ""
            label_str = str(label).strip() if label else ""
            if iri_str and label_str:
                matches.append(
                    {
                        "iri": iri_str,
                        "label": label_str,
                        "ontology": str(ontology).strip() if ontology else "",
                    }
                )

        if not matches:
            return {"error": f"No ontology matches found for: {text}", "ontology_id": None, "ontology_label": None}

        logger.info(f"Mapped '{text}' to {len(matches)} concepts")

        # Return format based on number of results; store in cache for reuse
        if max_results == 1 or len(matches) == 1:
            out = {"ontology_id": matches[0]["iri"], "ontology_label": matches[0]["label"], "ontology": matches[0]["ontology"]}
        else:
            out = {
                "ontology_id": [m["iri"] for m in matches],
                "ontology_label": [m["label"] for m in matches],
                "ontology": [m["ontology"] for m in matches],
            }
        # FIFO cache eviction strategy
        with _CONCEPT_MAPPING_CACHE_LOCK:
            max_size = _concept_mapping_cache_max_size()
            if len(_CONCEPT_MAPPING_CACHE) >= max_size:
                _CONCEPT_MAPPING_CACHE.pop(next(iter(_CONCEPT_MAPPING_CACHE)), None)
            _CONCEPT_MAPPING_CACHE[cache_key] = out
        return out

    def _map_batch_concepts(self, text: str, max_results: int, ontology_list: Optional[list]) -> dict:
        """Map multiple concepts (comma-separated) to ontologies"""
        # Parse and sanitize each concept (avoid empty or junk from extra chars)
        seen = set()
        text_list = []
        for t in text.split(","):
            cleaned = _sanitize_text(t)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                text_list.append(cleaned)

        if not text_list:
            return {"error": "No valid concepts provided"}

        logger.info(f"Batch mapping {len(text_list)} concepts")

        # Map each concept
        results = {}
        for concept_text in text_list:
            result = self._map_single_concept(text=concept_text, max_results=max_results, ontology_list=ontology_list)
            results[concept_text] = result

        return results

    def _run(self, text: str, max_results: Optional[int] = None, ontologies: Optional[str] = None) -> dict:
        """
        Execute the tool - automatically handles single or batch mapping

        Args:
            text: Input text or concept(s)
                  - Single: "diabetes"
                  - Batch: "diabetes,cancer,asthma"
                  - Sentence: "diabetic kidney disease"
            max_results: Maximum number of results per concept (default: 1)
                        - max_results=1: Returns single ontology_id and ontology_label
                        - max_results>1: Returns lists of ontology_id and ontology_label
            ontologies: Comma-separated ontology acronyms (optional)

        Returns:
            Dictionary with ontology_id and ontology_label

            Single concept, single result (max_results=1):
            {
                "ontology_id": "http://purl.obolibrary.org/obo/MONDO_0005015",
                "ontology_label": "diabetes mellitus",
                "ontology": "MONDO"
            }

            Single concept, multiple results (max_results>1):
            {
                "ontology_id": ["http://...", "http://..."],
                "ontology_label": ["diabetes mellitus", "Diabetes mellitus"],
                "ontology": ["MONDO", "SNOMEDCT"]
            }

            Batch concepts:
            {
                "diabetes": {
                    "ontology_id": "http://...",
                    "ontology_label": "diabetes mellitus",
                    "ontology": "MONDO"
                },
                "cancer": {
                    "ontology_id": "http://...",
                    "ontology_label": "malignant neoplasm",
                    "ontology": "MONDO"
                }
            }
        """
        # Normalize max_results (agent may pass None or wrong type)
        max_results = _normalize_max_results(max_results)

        # Sanitize text (strip, normalize whitespace, remove control chars, truncate)
        text = _sanitize_text(text)
        if not text:
            return {
                "error": "No valid text provided for concept mapping",
                "ontology_id": None,
                "ontology_label": None,
            }

        # Parse ontologies if provided (only non-empty acronyms)
        ontology_list = None
        raw_ontologies = _sanitize_ontologies_raw(ontologies)
        if raw_ontologies:
            ontology_list = [o.strip() for o in raw_ontologies.split(",") if o.strip()]

        # Determine if batch or single
        is_batch = self._is_batch_input(text)

        if is_batch:
            logger.info(f"Detected batch input: {text}")
            out = self._map_batch_concepts(text, max_results, ontology_list)
        else:
            logger.info(f"Detected single input: {text}")
            out = self._map_single_concept(text, max_results, ontology_list)

        # Record for pipeline so alignment agent's concept mapping can be preserved in final result
        if isinstance(out, dict) and "error" not in out:
            with _ALIGNMENT_TOOL_OUTPUTS_LOCK:
                _ALIGNMENT_TOOL_OUTPUTS.append({"input": text, "output": dict(out)})
        return out
