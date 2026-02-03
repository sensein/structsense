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
Unified CrewAI Tool for Ontology Concept Mapping using BioPortal API
Automatically handles single or batch mapping based on input
"""

import os
import requests
import logging
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from .types import ConceptMappingInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConceptMappingTool")


DEFAULT_MAX_CONCEPT_MAPPING_RESULTS = max(
    1,
    min(
        int(os.getenv("MAX_CONCEPT_MAPPING_RESULTS", "1")),
        50,  # safety
    ),
)

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
        "Maps biomedical/scientific text or concepts to ontology identifiers (IRIs) and labels. "
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
        object.__setattr__(self, 'session', requests.Session())
        self.session.headers.update({"Authorization": f"apikey token={api_key}"})
        logger.info("ConceptMappingTool initialized successfully")

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API error {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def _recommend_ontologies(self, text: str, max_ontologies: int = 10) -> list:
        """Use Recommender API to auto-detect relevant ontologies"""
        params = {
            "input": text,
            "input_type": 1,
            "output_type": 1
        }

        result = self._make_request("/recommender", params)

        if not result:
            # Fallback to common ontologies
            logger.info("Using fallback ontologies")
            return ["SNOMEDCT", "MONDO", "NCIT", "GO", "HP", "CHEBI"]

        ontologies = []
        for item in result[:max_ontologies]:
            ontology_info = item.get("ontologies", [{}])[0]
            acronym = ontology_info.get("acronym", "")
            score = item.get("evaluationScore", 0)

            if acronym and score > 0.1:
                ontologies.append(acronym)

        logger.info(f"Auto-detected ontologies: {ontologies}")
        return ontologies

    def _is_batch_input(self, text: str) -> bool:
        """
        Determine if input is batch (multiple concepts) or single
        Heuristic: Check if contains comma AND multiple distinct terms
        """
        if ',' not in text:
            return False

        # Split by comma and check if we have multiple non-empty terms
        terms = [t.strip() for t in text.split(',') if t.strip()]

        # If 2+ short terms (likely list), treat as batch
        # If 1 term with commas (likely sentence), treat as single
        if len(terms) >= 2 and all(len(t.split()) <= 3 for t in terms):
            return True

        return False

    def _map_single_concept(
            self,
            text: str,
            ontology_list: Optional[list],
            max_results: int = DEFAULT_MAX_CONCEPT_MAPPING_RESULTS,
    ) -> dict:
        """Map a single concept to ontology IRIs and labels"""
        # Auto-detect ontologies if not specified
        if ontology_list is None:
            ontology_list = self._recommend_ontologies(text)

        # Search for concepts
        params = {
            "q": text,
            "pagesize": min(max_results, 20),
            "also_search_obsolete": "false"
        }

        if ontology_list:
            params["ontologies"] = ",".join(ontology_list)

        result = self._make_request("/search", params)

        if not result or "collection" not in result:
            return {
                "error": f"No ontology matches found for: {text}",
                "ontology_id": None,
                "ontology_label": None
            }

        # Extract IRI and label pairs
        matches = []
        for item in result["collection"][:max_results]:
            iri = item.get("@id", "")
            label = item.get("prefLabel", "")
            ontology = item.get("links", {}).get("ontology", "").split("/")[-1]

            if iri and label:
                matches.append({
                    "iri": iri,
                    "label": label,
                    "ontology": ontology
                })

        if not matches:
            return {
                "error": f"No ontology matches found for: {text}",
                "ontology_id": None,
                "ontology_label": None
            }

        logger.info(f"Mapped '{text}' to {len(matches)} concepts")

        # Return format based on number of results
        if max_results == 1 or len(matches) == 1:
            # Single result - return as single values
            return {
                "ontology_id": matches[0]["iri"],
                "ontology_label": matches[0]["label"],
                "ontology": matches[0]["ontology"]
            }
        else:
            # Multiple results - return as lists
            return {
                "ontology_id": [m["iri"] for m in matches],
                "ontology_label": [m["label"] for m in matches],
                "ontology": [m["ontology"] for m in matches]
            }

    def _map_batch_concepts(
            self,
            text: str,
            max_results: int,
            ontology_list: Optional[list]
    ) -> dict:
        """Map multiple concepts (comma-separated) to ontologies"""
        # Parse input texts
        text_list = [t.strip() for t in text.split(",") if t.strip()]

        if not text_list:
            return {"error": "No valid concepts provided"}

        logger.info(f"Batch mapping {len(text_list)} concepts")

        # Map each concept
        results = {}
        for concept_text in text_list:
            result = self._map_single_concept(
                text=concept_text,
                max_results=max_results,
                ontology_list=ontology_list
            )
            results[concept_text] = result

        return results

    def _run(
            self,
            text: str,
            max_results: int = 1,
            ontologies: Optional[str] = None
    ) -> dict:
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
        # Parse ontologies if provided
        ontology_list = None
        if ontologies:
            ontology_list = [o.strip() for o in ontologies.split(",")]

        # Determine if batch or single
        is_batch = self._is_batch_input(text)

        if is_batch:
            logger.info(f"Detected batch input: {text}")
            return self._map_batch_concepts(text, max_results, ontology_list)
        else:
            logger.info(f"Detected single input: {text}")
            return self._map_single_concept(text, max_results, ontology_list)


