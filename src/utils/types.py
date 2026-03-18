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
# @File    : types.py
# @Software: PyCharm

from typing import Any, Dict, List, Union
from typing import Optional, Type
from pydantic import BaseModel, Field


class ExtractedTermsDynamic(BaseModel):
    extracted_structured_information: Any


class AlignedTermsDynamic(BaseModel):
    aligned_structured_information: Any


class JudgedTermsDynamic(BaseModel):
    judged_structured_information: Any


class ConceptMappingInput(BaseModel):
    """Input schema for ConceptMappingTool"""
    text: Union[str, List[Union[str, Dict[str, Any]]]] = Field(
        ...,
        description=(
            "Concepts to map — ALWAYS pass ALL terms together in ONE call.\n\n"
            "PREFERRED — list of dicts with per-term context (best accuracy):\n"
            '   [{"text": "hippocampus", "context": "Neurons in CA1 were recorded."}, '
            '{"text": "cortex", "context": "Prefrontal cortex activity was measured."}]\n\n'
            "Also accepted:\n"
            "- List of strings: ['hippocampus', 'cortex', 'amygdala']\n"
            "- Single string or phrase: 'hippocampus'  or  'diabetic kidney disease'\n\n"
            "Do NOT call this tool once per term. Pass every term from the current passage in a single call."
        )
    )
    max_results: int = Field(
        default=1,
        description="Maximum number of results per concept (1-20)"
    )
    ontologies: Optional[str] = Field(
        default=None,
        description="Comma-separated ontology acronyms (e.g., 'MONDO,NCIT,SNOMEDCT'). Auto-detected if not provided."
    )
    context: Optional[str] = Field(
        default=None,
        description=(
            "Optional shared context/sentence for disambiguation — applied to all terms when 'text' is "
            "a plain comma-separated string. Ignored when 'text' is a JSON array (each object has its own context). "
            "Example: 'Hippocampal neurons in CA1 were recorded during spatial navigation.'"
        )
    )