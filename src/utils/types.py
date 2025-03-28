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

from pydantic import BaseModel
from typing import List, Dict


from pydantic import BaseModel
from typing import List, Dict


class StructuredInformation(BaseModel):
    """Represents a structured information in text."""

    entity: str
    label: str
    sentence: str
    start: int
    end: int
    paper_location: str
    paper_title: str
    doi: str


class AlignedStructuredInformation(StructuredInformation):
    """Represents an aligned term with ontology mapping."""

    ontology_id: str
    ontology_label: str


class JudgedStructuredInformation(AlignedStructuredInformation):
    judge_score: float


class JudgeStructuredTerms(BaseModel):
    aligned_judged_terms: Dict[str, List[JudgedStructuredInformation]]


class AlignedStructuredTerms(BaseModel):
    """Pydantic model to validate aligned  terms."""

    aligned_ner_terms: Dict[str, List[AlignedStructuredInformation]]


class ExtractedStructuredTerms(BaseModel):
    """Pydantic model to validate extracted terms."""

    extracted_terms: Dict[str, List[StructuredInformation]]
