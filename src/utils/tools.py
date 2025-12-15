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
# @File    : tools.py
# @Software: PyCharm

from crewai.tools import tool
import spacy
import json
from typing import List, Dict, Any
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

@tool("spacy_ner_tool")
def extract_ner_terms(text: str) -> str:
    """
       Extract named entities using spaCy NER and return JSON:
       {
         "entities": [
           {"text": "...", "label": "...", "start": int, "end": int},
           ...
         ]
       }
       `start` and `end` are character offsets relative to THIS text (chunk).
       """
    doc = nlp(text)
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    return json.dumps({"entities": entities}, ensure_ascii=False)