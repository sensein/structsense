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

import json
from crewai.tools import tool
import spacy
from spacy.cli import download
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_nlp_model = None
MODEL = "en_core_web_sm" #to change later to read fromconfig file.

def get_nlp_model():
    """Get or initialize the spaCy NLP model.

    Automatically downloads the model if it's not found.
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(MODEL)
        except OSError:
            download(MODEL)
            _nlp_model = spacy.load(MODEL)
    return _nlp_model



@tool("extract_ner_terms")
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

    Args:
        text: The text to extract entities from

    Returns:
        JSON string with entities array
    """
    nlp = get_nlp_model()
    doc = nlp(text)
    ents = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
    return json.dumps({"entities": ents}, ensure_ascii=False)