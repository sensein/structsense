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
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from spacy.cli import download
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Biomedical NER models
HF_MODEL_NAMES = [
    "d4data/biomedical-ner-all",
    "mobashgr/BC5CDR-chem-WLT-384-BioELECTRA-Pubmed-ENS-20-5",
    "mobashgr/NCBI-disease-WLT-256-SciBERT-13INS",
    "alvaroalon2/biobert_genetic_ner",
]

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Cache for loaded models to avoid reloading
MODEL_CACHE = {}
_spacy_nlp = None


def get_spacy_model():
    """Get or initialize the spaCy NLP model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load(SPACY_MODEL)
        except OSError:
            logger.info(f"Downloading spaCy model: {SPACY_MODEL}")
            download(SPACY_MODEL)
            _spacy_nlp = spacy.load(SPACY_MODEL)
    return _spacy_nlp


def load_pipe(name):
    """Load a NER pipeline with caching and error handling."""
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    try:
        logger.info(f"Loading model: {name}")
        model = AutoModelForTokenClassification.from_pretrained(
            name,
            low_cpu_mem_usage=False,
            trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=-1,  # CPU, change to 0 for GPU
        )
        id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
        MODEL_CACHE[name] = (pipe, id2label)
        return pipe, id2label
    except Exception as e:
        logger.error(f"Error loading model {name}: {e}")
        MODEL_CACHE[name] = (None, {})
        return None, {}


def canonical_label(label: str) -> str:
    """Convert BIO labels like B-DISEASE -> DISEASE."""
    if not label:
        return "UNKNOWN"
    return label.split("-", 1)[1] if "-" in label else label


def process_text_with_spacy(text):
    """Process text with spaCy model."""
    try:
        nlp = get_spacy_model()
        doc = nlp(text)

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

        return entities

    except Exception as e:
        logger.error(f"Error processing with spaCy: {e}")
        return []


def process_text_with_model(text, model_name):
    """Process text with a single model."""
    try:
        ner, id2label = load_pipe(model_name)

        if ner is None:
            return []

        try:
            raw = ner(text)
        except RuntimeError as e:
            if "meta tensors" in str(e):
                logger.warning(f"Skipping {model_name} due to meta tensor issue")
                if model_name in MODEL_CACHE:
                    del MODEL_CACHE[model_name]
                return []
            raise

        entities = []
        for ent in raw:
            try:
                token_text = ent.get("word", "")
                if not token_text:
                    continue

                # Resolve label
                group = ent.get("entity_group", "UNKNOWN")
                if isinstance(group, str) and group.startswith("LABEL_"):
                    idx = int(group.replace("LABEL_", ""))
                    group = id2label.get(idx, group)

                label = canonical_label(str(group))

                entities.append({
                    "text": token_text,
                    "label": label,
                    "start": int(ent.get("start", 0)),
                    "end": int(ent.get("end", 0)),
                })
            except Exception as e:
                logger.error(f"Error processing entity from {model_name}: {e}")
                continue

        return entities

    except Exception as e:
        logger.error(f"Error processing with {model_name}: {e}")
        return []


def extract_all_entities(text, max_workers=4):
    """Extract all entities from all models (spaCy + biomedical) in parallel."""
    if not text or not text.strip():
        return []

    logger.info(f"Processing text with spaCy + {len(HF_MODEL_NAMES)} biomedical models")

    all_entities = []

    # Process all models in parallel (including spaCy)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit spaCy processing
        futures[executor.submit(process_text_with_spacy, text)] = "spaCy"

        # Submit biomedical model processing
        for model_name in HF_MODEL_NAMES:
            futures[executor.submit(process_text_with_model, text, model_name)] = model_name

        # Collect results as they complete
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                entities = future.result(timeout=60)
                if entities:
                    all_entities.extend(entities)
                    logger.info(f"✓ {model_name}: Found {len(entities)} entities")
            except Exception as e:
                logger.error(f"✗ {model_name}: Error - {e}")

    logger.info(f"Total entities extracted: {len(all_entities)}")

    return all_entities


@tool("extract_ner_terms")
def extract_ner_terms(text: str) -> str:
    """
    Extract named entities using spaCy + biomedical NER models and return JSON:
    {
      "entities": [
        {"text": "...", "label": "...", "start": int, "end": int},
        ...
      ]
    }

    Uses:
    - spaCy (en_core_web_sm): General named entities
    - d4data/biomedical-ner-all: General biomedical entities
    - BC5CDR models: Chemical and disease entities
    - NCBI-disease models: Disease entities
    - BioBERT genetic NER: Gene and protein entities

    `start` and `end` are character offsets relative to THIS text (chunk).

    Args:
        text: The text to extract entities from

    Returns:
        JSON string with entities array
    """
    try:
        logger.info(f"Extracting entities from text of length {len(text)}")

        # Extract all entities without any processing
        entities = extract_all_entities(text, max_workers=4)

        result = {"entities": entities}

        logger.info(f"Successfully extracted {len(entities)} entities")

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in extract_ner_terms: {e}")
        return json.dumps({"entities": []}, ensure_ascii=False)