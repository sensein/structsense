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
# @File    : ner_tool.py
# @Software: PyCharm

import json
from crewai.tools import tool
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from spacy.cli import download
import torch
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
_device = None
_models_warmed_up = False


def get_device():
    """
    Auto-detect the best available device (CUDA GPU, MPS, or CPU).

    Returns:
        int or str: Device identifier (0 for GPU, "mps" for Apple Silicon, -1 for CPU)
    """
    global _device

    if _device is not None:
        return _device

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        _device = 0
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {gpu_name}")
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device = "mps"
        logger.info("Using Apple Silicon MPS")
    # Default to CPU
    else:
        _device = -1
        logger.info("Using CPU")

    return _device


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
    """
    Load a NER pipeline with explicit control to avoid meta tensors.

    Key changes to prevent meta tensor issues:
    - Load model/tokenizer explicitly (not by name in pipeline)
    - Use low_cpu_mem_usage=False to prevent meta initialization
    - Move to device manually
    - Sequential loading only (no threading during load)
    """
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    try:
        device = get_device()

        # Determine torch device string
        if device == 0:
            torch_device = "cuda:0"
            use_fp16 = True
        elif device == "mps":
            torch_device = "mps"
            use_fp16 = False
        else:
            torch_device = "cpu"
            use_fp16 = False

        logger.info(f"Loading {name} on {torch_device} (no-meta path)")

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(
            name,
            use_fast=True,
            trust_remote_code=True
        )

        # Load model with explicit settings to avoid meta tensors
        model = AutoModelForTokenClassification.from_pretrained(
            name,
            trust_remote_code=True,
            device_map=None,              # Don't use device_map (causes meta tensors)
            low_cpu_mem_usage=False,      # CRITICAL: Prevents meta initialization
            torch_dtype=torch.float16 if use_fp16 else None,
        )

        # Move model to device
        model.to(torch_device)
        model.eval()

        # Create pipeline with the loaded model
        ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=0 if torch_device.startswith("cuda") else -1,
        )

        id2label = getattr(model.config, "id2label", {}) or {}
        MODEL_CACHE[name] = (ner, id2label)

        logger.info(f"✓ Successfully loaded {name} on {torch_device}")
        return ner, id2label

    except Exception as e:
        logger.error(f"✗ Error loading model {name}: {e}")
        MODEL_CACHE[name] = (None, {})
        return None, {}


def warmup_models():
    """
    Warm up all models sequentially before any threaded inference.
    This prevents concurrent loading which can cause issues.
    """
    global _models_warmed_up

    if _models_warmed_up:
        return

    logger.info("Warming up models sequentially...")

    # Load spaCy model
    try:
        get_spacy_model()
        logger.info("✓ spaCy model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load spaCy: {e}")

    # Load all biomedical models sequentially
    for model_name in HF_MODEL_NAMES:
        load_pipe(model_name)

    _models_warmed_up = True
    logger.info("All models warmed up and cached!")


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
                "source_model": SPACY_MODEL,
            }
            for ent in doc.ents
        ]

        return entities

    except Exception as e:
        logger.error(f"Error processing with spaCy: {e}")
        return []


def process_text_with_model(text, model_name):
    """
    Process text with a single model.
    Assumes model is already loaded in cache (via warmup_models).
    """
    try:
        ner, id2label = load_pipe(model_name)

        if ner is None:
            return []

        try:
            raw = ner(text)
        except Exception as e:
            logger.error(f"Error during inference with {model_name}: {e}")
            return []

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
                    "source_model": model_name,
                })
            except Exception as e:
                logger.error(f"Error processing entity from {model_name}: {e}")
                continue

        return entities

    except Exception as e:
        logger.error(f"✗ Error processing with {model_name}: {e}")
        return []


def extract_all_entities(text, max_workers=4):
    """
    Extract all entities from all models (spaCy + biomedical) in parallel.
    Models must be warmed up first via warmup_models().
    """
    if not text or not text.strip():
        return []

    # Ensure models are loaded sequentially first
    warmup_models()

    logger.info(f"Processing text with spaCy + {len(HF_MODEL_NAMES)} biomedical models")

    all_entities = []

    # Process all models in parallel (inference only, models already loaded)
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
        {"text": "...", "label": "...", "source_model": "...", "start": int, "end": int},
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
    `source_model` indicates which model detected the entity.

    Args:
        text: The text to extract entities from

    Returns:
        JSON string with entities array including source model information
    """
    try:
        logger.info(f"Extracting entities from text of length {len(text)}")

        # Extract all entities (models warmed up automatically)
        entities = extract_all_entities(text, max_workers=4)

        result = {"entities": entities}

        logger.info(f"Successfully extracted {len(entities)} entities")

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in extract_ner_terms: {e}")
        return json.dumps({"entities": []}, ensure_ascii=False)