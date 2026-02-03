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
import os
from typing import Any, Dict, List, Optional

from crewai.tools import tool
from openai import OpenAI
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

# Source label for LLM-based NER (for postprocessing weights)
LLM_NER_SOURCE_MODEL = "llm_ner"

# Cache for loaded models to avoid reloading
MODEL_CACHE = {}
_spacy_nlp = None
_device = None
_models_warmed_up = False

# Optional domain context for LLM-based NER (set by app when attaching tool with agent/task config)
_ner_domain_context: Optional[Dict[str, Any]] = None


def set_ner_domain_context(
    agent_role: str = "",
    agent_goal: str = "",
    task_description: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    enable_llm_ner: bool = True,
) -> None:
    """Set domain context for extract_ner_terms so it also runs LLM-based NER. Call before attaching the tool to the agent."""
    global _ner_domain_context
    _ner_domain_context = {
        "agent_role": (agent_role or "").strip(),
        "agent_goal": (agent_goal or "").strip(),
        "task_description": (task_description or "").strip(),
        "llm_config": llm_config or {},
        "api_key": api_key,
        "enable_llm_ner": bool(enable_llm_ner),
    } if (agent_role or agent_goal or task_description) and enable_llm_ner else None


def clear_ner_domain_context() -> None:
    """Clear domain context so extract_ner_terms runs ML-only."""
    global _ner_domain_context
    _ner_domain_context = None


def get_ner_domain_context() -> Optional[Dict[str, Any]]:
    """Return current domain context for NER (for tests)."""
    return _ner_domain_context


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
                "entity": ent.text,
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
                    "entity": token_text,
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


def _find_entity_span(text: str, entity_text: str) -> tuple:
    """
    Find first occurrence of entity_text in text (case-insensitive search).
    Returns (start, end) character offsets, or (0, 0) if not found.
    """
    if not entity_text or not text:
        return (0, 0)
    ent = entity_text.strip()
    if not ent:
        return (0, 0)
    text_lower = text.lower()
    ent_lower = ent.lower()
    idx = text_lower.find(ent_lower)
    if idx == -1:
        return (0, 0)
    return (idx, idx + len(ent))


def _safe_parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM output (may be wrapped in markdown or text)."""
    if not raw_text or not raw_text.strip():
        return {}
    text = raw_text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {}


def extract_entities_with_llm(
    text: str,
    agent_role: str = "",
    agent_goal: str = "",
    task_description: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract named entities using an LLM with domain context from extractor agent
    role, goal, and task description. Returns entities in the same format as
    ML-based NER: {"entity", "label", "start", "end", "source_model"}.

    Args:
        text: Input text to extract entities from.
        agent_role: Extractor agent role (domain context).
        agent_goal: Extractor agent goal (domain context).
        task_description: Extraction task description (domain context).
        llm_config: Optional dict with "model" and "base_url" for OpenAI-compatible API.
        api_key: Optional API key; falls back to OPENROUTER_API_KEY env.

    Returns:
        List of entity dicts with entity, label, start, end, source_model.
    """
    if not text or not text.strip():
        return []

    llm_config = llm_config or {}
    base_url = llm_config.get("base_url") or "https://openrouter.ai/api/v1"
    model = llm_config.get("model") or "openai/gpt-4o-mini"

    # Ollama endpoints don't require an API key
    base_lower = (base_url or "").lower()
    is_ollama_or_local = (
        "ollama" in base_lower
        or "localhost" in base_lower
        or "127.0.0.1" in base_lower
        or base_lower.startswith("http://")
    )
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not is_ollama_or_local and not api_key:
        logger.warning(
            "No API key for LLM NER (OpenRouter/OpenAI); skipping LLM-based extraction. "
            "Use Ollama/local base_url or set OPENROUTER_API_KEY/OPENAI_API_KEY for cloud models."
        )
        return []

    # OpenRouter expects model ID without "openrouter/" prefix (e.g. openai/gpt-4o-mini)
    if "openrouter" in base_lower and isinstance(model, str) and model.startswith("openrouter/"):
        model = model.replace("openrouter/", "", 1)
        logger.info(f"LLM NER: normalized OpenRouter model ID to {model!r}")

    print(f"[LLM NER] base_url={base_url!r}, model={model!r}, text_len={len(text)}")
    logger.info(f"LLM NER: base_url={base_url!r}, model={model!r}")

    domain_context = ""
    if agent_role or agent_goal or task_description:
        parts = []
        if agent_role:
            parts.append(f"Agent role: {agent_role.strip()}")
        if agent_goal:
            parts.append(f"Agent goal: {agent_goal.strip()}")
        if task_description:
            # Truncate very long task description for prompt
            desc = task_description.strip()
            if len(desc) > 1500:
                desc = desc[:1500] + "..."
            parts.append(f"Task description: {desc}")
        domain_context = "\n".join(parts)
    else:
        domain_context = "Extract named entities (domain-specific types, such as anatomical regions, experimental conditions based on domains like neurosceice, biomedical)."

    prompt = f"""You are a named entity recognition (NER) expert. Given the domain context below, extract ALL named entities from the input text.

Domain context:
{domain_context}

Output MUST be a single JSON object with exactly one key "entities", whose value is a list of objects. Each object must have:
- "entity": the exact span of text (substring of the input)
- "label": the entity type (e.g. BRAIN_REGION, CELL_TYPE, DISEASE, GENE, PERSON, ORGANIZATION)

Do not include "start" or "end" in the output; they will be computed from the text.

Input text:
---
{text[:12000]}
---

Return only the JSON object, no other text."""

    try:
        print("[LLM NER] Calling API...")
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = (response.choices[0].message.content or "").strip()
        data = _safe_parse_llm_json(raw_text)
        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list):
            entities_raw = []
        print(f"[LLM NER] API success: parsed {len(entities_raw)} entities from response")
        logger.info(f"LLM NER: API success, {len(entities_raw)} entities")
    except Exception as e:
        print(f"[LLM NER] API error: {e}")
        logger.error(f"LLM NER API error (base_url={base_url!r}, model={model!r}): {e}")
        return []

    entities = []
    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        ent_text = (item.get("entity") or item.get("word") or "").strip()
        label = (item.get("label") or item.get("entity_group") or "UNKNOWN").strip()
        if not ent_text:
            continue
        start, end = _find_entity_span(text, ent_text)
        entities.append({
            "entity": ent_text,
            "label": canonical_label(label),
            "start": start,
            "end": end,
            "source_model": LLM_NER_SOURCE_MODEL,
        })
    if entities:
        logger.info(f"✓ LLM NER: Found {len(entities)} entities")
    return entities


@tool("extract_ner_terms")
def extract_ner_terms(text: str) -> str:
    """
    Extract named entities using ML models (spaCy + biomedical) and, when domain
    context is set by the app, LLM-based NER. All combined into one tool.

    Always runs: spaCy + biomedical NER models. If set_ner_domain_context() was
    called with extractor agent role/task description and LLM config, also runs
    LLM-based NER and merges results. Output format is the same either way.

    Returns JSON:
    {
      "entities": [
        {"entity": "...", "label": "...", "source_model": "...", "start": int, "end": int},
        ...
      ]
    }

    ML models: spaCy (en_core_web_sm), d4data/biomedical-ner-all, BC5CDR chemical/disease,
    NCBI-disease, BioBERT genetic NER. When context is set, adds LLM NER (source_model: llm_ner).

    `start` and `end` are character offsets relative to THIS text (chunk).
    `source_model` indicates which model detected the entity.

    Args:
        text: The text to extract entities from

    Returns:
        JSON string with entities array including source model information
    """
    try:
        logger.info(f"Extracting entities from text of length {len(text)}")

        entities = extract_all_entities(text, max_workers=4)

        ctx = get_ner_domain_context()
        if ctx and ctx.get("enable_llm_ner"):
            print("[LLM NER] Domain context set, running LLM-based NER...")
            llm_entities = extract_entities_with_llm(
                text,
                agent_role=ctx.get("agent_role") or "",
                agent_goal=ctx.get("agent_goal") or "",
                task_description=ctx.get("task_description") or "",
                llm_config=ctx.get("llm_config"),
                api_key=ctx.get("api_key"),
            )
            if llm_entities:
                entities = entities + llm_entities

        result = {"entities": entities}
        logger.info(f"Successfully extracted {len(entities)} entities")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in extract_ner_terms: {e}")
        return json.dumps({"entities": []}, ensure_ascii=False)