"""Task-type detection from task configuration (LLM-based taxonomy classification).

This module classifies the extractor task (e.g. from task description in config)
into a canonical task type (ner, resource, extraction, keyphrase_extraction, etc.)
so the pipeline can attach the right tools and post-processors.

Key components
--------------
- **DEFAULT_TAXONOMY** (dict): Task type → description (and optional schema/context).
  Used to build the classification prompt. Includes ner, resource, extraction,
  keyphrase_extraction, relation_extraction, marr_extraction, etc.
- **TOOLS_BY_TASK_TYPE** (dict): Task type → list of tool names (e.g. ner → ["extract_ner_terms"]).
  Consumed by :mod:`task_tools` when resolving tools for the extractor agent.
- :class:`TaskDetection` : Dataclass holding task_type, confidence, rationale, and raw LLM output.
- :func:`detect_task_type` : Run LLM-based classification and return :class:`TaskDetection`.

See Also
--------
- :mod:`task_tools` – Uses :data:`TOOLS_BY_TASK_TYPE` to attach tools per task type.
- :mod:`postprocessing` – Uses task type to select post-processor and result merger.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Tuple
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskDetection:
    """Structured result of LLM-based task-type detection.

    Attributes
    ----------
    task_type : str
        Canonical type from taxonomy (e.g. ner, resource, extraction).
    confidence : float
        Confidence score in [0.0, 1.0].
    labels : list of str
        Optional sublabels (e.g. "multi_label", "few_shot").
    rationale : str
        Short explanation from the model.
    raw : dict
        Full parsed JSON from the LLM response.
    """

    task_type: str  # canonical type from taxonomy
    confidence: float  # 0.0 - 1.0
    labels: List[str]  # optional sublabels (e.g., "multi_label", "few_shot")
    rationale: str  # short explanation
    raw: Dict[str, Any]  # full parsed JSON from LLM


DEFAULT_TAXONOMY: Dict[str, Union[str, Dict[str, Any]]] = {
    # ----------------------------
    # Taxonomy for intelligent task detection based on prompt aka task description for extractor agent.
    # ----------------------------
    "extraction": (
        "Generic information extraction when the specific subtype is unclear. "
        "Use for pulling salient structured facts from text (often as JSON) without committing to a narrower label."
    ),
    "ner": (
        "Named Entity Recognition: detect and extract spans that refer to real-world or domain-specific entities. "
        "Common entity types include PERSON, ORGANIZATION, LOCATION, DATE/TIME, PRODUCT, EVENT. "
        "Domain extensions should include scientific/technical entities such as DISEASE, DRUG, GENE/PROTEIN, "
        "CELL_TYPE, ANATOMICAL_REGION, MOLECULE/RECEPTOR, PATHWAY, ASSAY/METHOD, DATASET, METRIC, MODEL, "
        "BRAIN_REGION, NEUROTRANSMITTER, DISORDER, CLINICAL_TRIAL, and DEVICE/INSTRUMENT. "
        "Output is typically entity spans with type labels and (optionally) normalization IDs."
    ),
    "relation_extraction": (
        "Extract typed relationships between entities (often from NER output). "
        "Examples: WORKS_FOR(person, org), LOCATED_IN(org, location), TREATS(drug, disease), "
        "BINDS(protein, receptor), EXPRESSED_IN(gene, cell_type), PROJECTS_TO(region_a, region_b), "
        "CAUSES(risk_factor, outcome), ASSOCIATED_WITH(variant, phenotype). "
        "Output is usually (subject, relation, object) triples with evidence spans."
    ),
    "event_extraction": (
        "Extract events and their arguments/roles (who did what to whom, when, where, how). "
        "Examples: 'drug administration' (agent, dose, route, time), 'gene knockout' (gene, organism), "
        "'neural firing increase' (region/cell type, condition), 'clinical trial enrollment' (population, arm). "
        "Outputs often include trigger, event type, arguments, temporal info, and confidence."
    ),
    "keyphrase_extraction": (
        "Extract a concise set of important phrases that capture the main topics, mechanisms, or claims. "
        "Compared to NER, keyphrases can be broader concepts (e.g., 'synaptic plasticity', 'reward prediction error', "
        "'transformer attention', 'single-cell RNA-seq'). Output is a ranked list, optionally with relevance scores."
    ),
    "structured_extraction": (
        "Extract information into a predefined schema (e.g., JSON) with strict fields and types. "
        "Use when you have a template like {patient_age, diagnosis, medication_list} or "
        "{paper_title, method, dataset, results}. Prioritize completeness, correct typing, and traceability to source spans."
    ),
    "resource": (
        "Extract resources that a paper or document targets: datasets, tools, models, software, benchmarks, and related papers. "
        "Output is a list of resource objects with name, description, type (Dataset, Tool, Model, etc.), category, target, "
        "specific_target, url, and mentions (datasets, models, tools, papers). Use when the task asks for resource extraction, "
        "extracted resources, or structured resource schema with mentions."
    ),
    # ----------------------------
    # Marr’s framework extraction (3-level organization) - see https://github.com/sensein/ner_reader/blob/main/docs/README_marr_system.md
    # ----------------------------
    "marr_extraction": (
        "Organize extracted items into Marr’s three explanatory levels. "
        "Return entities and/or statements categorized into L1 (Computational), L2 (Algorithmic), and L3 (Implementational). "
        "Useful for neuroscience/biomedicine summaries where claims span goals, algorithms, and substrates."
    ),
    # Helper context
    "marr_extraction_schema": {
        "L1_computational_level": {
            "description": "What & Why: behavioral goals, tasks, and computational objectives the system solves.",
            "examples": ["visual form discrimination", "spatial navigation", "reward prediction"],
            "entity_types": ["BEHAVIORAL_TASK", "COGNITIVE_PROCESS", "COMPUTATIONAL_OBJECTIVE"],
        },
        "L2_algorithmic_level": {
            "description": "How: representations, algorithms, learning rules, neural coding schemes, and dynamics.",
            "examples": ["place coding", "temporal difference learning", "population vector coding"],
            "entity_types": ["NEURAL_CODING_SCHEME", "COMPUTATION_ALGORITHM", "NEURAL_DYNAMICS"],
        },
        "L3_implementational_level": {
            "description": "Physical substrate: biological 'hardware' such as cells, molecules, regions, and circuits.",
            "examples": ["pyramidal neurons", "dopamine", "hippocampus", "NMDA receptors"],
            "entity_types": ["ANATOMICAL_REGION", "CELL_TYPE", "MOLECULAR_COMPONENT"],
        },
    },
    # ----------------------------
    # Classification
    # ----------------------------
    "classification": (
        "Assign exactly one category/label to an input (single-label). "
        "Use for routing or categorizing a document, message, claim, or record."
    ),
    "multi_label_classification": (
        "Assign multiple applicable labels/tags to the same input (not mutually exclusive). "
        "Use when content spans multiple topics/intents."
    ),
    "sentiment_analysis": (
        "Detect sentiment/polarity (e.g., positive/negative/neutral) and optionally intensity. "
        "Use for opinions, reviews, or affective language (not factual scientific tone unless explicitly needed)."
    ),
    "intent_classification": (
        "Identify the user’s goal or requested action (e.g., 'extract fields', 'summarize', 'debug code', 'plan trip'). "
        "Used for agent routing and tool selection."
    ),
    "topic_classification": (
        "Categorize text by subject area (e.g., neuroscience, biomedicine, finance, legal). "
        "Useful for indexing, routing, and downstream specialized handling."
    ),
    # ----------------------------
    # QA & Retrieval
    # ----------------------------
    "question_answering": (
        "Answer a question directly. May be extractive (from provided text) or abstractive (synthesized). "
        "Should prioritize correctness, citing provided context when available."
    ),
    "retrieval_augmented_qa": (
        "Answer a question by first retrieving relevant documents/passages, then synthesizing an answer grounded in them. "
        "Use when the answer depends on external context or a knowledge base."
    ),
    # ----------------------------
    # Transformation
    # ----------------------------
    "summarization": (
        "Compress text while preserving the key points. Can be abstractive or extractive. "
        "Often parameterized by length, audience, and focus (e.g., 'methods-focused', 'clinical implications')."
    ),
    "translation": (
        "Translate text between languages while preserving meaning, tone, and technical terminology. "
        "For scientific content, preserve entity names, units, gene/protein symbols, and citations carefully."
    ),
    "paraphrase_rewrite": (
        "Rewrite text to preserve meaning while changing wording/structure. "
        "Used for clarity, de-duplication, or adapting to constraints without changing the facts."
    ),
    "style_transfer": (
        "Rewrite text to match a target style/tone/voice (e.g., formal, concise, layperson-friendly, academic). "
        "Meaning should remain stable unless asked to change content."
    ),
    # ----------------------------
    # Generation
    # ----------------------------
    "generation": (
        "Generate new content that is not a strict transformation of the input. "
        "Includes drafting, proposing, inventing examples, or composing content from instructions."
    ),
    "creative_writing": (
        "Generate imaginative or narrative text such as stories, scripts, poems, or dialogue. "
        "Primary objective is creativity and stylistic quality."
    ),
    "data_to_text": (
        "Generate natural language from structured data (tables/JSON/metrics). "
        "Should be faithful to the data, mention key trends, and avoid hallucinating numbers."
    ),
    # ----------------------------
    # Code
    # ----------------------------
    "code_generation": (
        "Write code based on requirements/specifications. " "Includes creating functions, scripts, APIs, tests, and integrations."
    ),
    "code_explanation": (
        "Explain what code does, how it works, and why. " "May include step-by-step walkthroughs, complexity notes, and edge cases."
    ),
    "code_refactoring": (
        "Improve code structure/readability/maintainability without changing behavior. "
        "Includes renaming, modularizing, simplifying logic, and improving performance safely."
    ),
    # ----------------------------
    # Evaluation
    # ----------------------------
    "evaluation_grading": (
        "Score or judge outputs against criteria or a rubric (correctness, completeness, style, safety, etc.). "
        "Often returns a grade plus rationale and suggested fixes."
    ),
    "fact_checking": (
        "Assess factual accuracy of claims and identify unsupported statements. "
        "Prefer explicit evidence, highlight uncertainty, and separate verified vs unverified content."
    ),
    # ----------------------------
    # Fallback
    # ----------------------------
    "other": ("Use only when the task does not fit any defined category. " "If possible, prefer the closest match rather than 'other'."),
}

# ----------------------------
# Tools by task type (taxonomy-aligned)
# ----------------------------
#: Task type → list of tool names for the extractor agent. Keys must match taxonomy.
#: Used by :mod:`task_tools` to resolve tools; task types not listed get no tools.
TOOLS_BY_TASK_TYPE: Dict[str, List[str]] = {
    "ner": ["extract_ner_terms"],
    "keyphrase_extraction": ["extract_ner_terms"],
    # extraction, structured_extraction, relation_extraction, etc. → no entry = no tools
}


def get_tool_names_for_task_type(task_type: str) -> List[str]:
    """Return tool names for a task type from :data:`TOOLS_BY_TASK_TYPE`.

    Only task types listed in :data:`TOOLS_BY_TASK_TYPE` receive tools;
    all others return an empty list.

    Parameters
    ----------
    task_type : str
        Canonical task type (e.g. from :func:`detect_task_type`).

    Returns
    -------
    list of str
        Tool names to resolve via :func:`task_tools._resolve_tool`.
    """
    if not task_type:
        return []
    return list(TOOLS_BY_TASK_TYPE.get(str(task_type).strip().lower(), []))


def _split_taxonomy(taxonomy: Dict[str, Union[str, Dict[str, Any]]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Split taxonomy into:
      - task_descriptions: {task_type: description}
      - extra_context: misc objects that are helpful context but not valid task types
    """
    task_descriptions: Dict[str, str] = {}
    extra_context: Dict[str, Any] = {}

    for k, v in taxonomy.items():
        if isinstance(v, str):
            task_descriptions[k] = v
        else:
            # Anything non-string is treated as additional context (not a selectable task_type)
            extra_context[k] = v

    return task_descriptions, extra_context


def _build_task_detection_prompt(taskconfig: str, taxonomy: Dict[str, Union[str, Dict[str, Any]]]) -> str:
    """
    Strict JSON-only instruction with schema, enriched taxonomy descriptions, and optional helper context.
    """
    task_descriptions, extra_context = _split_taxonomy(taxonomy)
    task_types = sorted(task_descriptions.keys())

    taxonomy_block = [{"task_type": t, "description": task_descriptions[t]} for t in task_types]

    prompt_obj = {
        "role": "task-classification-engine",
        "instruction": (
            "Read TASK_CONFIG and choose the single best matching task_type from TAXONOMY.\n"
            "Return ONLY valid JSON matching the schema. No markdown. No extra keys.\n"
            "Prefer the most specific matching task. If unclear, choose 'other' with low confidence."
        ),
        "taxonomy": taxonomy_block,
        "additional_context": extra_context,
        "task_config": taskconfig,
        "json_schema": {
            "task_type": f"<one of: {task_types}>",
            "confidence": "number between 0 and 1",
            "labels": "list of short strings (optional)",
            "rationale": "1-2 sentences, concise",
        },
        "rules": [
            "Choose exactly ONE task_type from the taxonomy.",
            "If multiple fit, pick the primary intent (most central).",
            "Prefer more specific types (e.g., 'ner' over 'extraction').",
            "If task mentions Marr-level organization (L1/L2/L3), choose 'marr_extraction'.",
            "Do not output any keys other than: task_type, confidence, labels, rationale.",
        ],
    }

    return json.dumps(prompt_obj, indent=2)


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Attempts strict JSON parse; if model wraps with text, tries to extract the first JSON object.
    """
    text = (text or "").strip()

    # First try strict
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("LLM output is not valid JSON.")


def detect_task_type(
    taskconfig: str,
    api_key: str,
    llm_config: Dict[str, Any],
    taxonomy: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
) -> TaskDetection:
    """Classify task type from task configuration using an LLM and taxonomy.

    Sends the task config (e.g. task description) and taxonomy to the configured
    LLM (OpenRouter/OpenAI) and parses the response into a single task_type
    with confidence and rationale. Used by the pipeline to select tools and
    post-processors.

    Parameters
    ----------
    taskconfig : str
        CrewAI task configuration (typically JSON or string with task description).
    api_key : str
        API key for the LLM (e.g. OpenRouter or OpenAI).
    llm_config : dict
        LLM settings: ``base_url``, ``model``, and any provider-specific options.
    taxonomy : dict, optional
        Task type → description (and optional context). Defaults to :data:`DEFAULT_TAXONOMY`.

    Returns
    -------
    TaskDetection
        Structured result with task_type, confidence, labels, rationale, and raw JSON.

    Raises
    ------
    ValueError
        If the LLM output is not valid JSON or does not match the expected schema.
    """
    taxonomy = taxonomy or DEFAULT_TAXONOMY

    # Use rich taxonomy in prompt
    prompt = _build_task_detection_prompt(taskconfig, taxonomy)

    logger.info("Running LLM-based task detection...")

    base_url = (llm_config.get("base_url") or "").lower()
    model = llm_config.get("model") or "openai/gpt-4o-mini"
    # OpenRouter expects model ID without "openrouter/" prefix (e.g. openai/gpt-4o-mini)
    if "openrouter" in base_url and isinstance(model, str) and model.startswith("openrouter/"):
        model = model.replace("openrouter/", "", 1)

    client = OpenAI(
        base_url=llm_config["base_url"],
        api_key=api_key,
    )

    completion_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": True}},
    )

    raw_text = completion_response.choices[0].message.content

    try:
        data = _safe_parse_json(raw_text)
    except Exception as e:
        logger.exception("Task detection failed to parse JSON; falling back to 'other'.")
        return TaskDetection(
            task_type="other",
            confidence=0.0,
            labels=["parse_error"],
            rationale=f"Failed to parse LLM JSON output: {e}",
            raw={"llm_text": raw_text},
        )

    # Normalize / validate
    task_descriptions, _extra_context = _split_taxonomy(taxonomy)
    allowed_task_types = set(task_descriptions.keys())

    task_type = str(data.get("task_type", "other")).strip()
    confidence = float(data.get("confidence", 0.0))
    labels = data.get("labels", [])
    rationale = str(data.get("rationale", "")).strip()

    if task_type not in allowed_task_types:
        logger.warning(f"LLM returned out-of-taxonomy task_type='{task_type}'. Forcing 'other'.")
        task_type = "other"
        confidence = min(confidence, 0.3)
        if isinstance(labels, list):
            labels = list(set(labels + ["out_of_taxonomy"]))
        else:
            labels = ["out_of_taxonomy"]

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    if not isinstance(labels, list):
        labels = [str(labels)]

    result = TaskDetection(
        task_type=task_type,
        confidence=confidence,
        labels=[str(x) for x in labels],
        rationale=rationale,
        raw=data,
    )

    logger.info(f"Detected task type: {result.task_type} (confidence={result.confidence:.2f})")
    return result
