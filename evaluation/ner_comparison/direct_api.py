"""
Direct LiteLLM call for neuroscientific NER.

Single-call baseline: no CrewAI orchestration, no ontology alignment, no multi-agent staging.
Labels are assigned freely by the LLM — no fixed category list is provided.
Comparison against StructSense output uses canonical label normalization from ner_eval.py.
"""

import json
import logging
import os
from typing import Any

import litellm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert neuroscientist and NLP specialist. Extract ALL named entities from the \
neuroscience text provided. For each entity, assign the most precise and descriptive label \
you think is appropriate — do not constrain yourself to a fixed list of categories.

Guidelines:
- Extract the exact surface form from the text — do not paraphrase or normalize.
- Include the character offsets (start, end) within the text you were given.
- Include the complete sentence from the text that contains the entity in the "sentence" field.
- Do not hallucinate entities that are not present in the text.
- Assign labels that reflect the entity's scientific role (e.g. BRAIN_REGION, CELL_TYPE, \
GENE, DISEASE, METHOD, CHEMICAL, SPECIES, SOFTWARE) — but use whatever label best fits.
- If an entity is genuinely ambiguous, pick the most specific label.

Return ONLY a JSON object in this exact format — no commentary, no markdown fences:
{
  "entities": [
    {"entity": "<surface form>", "label": "<YOUR_LABEL>", "start": <int>, "end": <int>, "sentence": "<complete sentence containing this entity>"},
    ...
  ]
}
"""


def extract_entities(
    text: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    extra_litellm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single LiteLLM completion call to extract neuroscientific entities.

    Args:
        text: Raw input text (PDF-extracted, pre-processed).
        model: LiteLLM model string, e.g. "openrouter/openai/gpt-4o-mini".
        base_url: Optional API base URL (required for OpenRouter or local endpoints).
        api_key: API key. Falls back to OPENROUTER_API_KEY or OPENAI_API_KEY env vars.
        temperature: Sampling temperature — use 0.0 for deterministic output.
        max_tokens: Max tokens in the completion response.
        extra_litellm_kwargs: Any additional kwargs passed directly to litellm.completion().

    Returns:
        Dict with keys:
          "entities": list of {entity, label, start, end, sentence}
          "model": model string used
          "usage": token usage dict from the API response
          "raw_response": raw completion text (for debugging)
    """
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all neuroscientific named entities from the following text:\n\n{text}"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if resolved_key:
        kwargs["api_key"] = resolved_key
    if extra_litellm_kwargs:
        kwargs.update(extra_litellm_kwargs)

    logger.info("direct_api: calling model=%r, text_len=%d", model, len(text))
    response = litellm.completion(**kwargs)

    raw = response.choices[0].message.content or ""
    usage = json.loads(json.dumps(dict(response.usage), default=lambda o: vars(o) if hasattr(o, "__dict__") else str(o))) if response.usage else {}

    if response.usage and response.usage.completion_tokens >= max_tokens:
        logger.warning("direct_api: completion_tokens hit max_tokens=%d — response likely truncated", max_tokens)

    entities = _parse_entities(raw)
    logger.info("direct_api: extracted %d entities", len(entities))

    return {
        "entities": entities,
        "model": model,
        "usage": usage,
        "raw_response": raw,
    }


def extract_entities_chunked(
    text: str,
    model: str,
    chunk_size: int = 30000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract entities by splitting text into chunks and merging results.

    Uses StructSense's sentence-boundary chunker (text_chunking._chunk_doc_by_sentences)
    and offset globalizer (_globalize_entities) when spaCy is available, so each
    returned entity has an accurate sentence field drawn directly from the document
    text — the same source StructSense uses. Falls back to paragraph-boundary
    splitting if spaCy is unavailable.

    Args:
        text: Full input text.
        model: LiteLLM model string.
        chunk_size: Max characters per chunk (default 30000).
        **kwargs: Passed through to extract_entities() (e.g. max_tokens, api_key).

    Returns:
        Same structure as extract_entities(), with entities merged across all chunks
        and usage counts summed.
    """
    # Try sentence-boundary chunking via StructSense utilities + spaCy
    use_spacy = False
    raw_chunks: list[dict] = []
    full_doc = None
    try:
        import spacy
        from utils.text_chunking import _chunk_doc_by_sentences, _globalize_entities

        nlp = spacy.load("en_core_web_sm")
        full_doc = nlp(text)
        raw_chunks = _chunk_doc_by_sentences(full_doc, chunk_size)
        use_spacy = True
        logger.info("extract_entities_chunked: using sentence-boundary chunking (spaCy)")
    except Exception as e:
        logger.warning("extract_entities_chunked: spaCy unavailable (%s) — falling back to paragraph chunking", e)
        raw_chunks = [{"text": c, "start": 0} for c in _chunk_text_by_paragraphs(text, chunk_size)]

    logger.info("extract_entities_chunked: %d chunks (chunk_size=%d)", len(raw_chunks), chunk_size)

    all_entities: list[dict[str, Any]] = []
    merged_usage: dict[str, Any] = {}

    for i, chunk in enumerate(raw_chunks):
        chunk_text = chunk["text"]
        logger.info("  chunk %d/%d  len=%d chars", i + 1, len(raw_chunks), len(chunk_text))
        result = extract_entities(text=chunk_text, model=model, **kwargs)

        if use_spacy and full_doc is not None:
            # Globalise offsets and attach sentence context from the full document
            from utils.text_chunking import _globalize_entities
            entities = _globalize_entities(text, full_doc, chunk, result["entities"])
        else:
            entities = result["entities"]

        all_entities.extend(entities)
        for k, v in result.get("usage", {}).items():
            if isinstance(v, (int, float)):
                merged_usage[k] = merged_usage.get(k, 0) + v

    logger.info("extract_entities_chunked: %d total entities across %d chunks", len(all_entities), len(raw_chunks))
    return {
        "entities": all_entities,
        "model": model,
        "usage": merged_usage,
        "raw_response": f"[chunked: {len(raw_chunks)} chunks, {len(all_entities)} entities]",
        "chunk_count": len(raw_chunks),
    }


def _chunk_text_by_paragraphs(text: str, max_chars: int) -> list[str]:
    """Fallback: split text at paragraph boundaries when spaCy is unavailable."""
    import re

    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _parse_entities(raw: str) -> list[dict[str, Any]]:
    """Parse the JSON entity list from the model's raw response.

    Falls back to regex extraction of complete entity objects when the response
    is truncated (e.g. model hit output token limit mid-stream).
    """
    import re

    raw = raw.strip()
    # Strip markdown code fences if the model wrapped the JSON
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```"))

    try:
        data = json.loads(raw)
        return data.get("entities", [])
    except json.JSONDecodeError:
        pass

    # Recover complete entity objects from a truncated response
    entity_re = re.compile(r'\{[^{}]*"entity"\s*:\s*"[^"]*"[^{}]*\}')
    recovered = []
    for m in entity_re.finditer(raw):
        try:
            recovered.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    if recovered:
        logger.warning("direct_api: truncated response — recovered %d entities via regex", len(recovered))
        return recovered

    logger.warning("direct_api: failed to parse JSON response; returning empty entity list")
    logger.debug("raw response was: %r", raw)
    return []
