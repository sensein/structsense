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
- Do not hallucinate entities that are not present in the text.
- Assign labels that reflect the entity's scientific role (e.g. BRAIN_REGION, CELL_TYPE, \
GENE, DISEASE, METHOD, CHEMICAL, SPECIES, SOFTWARE) — but use whatever label best fits.
- If an entity is genuinely ambiguous, pick the most specific label.

Return ONLY a JSON object in this exact format — no commentary, no markdown fences:
{
  "entities": [
    {"entity": "<surface form>", "label": "<YOUR_LABEL>", "start": <int>, "end": <int>},
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
          "entities": list of {entity, label, start, end}
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
