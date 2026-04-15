"""Model family context-window registry and token-aware downstream chunk sizing.

Patterns are checked in order from most-specific (longest) to least-specific so
the first match wins.  All sizes are in *tokens* (not characters).

Sources
-------
- https://codingscape.com/blog/most-powerful-llms-large-language-models
- https://explodingtopics.com/blog/list-of-llms
- Official provider documentation (OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek)
"""

import json
import math
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter live context-window cache
# Populated by fetch_and_cache_openrouter_context_window() at pipeline startup.
# get_model_context_window() checks this cache before falling back to static patterns.
# ---------------------------------------------------------------------------
_openrouter_context_cache: Dict[str, int] = {}

# Regex to parse the actual context limit out of OpenRouter 400 error messages:
#   "This endpoint's maximum context length is 1048576 tokens.
#    However, you requested about 1618708 tokens (1618708 of text input)."
_CTX_ERROR_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?requested about (\d+) tokens",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_openrouter_model_id(model_str: str) -> str:
    """Convert a full model config string to a bare OpenRouter model ID.

    Examples
    --------
    ``"openrouter/google/gemini-3.1-flash-lite-preview:nitro"``
      → ``"google/gemini-3.1-flash-lite-preview"``
    ``"openai/gpt-4o-mini"``
      → ``"openai/gpt-4o-mini"``
    """
    m = (model_str or "").strip()
    if m.lower().startswith("openrouter/"):
        m = m[len("openrouter/") :]
    # Strip ":variant" suffixes like ":nitro", ":extended", ":free"
    if ":" in m:
        m = m.split(":")[0]
    return m


def fetch_and_cache_openrouter_context_window(
    model_str: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> Optional[int]:
    """Fetch the real context-window size for *model_str* from OpenRouter's
    ``GET /api/v1/models`` endpoint and store it in ``_openrouter_context_cache``.

    Only runs when *base_url* points to OpenRouter and *api_key* is set.
    Results are cached by normalised model ID so subsequent calls are free.
    Returns the context window (tokens) or ``None`` if the fetch fails or the
    model is not found.

    Parameters
    ----------
    model_str:
        Full model identifier as it appears in the YAML config, e.g.
        ``"openrouter/google/gemini-3.1-flash-lite-preview:nitro"``.
    api_key:
        OpenRouter API key (``OPENROUTER_API_KEY``).
    base_url:
        OpenRouter base URL (default ``https://openrouter.ai/api/v1``).
    """
    if not api_key or "openrouter" not in (base_url or "").lower():
        return None

    normalized = _normalize_openrouter_model_id(model_str)
    if not normalized:
        return None

    if normalized in _openrouter_context_cache:
        logger.debug(
            "[openrouter] Context window for '%s' already cached: %d",
            normalized,
            _openrouter_context_cache[normalized],
        )
        return _openrouter_context_cache[normalized]

    try:
        import requests as _req

        url = base_url.rstrip("/") + "/models"
        resp = _req.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])

        # Exact match first, then prefix match for :variant stripped IDs
        matched_ctx: Optional[int] = None
        for entry in models:
            entry_id = (entry.get("id") or "").strip()
            entry_normalized = _normalize_openrouter_model_id(entry_id)
            if entry_normalized.lower() == normalized.lower():
                ctx = entry.get("context_length")
                if ctx:
                    matched_ctx = int(ctx)
                    break

        if matched_ctx:
            _openrouter_context_cache[normalized] = matched_ctx
            logger.info(
                "[openrouter] Fetched context window for '%s' (%s): %d tokens",
                model_str,
                normalized,
                matched_ctx,
            )
            return matched_ctx

        logger.debug(
            "[openrouter] Model '%s' not found in /models response (%d entries)",
            normalized,
            len(models),
        )

    except Exception as exc:
        logger.debug(
            "[openrouter] Failed to fetch model info for '%s': %s",
            model_str,
            exc,
        )

    return None


def parse_context_length_error(error_str: str) -> Optional[Tuple[int, int]]:
    """Extract context limits from an OpenRouter 400 context-length error string.

    Parses messages of the form:
    ``"This endpoint's maximum context length is 1048576 tokens. However,
    you requested about 1618708 tokens (1618708 of text input)."``

    Returns
    -------
    ``(max_context_tokens, requested_tokens)`` or ``None`` if the string does
    not match the expected pattern.
    """
    m = _CTX_ERROR_RE.search(str(error_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def is_context_length_error(error_str: str) -> bool:
    """Return ``True`` if *error_str* is an OpenRouter context-length 400 error."""
    return parse_context_length_error(error_str) is not None


def update_context_cache_from_error(model_str: str, error_str: str) -> Optional[int]:
    """Parse a context-length error and cache the real limit for *model_str*.

    Call this whenever a downstream agent call fails with a 400 context error
    so subsequent chunk-size calculations use the authoritative limit.

    Returns the parsed ``max_context_tokens`` or ``None``.
    """
    parsed = parse_context_length_error(error_str)
    if not parsed:
        return None
    max_ctx, requested = parsed
    normalized = _normalize_openrouter_model_id(model_str)
    _openrouter_context_cache[normalized] = max_ctx
    logger.info(
        "[openrouter] Learned context window from error for '%s': %d tokens " "(request was %d tokens)",
        model_str,
        max_ctx,
        requested,
    )
    return max_ctx


# ---------------------------------------------------------------------------
# Fallback prompt overhead when the actual agent/task config is not available.
# Covers role + goal + backstory + task description + CrewAI framework.
# The actual value is computed adaptively in estimate_agent_prompt_tokens().
# ---------------------------------------------------------------------------
_PROMPT_OVERHEAD_TOKENS: int = 10_000

# Fixed token budget for CrewAI's ReAct framework boilerplate that is injected
# into every LLM call regardless of YAML config content:
#   - ReAct thought/action/observation template
#   - JSON output-schema reminder and format instructions
#   - Tool-call wrapper (even when the agent has no tools)
# Measured empirically at ~1.5 k–5 k tokens.  3 k is a conservative estimate.
_CREWAI_FRAMEWORK_OVERHEAD_TOKENS: int = 3_000

# Safety factor: the model needs room to *generate* a structured JSON response.
# We only use this fraction of (context − prompt_overhead) for payload data.
# 0.70 means 30 % of the remaining window is reserved for the output.
_CONTEXT_SAFETY_FACTOR: float = 0.70

# ---------------------------------------------------------------------------
# Model family → context window (tokens).
# Ordered from MOST SPECIFIC to LEAST SPECIFIC so the first substring match
# wins.  Always keep more-specific patterns (e.g. "llama-4-scout") before
# their parent family (e.g. "llama-4", "llama").
# ---------------------------------------------------------------------------
_MODEL_CONTEXT_PATTERNS: list = [
    # ── Meta Llama 4 ──────────────────────────────────────────────────────
    ("llama-4-scout", 10_000_000),  # Llama 4 Scout: 10M context
    ("llama-4-maverick", 1_000_000),  # Llama 4 Maverick: 1M
    ("llama-4", 1_000_000),  # Llama 4 generic
    # Llama 3.x
    ("llama-3.3", 128_000),
    ("llama-3.2", 128_000),
    ("llama-3.1", 128_000),
    ("llama-3", 8_000),
    ("llama", 128_000),  # fallback for any Llama
    # ── OpenAI GPT ────────────────────────────────────────────────────────
    ("gpt-4.1", 1_000_000),  # GPT-4.1: 1M
    ("gpt-5-mini", 400_000),
    ("gpt-5-nano", 400_000),
    ("gpt-5", 400_000),
    ("gpt-4o-mini", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 128_000),
    ("gpt-3.5-turbo-16k", 16_000),
    ("gpt-3.5", 4_000),
    ("o3-mini", 200_000),
    ("o3", 200_000),
    ("o1-mini", 128_000),
    ("o1", 128_000),
    # ── Anthropic Claude ──────────────────────────────────────────────────
    # All Claude 3+ models: 200 k (1 M in beta)
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-5-haiku", 200_000),
    ("claude-3-5", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-3", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude", 200_000),  # fallback for any Claude
    # ── Google Gemini ─────────────────────────────────────────────────────
    ("gemini-2.5", 1_000_000),
    ("gemini-2.0", 1_000_000),
    ("gemini-2", 1_000_000),
    ("gemini-1.5", 1_000_000),
    ("gemini-1.0", 32_000),
    ("gemini-pro-1", 32_000),
    ("gemini", 1_000_000),  # fallback for any Gemini
    # ── DeepSeek ──────────────────────────────────────────────────────────
    ("deepseek-r1", 128_000),
    ("deepseek-v3", 128_000),
    ("deepseek-v2", 128_000),
    ("deepseek-coder", 128_000),
    ("deepseek", 128_000),  # fallback
    # ── Mistral ───────────────────────────────────────────────────────────
    ("mistral-large-3", 256_000),
    ("mistral-large-2", 128_000),
    ("mistral-large", 128_000),
    ("mistral-medium-3", 131_000),
    ("mistral-medium", 128_000),
    ("mistral-small", 128_000),
    ("mistral-7b", 32_000),
    ("mixtral-8x22b", 65_000),
    ("mixtral-8x7b", 32_000),
    ("mixtral", 32_000),
    ("ministral", 256_000),
    ("mistral", 128_000),  # fallback
    # ── Qwen (Alibaba) ────────────────────────────────────────────────────
    ("qwen3-coder", 256_000),
    ("qwen3-235b", 256_000),
    ("qwen3-32b", 128_000),
    ("qwen3", 256_000),
    ("qwen2.5-72b", 128_000),
    ("qwen2.5", 128_000),
    ("qwen2", 128_000),
    ("qwen-long", 1_000_000),
    ("qwen", 128_000),  # fallback
    # ── Nvidia Nemotron ───────────────────────────────────────────────────
    ("nemotron-3-ultra", 1_000_000),
    ("nemotron-3-super", 1_000_000),
    ("nemotron-3-nano", 1_000_000),
    ("nemotron", 1_000_000),  # fallback
    # ── xAI Grok ─────────────────────────────────────────────────────────
    ("grok-3", 131_000),
    ("grok-2", 131_000),
    ("grok-1.5", 131_000),
    ("grok", 131_000),
    # ── Cohere Command ────────────────────────────────────────────────────
    ("command-r-plus", 128_000),
    ("command-r", 128_000),
    ("command", 4_000),
    # ── Amazon Nova / Titan ───────────────────────────────────────────────
    ("nova-pro", 300_000),
    ("nova-lite", 300_000),
    ("nova-micro", 128_000),
    ("nova", 300_000),
    ("titan", 32_000),
    # ── Microsoft Phi ─────────────────────────────────────────────────────
    ("phi-4", 16_000),
    ("phi-3.5", 128_000),
    ("phi-3", 128_000),
    ("phi", 128_000),
    # ── Yi (01.AI) ────────────────────────────────────────────────────────
    ("yi-large", 200_000),
    ("yi-34b", 32_000),
    ("yi", 32_000),
    # ── Falcon ────────────────────────────────────────────────────────────
    ("falcon", 8_000),
    # ── Perplexity / Sonar ────────────────────────────────────────────────
    ("sonar-pro", 127_000),
    ("sonar", 127_000),
    # ── WizardLM / Vicuna / misc open-source ─────────────────────────────
    ("wizardlm", 32_000),
    ("vicuna", 4_000),
]

# Default used when no pattern matches
_DEFAULT_CONTEXT_WINDOW: int = 128_000


def get_model_context_window(model_str: str) -> int:
    """Return the context-window size (tokens) for *model_str*.

    Checks the live OpenRouter probe cache first (populated by
    ``probe_openrouter_context_window``), then falls back to the static
    ``_MODEL_CONTEXT_PATTERNS`` list, and finally to ``_DEFAULT_CONTEXT_WINDOW``.

    Parameters
    ----------
    model_str:
        Full model identifier as it appears in the YAML config or environment,
        e.g. ``"openrouter/deepseek/deepseek-chat-v3-0324"`` or
        ``"openai/gpt-4o-mini"``.  Case-insensitive.
    """
    if not model_str:
        return _DEFAULT_CONTEXT_WINDOW

    # Check live probe cache first (populated by probe_openrouter_context_window).
    normalized = _normalize_openrouter_model_id(model_str)
    if normalized in _openrouter_context_cache:
        cached = _openrouter_context_cache[normalized]
        logger.debug(
            "Model context window: cache hit for '%s' (%s) → %d tokens",
            model_str,
            normalized,
            cached,
        )
        return cached

    m = model_str.lower()
    for pattern, ctx in _MODEL_CONTEXT_PATTERNS:
        if pattern in m:
            logger.debug(
                "Model context window: matched '%s' → pattern='%s' → %d tokens",
                model_str,
                pattern,
                ctx,
            )
            return ctx

    logger.debug(
        "Model context window: no pattern matched '%s', using default %d",
        model_str,
        _DEFAULT_CONTEXT_WINDOW,
    )
    return _DEFAULT_CONTEXT_WINDOW


def probe_openrouter_context_window(
    model_str: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    *,
    oversize_factor: float = 1.10,
) -> int:
    """Probe the real context-window size for *model_str* via an intentional
    oversized request to OpenRouter.

    Strategy
    --------
    1. Look up the static dictionary estimate: ``static_ctx = _static_context_window(model_str)``.
    2. Build a dummy chat-completion payload whose token count is
       ``static_ctx × oversize_factor`` (10 % over by default).
    3. POST to ``{base_url}/chat/completions``.  OpenRouter returns a 400 error
       whose message contains the real maximum::

           "This endpoint's maximum context length is 1048576 tokens.
            However, you requested about 1152000 tokens …"

    4. Parse the error → cache the real limit → return it.
    5. If the model actually accepts the oversized request (context is larger
       than the static estimate) **or** the probe fails for any other reason,
       fall back to the static estimate.

    The result is stored in ``_openrouter_context_cache`` so subsequent calls
    to ``get_model_context_window`` return the authoritative value without
    making another HTTP request.

    Parameters
    ----------
    model_str:
        Full model identifier, e.g.
        ``"openrouter/google/gemini-3.1-flash-lite-preview:nitro"``.
    api_key:
        OpenRouter API key (``OPENROUTER_API_KEY``).
    base_url:
        OpenRouter base URL (default ``https://openrouter.ai/api/v1``).
    oversize_factor:
        Multiplier applied to the static context estimate to build the dummy
        payload.  Must be > 1.0 so the request is guaranteed to exceed the
        static estimate.  Default 1.10 (10 % over).

    Returns
    -------
    int
        Confirmed (or static fallback) context-window size in tokens.
    """
    if not api_key or not model_str:
        return get_model_context_window(model_str)

    normalized = _normalize_openrouter_model_id(model_str)

    # Return cached value immediately — no network call needed.
    if normalized in _openrouter_context_cache:
        return _openrouter_context_cache[normalized]

    # Static estimate from the built-in dictionary (bypass cache to avoid recursion).
    static_ctx = _static_context_window(model_str)

    # Build a dummy user message whose token count exceeds static_ctx × factor.
    # 4 chars ≈ 1 token (conservative GPT-style estimate).
    target_tokens = int(static_ctx * max(oversize_factor, 1.01))
    dummy_chars = target_tokens * 4
    # Use a short repeating phrase to avoid triggering content filters.
    filler = "context probe " * (dummy_chars // 14 + 1)
    dummy_content = filler[:dummy_chars]

    payload = {
        "model": normalized,
        "messages": [{"role": "user", "content": dummy_content}],
        "max_tokens": 1,
    }

    try:
        import requests as _req

        url = base_url.rstrip("/") + "/chat/completions"
        resp = _req.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15.0,
        )

        if resp.status_code == 400:
            error_text = ""
            try:
                error_text = resp.json().get("error", {}).get("message", "") or resp.text
            except Exception:
                error_text = resp.text or ""

            parsed = parse_context_length_error(error_text)
            if parsed:
                real_ctx, requested = parsed
                _openrouter_context_cache[normalized] = real_ctx
                logger.info(
                    "[probe] Confirmed context window for '%s' (%s): %d tokens " "(static estimate was %d, probe sent %d tokens)",
                    model_str,
                    normalized,
                    real_ctx,
                    static_ctx,
                    requested,
                )
                return real_ctx

            # 400 but not a context-length error — fall through to static.
            logger.debug(
                "[probe] 400 response for '%s' but no context-length pattern: %s",
                model_str,
                error_text[:200],
            )

        elif resp.status_code == 200:
            # The model accepted more tokens than the static estimate → cache
            # the oversized target as a confirmed lower bound.
            _openrouter_context_cache[normalized] = target_tokens
            logger.info(
                "[probe] Model '%s' accepted %d-token probe (static est. %d). " "Using %d as confirmed lower bound.",
                model_str,
                target_tokens,
                static_ctx,
                target_tokens,
            )
            return target_tokens

        else:
            logger.debug(
                "[probe] Unexpected status %d for '%s', falling back to static %d",
                resp.status_code,
                model_str,
                static_ctx,
            )

    except Exception as exc:
        logger.debug(
            "[probe] HTTP probe failed for '%s': %s — using static estimate %d",
            model_str,
            exc,
            static_ctx,
        )

    return static_ctx


def _static_context_window(model_str: str) -> int:
    """Return the static dictionary estimate for *model_str* without checking
    the live cache.  Used internally by the probe function to avoid recursion.
    """
    if not model_str:
        return _DEFAULT_CONTEXT_WINDOW
    m = model_str.lower()
    for pattern, ctx in _MODEL_CONTEXT_PATTERNS:
        if pattern in m:
            return ctx
    return _DEFAULT_CONTEXT_WINDOW


def estimate_payload_tokens(payload: Any) -> int:
    """Estimate token count for *payload*.

    Uses the character count of the JSON-serialised form as the token estimate.

    Why not the classic 4 chars-per-token rule?
    -------------------------------------------
    That rule-of-thumb applies to flowing natural-language text.  For
    structured JSON data — entities with ontology IDs, sentence offsets,
    label strings, nested arrays — every delimiter character (``{``, ``}``\,
    ``"``, ``:``, ``,``, ``[``, ``]``) is its own token.  Empirically, a
    2 130-entity aligned payload serialising to ~1.62 M characters was counted
    as ~1.62 M tokens by OpenRouter/Gemini, confirming the 1:1 approximation
    for this data shape.

    Using 1 char ≈ 1 token is deliberately conservative: it may slightly
    overestimate for text-heavy content (triggering extra chunking that is
    harmless) but will never under-estimate and cause context-overflow errors.
    """
    try:
        return max(1, len(json.dumps(payload, default=str)))
    except Exception:
        return 1


def estimate_agent_prompt_tokens(
    agent_config: Dict[str, Any],
    task_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Estimate the prompt overhead (tokens) for a single downstream agent call.

    Reads the *actual* YAML config content so the estimate adapts to the user's
    specific role/goal/backstory/task description — which can vary enormously
    across configs (a minimal config may be 1 k tokens; a richly annotated one
    with many entity examples may exceed 50 k).

    Parameters
    ----------
    agent_config:
        The agent sub-dict from the loaded YAML, e.g.
        ``self.agent_config.get("judge_agent", {})``.  Expected fields:
        ``role``, ``goal``, ``backstory``.
    task_config:
        The task sub-dict, e.g.
        ``self.task_config.get("judge_task", {})``.  Expected fields:
        ``description``, ``expected_output``.  Template placeholders
        (``{aligned_structured_information}``) are stripped so the payload
        itself is not double-counted.

    Returns
    -------
    int
        Conservative token estimate: config text tokens +
        ``_CREWAI_FRAMEWORK_OVERHEAD_TOKENS`` for CrewAI boilerplate.
    """
    text_parts: list = []

    # Agent identity fields (role, goal, backstory)
    for field in ("role", "goal", "backstory"):
        val = agent_config.get(field, "")
        if isinstance(val, str) and val:
            text_parts.append(val)

    if task_config:
        # Task description template and expected output format.
        # Strip {placeholder} substitution markers so the payload JSON that
        # gets injected at runtime is not counted here (it's tracked separately
        # in the payload estimate).
        for field in ("description", "expected_output"):
            val = task_config.get(field, "")
            if isinstance(val, str) and val:
                # Remove {variable_name} placeholders
                cleaned = re.sub(r"\{[^}]+\}", "", val)
                text_parts.append(cleaned)

    # Concatenate all config text and estimate its token count.
    # Config text is mostly natural language, but 1 char/token is still
    # a safe upper bound (better to over-estimate and chunk more than under-).
    combined = " ".join(text_parts)
    config_tokens = max(1, len(combined))

    # Add the fixed CrewAI framework overhead on top.
    return config_tokens + _CREWAI_FRAMEWORK_OVERHEAD_TOKENS


def compute_downstream_chunk_size(
    payload: Dict[str, Any],
    model_str: str,
    max_workers: int,
    extraction_chunk_count: Optional[int] = None,
    explicit_chunk_size: Optional[int] = None,
    context_window_override: Optional[int] = None,
    prompt_overhead_tokens: Optional[int] = None,
) -> Tuple[int, bool]:
    """Compute the optimal entities-per-chunk for a downstream stage.

    Token budget formula
    --------------------
    The LLM context window is shared between four consumers:

        context_window = prompt_overhead + payload + generated_output + safety

    We allocate budgets as follows::

        input_budget    = context_window × safety_factor   (0.70)
        payload_budget  = input_budget − prompt_overhead

    If ``payload_tokens ≤ payload_budget`` → single call (no chunking).
    Otherwise → split into ``min_chunks = ceil(payload / budget)`` chunks,
    capped at ``max_workers``.

    The *prompt_overhead* should be the **adaptive** estimate from
    ``estimate_agent_prompt_tokens()`` — i.e., the actual character count of
    the agent's role/goal/backstory + task description template + the fixed
    CrewAI framework boilerplate.  This way the budget automatically shrinks
    when the user has a large, detailed config and expands for compact configs.

    Why the 1-char-per-token payload estimate?
    ------------------------------------------
    Structured JSON data (entities, ontology IDs, offsets, nested arrays) has
    dense tokenisation: every ``{``, ``"``, ``:``, ``,`` is one token.
    Empirically, a 2 130-entity payload of ~1.62 M characters was billed as
    ~1.62 M tokens by OpenRouter/Gemini.  Using ``len(json.dumps(payload))``
    directly avoids the 4× underestimate that caused context-overflow errors.

    Decision logic
    --------------
    1. **Explicit override** — if *explicit_chunk_size* is set, use it directly.
    2. **Fits in one call** — ``payload_tokens ≤ payload_budget`` → single call.
    3. **Must split** — use minimum chunks required, capped at ``max_workers``.

    Parameters
    ----------
    payload:
        The structured dict that will be serialised and injected into the
        agent task (e.g. ``{"entities": [...], "key_terms": [...]}``.
    model_str:
        Full model identifier from YAML config (e.g.
        ``"openrouter/google/gemini-3.1-flash-lite-preview:nitro"``).
    max_workers:
        Maximum parallel workers available.  Chunks are capped at this value.
    extraction_chunk_count:
        If the extraction stage ran in N chunks, use N as the target for
        downstream parallelism (so alignment/judge match extraction concurrency).
    explicit_chunk_size:
        Hard-coded entities-per-chunk override (e.g. from ``--downstream_chunk_size``).
        Bypasses all token math when set.
    context_window_override:
        User-supplied context window in tokens (e.g. ``--model_context_window``).
        Overrides both the static dictionary *and* the OpenRouter probe cache.
    prompt_overhead_tokens:
        Adaptive estimate of the agent's prompt (role + goal + backstory +
        task description template + CrewAI framework boilerplate).
        Computed by ``estimate_agent_prompt_tokens()`` in the calling code.
        Falls back to ``_PROMPT_OVERHEAD_TOKENS`` (10 k) when not supplied.

    Returns
    -------
    (entities_per_chunk, should_chunk)
        ``should_chunk=False`` → caller sends the whole payload in one agent
        run without splitting.
    """
    n_entities = len(payload.get("entities") or [])
    n_resources = len(payload.get("resources") or [])
    n_items = n_entities or n_resources

    if n_items == 0:
        return 70, False

    # ── 1. Explicit override ──────────────────────────────────────────────
    if explicit_chunk_size:
        return explicit_chunk_size, n_items > explicit_chunk_size

    # ── 2 & 3. Adaptive context-window–based decision ────────────────────
    # User override takes precedence over auto-detected / probed value.
    context_window = context_window_override or get_model_context_window(model_str)
    if context_window_override:
        logger.info(
            "[chunk_size] Using user-supplied context window: %d tokens (model='%s')",
            context_window_override,
            model_str,
        )

    # Prompt overhead: use the adaptive estimate when available, else fallback.
    # The adaptive estimate is computed from the *actual* agent config content
    # (role + goal + backstory + task description template + CrewAI boilerplate)
    # so it automatically adjusts to how large or small the user's YAML is.
    prompt_overhead = prompt_overhead_tokens if prompt_overhead_tokens else _PROMPT_OVERHEAD_TOKENS

    # payload_budget = input_budget − prompt_overhead
    #   input_budget  = context × safety_factor   (leaves room for generation)
    #   prompt_overhead = adaptive config estimate (or fallback 10 k)
    #
    # Historical note: the old formula (context − overhead) × 0.70 used a
    # hardcoded 10 k overhead, which was ~800× too small for large configs
    # (the actual CrewAI call for a 2130-entity judge stage sent 1.62 M tokens
    # while we estimated only 404 k — 4× wrong on the payload side AND ignoring
    # the actual prompt size).
    input_budget = context_window * _CONTEXT_SAFETY_FACTOR
    usable_tokens = max(1_000, input_budget - prompt_overhead)

    payload_tokens = estimate_payload_tokens(payload)

    if payload_tokens <= usable_tokens:
        # Everything fits → single call, maximise context utilisation.
        logger.info(
            "[chunk_size] model='%s' ctx=%d prompt_overhead=%d usable=%.0f " "payload=%d tokens → fits in one call, no chunking",
            model_str,
            context_window,
            prompt_overhead,
            usable_tokens,
            payload_tokens,
        )
        return n_items, False

    # Payload too large for one call.
    # Use the *minimum* number of chunks needed so each chunk fits within
    # usable_tokens.  Cap at max_workers (more chunks than workers gives no
    # speed benefit since they can't run in parallel anyway).
    min_chunks = math.ceil(payload_tokens / usable_tokens)
    target_workers = extraction_chunk_count if extraction_chunk_count else (max_workers or 4)
    n_chunks = min(min_chunks, target_workers)
    ecs = max(1, math.ceil(n_items / n_chunks))

    logger.info(
        "[chunk_size] model='%s' ctx=%d prompt_overhead=%d usable=%.0f "
        "payload=%d tokens n_items=%d → min_chunks=%d (capped at workers=%d) ecs=%d",
        model_str,
        context_window,
        prompt_overhead,
        usable_tokens,
        payload_tokens,
        n_items,
        min_chunks,
        target_workers,
        ecs,
    )
    return ecs, True
