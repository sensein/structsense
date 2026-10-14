#!/usr/bin/env python3
"""
Runner for Layer 1 NER comparison: StructSense vs. direct API call.

Loads StructSense staged outputs, runs (or loads cached) direct API extraction,
then computes Layer 1A (pre-alignment) and Layer 1B (post-alignment) metrics.

Output files are written to:
    outputs/<paper_id>/<model>/chunking/<paper_id>_<model>_chunking_direct_api.json
    outputs/<paper_id>/<model>/chunking/<paper_id>_<model>_chunking_layer1.json
    outputs/<paper_id>/<model>/no_chunking/<paper_id>_<model>_no_chunking_direct_api.json
    outputs/<paper_id>/<model>/no_chunking/<paper_id>_<model>_no_chunking_layer1.json

These directories are created automatically. --api-cache and --output override
the defaults if explicit paths are needed.

Usage:
    # With a pre-extracted text file:
    python run_evaluation.py \\
        --text input.txt \\
        --staged-dir path/to/staged_nhil \\
        --model openrouter/google/gemini-2.0-flash-001 \\
        [--paper-id my_paper]

    # With a PDF (requires a running Grobid server):
    python run_evaluation.py \\
        --pdf paper.pdf \\
        --staged-dir path/to/staged_nhil \\
        --model openrouter/google/gemini-2.0-flash-001 \\
        [--grobid-url http://localhost:8070] \\
        [--paper-id my_paper]

Inputs:
    --text / --pdf  Input text: either a plain-text file or a PDF extracted via Grobid.
                    Exactly one must be provided.
    --staged-dir    Directory containing StructSense staged outputs:
                      00_extractor_agent_extraction_task.json  (Layer 1A)
                      02_judge_agent_judge_task.json           (Layer 1B)
    --model         LiteLLM model string, e.g. "openrouter/google/gemini-2.0-flash-001".
    --grobid-url    Grobid server URL (default: $GROBID_SERVER_URL_OR_EXTERNAL_SERVICE
                    or http://localhost:8070). Only used with --pdf.
    --paper-id      Identifier string for this paper (defaults to staged-dir parent name).
    --api-cache     Override default path for the direct API JSON cache.
    --output        Override default path for the metrics JSON output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve sibling modules
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_OUTPUTS_ROOT = _HERE / "outputs"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "ner" / "analysis"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from direct_api import extract_entities, extract_entities_chunked  # noqa: E402
from layer1_metrics import compute_layer1a, compute_layer1b  # noqa: E402
from utils.utils import _structured_data_to_text, extract_pdf_content  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_slug(model: str) -> str:
    """Return a filesystem-safe name for a model string.

    Takes the last path component, e.g. 'openrouter/openai/gpt-4o' -> 'gpt-4o'.
    """
    return model.split("/")[-1]


def _resolve_output_dir(paper_id: str, model: str, chunked: bool) -> Path:
    """Return outputs/<paper_id>/<model_slug>/<chunking|no_chunking>/, creating it if needed."""
    out_dir = _OUTPUTS_ROOT / paper_id / _model_slug(model) / ("chunking" if chunked else "no_chunking")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_entities(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    entities = data.get("entities")
    if entities is None:
        raise ValueError(f"No 'entities' key found in {path}")
    return entities


def _print_layer1a(r) -> None:
    print("\n── Layer 1A  (pre-alignment: extractor vs. direct API) ──────────────────")
    print(f"  Paper:              {r.paper_id}")
    print(f"  Jaccard overlap:    {r.jaccard:.3f}  ({r.shared_count} shared spans)")
    print(f"  Label agreement:    {r.label_agreement_rate:.3f}  ({len(r.label_disagreements)} disagreements)")
    print(f"  StructSense-only:   {len(r.structsense_only)} spans")
    print(f"  API-only:           {len(r.api_only)} spans")
    if r.label_disagreements:
        print("  Top 10 label disagreements:")
        for d in r.label_disagreements[:10]:
            print(f"    '{d['entity_normalized']}' [{d['sentence_fingerprint']}...] — SS:{d['structsense_canonical']} vs API:{d['api_canonical']}")


def _print_layer1b(r) -> None:
    print("\n── Layer 1B  (post-alignment: judge-filtered vs. direct API) ────────────")
    print(f"  Paper:              {r.paper_id}")
    print(f"  Jaccard (high-conf):{r.jaccard_high_conf:.3f}  ({r.shared_count} shared spans)")
    print(f"  Label agreement:    {r.label_agreement_rate:.3f}  ({len(r.label_disagreements)} disagreements)")
    print(f"  StructSense-only:   {len(r.structsense_only)} spans")
    print(f"  API-only:           {len(r.api_only)} spans")
    print(f"  Low-conf filtered:  {r.low_conf_count}/{r.total_structsense_count}  ({r.low_conf_rate:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 1 NER evaluation: StructSense vs. direct API")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=Path, help="Plain-text input file")
    text_group.add_argument("--pdf", type=Path, help="PDF input file (extracted via Grobid)")
    parser.add_argument("--grobid-url", default=None, help="Grobid server URL (default: $GROBID_SERVER_URL_OR_EXTERNAL_SERVICE or http://localhost:8070)")
    parser.add_argument("--staged-dir", required=True, type=Path, help="Path to StructSense staged_nhil directory")
    parser.add_argument("--model", required=True, help="LiteLLM model string")
    parser.add_argument("--paper-id", default=None, help="Paper identifier (defaults to staged-dir parent name)")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens for the direct API call (default: 16384)")
    parser.add_argument("--chunk", action="store_true", default=False, help="Split text into chunks before calling the API (improves recall on long documents)")
    parser.add_argument("--chunk-size", type=int, default=30000, help="Max characters per chunk when --chunk is enabled (default: 30000)")
    parser.add_argument("--api-cache", default=None, type=Path, help="Override default path for the direct API JSON cache")
    parser.add_argument("--output", "-o", default=None, type=Path, help="Override default path for the metrics JSON output")
    args = parser.parse_args()

    # Validate inputs
    if args.text and not args.text.is_file():
        parser.error(f"--text file not found: {args.text}")
    if args.pdf and not args.pdf.is_file():
        parser.error(f"--pdf file not found: {args.pdf}")
    if not args.staged_dir.is_dir():
        parser.error(f"--staged-dir not found: {args.staged_dir}")

    extractor_file = args.staged_dir / "00_extractor_agent_extraction_task.json"
    judge_file = args.staged_dir / "02_judge_agent_judge_task.json"
    for f in (extractor_file, judge_file):
        if not f.is_file():
            parser.error(f"Required staged file not found: {f}")

    paper_id = args.paper_id or args.staged_dir.parent.name
    out_dir = _resolve_output_dir(paper_id, args.model, args.chunk)
    chunk_label = "chunking" if args.chunk else "no_chunking"
    stem = f"{paper_id}_{_model_slug(args.model)}_{chunk_label}"
    api_cache: Path = args.api_cache or (out_dir / f"{stem}_direct_api.json")
    output_path: Path = args.output or (out_dir / f"{stem}_layer1.json")
    logger.info("Output directory: %s", out_dir)

    # Load StructSense entities
    logger.info("Loading extractor stage output: %s", extractor_file)
    ss_extractor_entities = _load_entities(extractor_file)
    logger.info("  %d entities", len(ss_extractor_entities))

    logger.info("Loading judge stage output: %s", judge_file)
    ss_judge_entities = _load_entities(judge_file)
    logger.info("  %d entities", len(ss_judge_entities))

    # Direct API — use cache if available
    if api_cache.is_file():
        logger.info("Loading cached direct API output: %s", api_cache)
        api_result = json.loads(api_cache.read_text())
    else:
        if args.text:
            text = args.text.read_text(encoding="utf-8")
        else:
            import os
            grobid_url = args.grobid_url or os.environ.get("GROBID_SERVER_URL_OR_EXTERNAL_SERVICE", "http://localhost:8070")
            logger.info("Extracting PDF via Grobid: %s  server=%s", args.pdf, grobid_url)
            raw = extract_pdf_content(file_path=str(args.pdf), grobid_server=grobid_url, external_service="false")
            text = _structured_data_to_text(raw)
            logger.info("Extracted %d chars from PDF", len(text))
        if args.chunk:
            logger.info("Calling direct API (chunked)  model=%s  text_len=%d chars  chunk_size=%d  max_tokens=%d",
                        args.model, len(text), args.chunk_size, args.max_tokens)
            api_result = extract_entities_chunked(text=text, model=args.model, chunk_size=args.chunk_size, max_tokens=args.max_tokens)
        else:
            logger.info("Calling direct API  model=%s  text_len=%d chars  max_tokens=%d", args.model, len(text), args.max_tokens)
            api_result = extract_entities(text=text, model=args.model, max_tokens=args.max_tokens)
        logger.info("  %d entities extracted", len(api_result["entities"]))
        api_cache.write_text(json.dumps(api_result, indent=2))
        logger.info("Direct API output cached to: %s", api_cache)

    api_entities = api_result["entities"]

    # Compute metrics
    r1a = compute_layer1a(paper_id, ss_extractor_entities, api_entities)
    r1b = compute_layer1b(paper_id, ss_judge_entities, api_entities)

    # Print
    _print_layer1a(r1a)
    _print_layer1b(r1b)
    print()

    # Save
    from dataclasses import asdict
    results = {
        "paper_id": paper_id,
        "model": args.model,
        "layer1a": asdict(r1a),
        "layer1b": asdict(r1b),
        "direct_api_entity_count": len(api_entities),
        "direct_api_model": api_result.get("model"),
        "direct_api_usage": api_result.get("usage"),
    }
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
