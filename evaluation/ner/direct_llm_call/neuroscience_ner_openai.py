#!/usr/bin/env python3
"""Run neuroscience-wide NER on an uploaded file via an OpenAI model.

Usage:
    python neuroscience_ner_openai.py --file paper.pdf --model gpt-5.5
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Default system/extractor prompt shipped alongside this script. Override with
# --prompt-file to test alternative prompts.
DEFAULT_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompts",
    "extractor_neuroscience_ner.txt",
)

USER_PROMPT = """\
INPUT TEXT:
The text to process is the attached file. Treat the full extracted text of
the attached document as the INPUT TEXT.

METADATA (paper_title / doi / source_path) — populate `source_metadata` from
this; do NOT repeat on every entity:
{metadata_json}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run neuroscience-wide NER on an uploaded file via an OpenAI model."
    )
    parser.add_argument("--file", "-f", required=True, help="Path to the file to upload.")
    parser.add_argument("--model", "-m", required=True, help="Model name, e.g. gpt-5.5.")
    parser.add_argument(
        "--prompt-file",
        "-p",
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to the system/extractor prompt file. Default: {DEFAULT_PROMPT_FILE}",
    )
    parser.add_argument(
        "--metadata",
        default="{}",
        help='JSON string with paper_title / doi / source_path. Default: "{}".',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to write the output JSON file into. Default: current dir.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print step-by-step progress and streaming updates. Default: quiet.",
    )
    args = parser.parse_args()

    def vprint(*a, **k):
        """Print only when --verbose is set."""
        if args.verbose:
            print(*a, **k)

    # Parse the metadata once. We both pass it to the model (for context) and
    # stamp it authoritatively into the saved JSON below, so the output's
    # source_metadata is correct regardless of what the model echoes.
    try:
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise ValueError("--metadata must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid --metadata: {exc}")

    # Default source_path to the input filename when not explicitly provided.
    metadata.setdefault("source_path", args.file)

    vprint(f"Loading system prompt from: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as fh:
        system_prompt = fh.read()

    vprint("Initializing OpenAI client...")
    # Generous read timeout for very long generations; fail rather than hang forever.
    client = OpenAI(timeout=1800.0, max_retries=2)

    vprint(f"Uploading file: {args.file} ...")
    with open(args.file, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="user_data")
    vprint(f"  uploaded (file_id={uploaded.id})")

    vprint(f"Sending request to model '{args.model}' (streaming; this may take a while)...")
    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": uploaded.id},
                {
                    "type": "input_text",
                    "text": USER_PROMPT.format(metadata_json=json.dumps(metadata)),
                },
            ],
        }
    ]

    chunks = []
    chars = 0
    next_report = 2000  # print a progress line every ~2000 chars
    with client.responses.stream(
        model=args.model,
        instructions=system_prompt,
        input=request_input,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
                chars += len(event.delta)
                if chars >= next_report:
                    vprint(f"  ...streaming, {chars} chars received so far")
                    next_report += 2000
            elif event.type == "error":
                print(f"  stream error: {event.error}")
        stream.get_final_response()  # surfaces any terminal API error

    output_text = "".join(chunks)
    vprint(f"  response complete — {chars} chars total.")

    # Build a filesystem-safe filename with timestamp and model.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out_name = f"neuroscience_ner_{safe_model}_{timestamp}.json"
    out_path = os.path.join(args.output_dir, out_name)

    # Persist parsed JSON when possible; otherwise wrap the raw text.
    vprint("Parsing model output as JSON...")
    try:
        payload = json.loads(output_text)
        n_entities = len(payload.get("entities", [])) if isinstance(payload, dict) else 0
        n_terms = len(payload.get("key_terms", [])) if isinstance(payload, dict) else 0
        vprint(f"  parsed OK — {n_entities} entities, {n_terms} key_terms.")
    except json.JSONDecodeError:
        print("  output was not valid JSON — wrapping raw text under 'raw_output'.")
        payload = {"raw_output": output_text}

    # Stamp metadata authoritatively, overriding whatever the model echoed.
    if isinstance(payload, dict):
        payload["source_metadata"] = metadata
        vprint(f"  stamped source_metadata: {metadata}")

        # Compute extraction statistics and stamp them into the metadata.
        entities = payload.get("entities", []) or []
        label_counts = {}
        for ent in entities:
            if isinstance(ent, dict):
                label = ent.get("label", "Unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
        stats = {
            "model": args.model,
            "extracted_at": timestamp,
            "total_entities": len(entities),
            "entities_by_label": dict(
                sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
            ),
        }
        payload["source_metadata"]["statistics"] = stats
        print(
            f"Extracted {stats['total_entities']} entities "
            f"across {len(label_counts)} labels."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    vprint(f"Writing output to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Done. Wrote output to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
