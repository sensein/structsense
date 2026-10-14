#!/usr/bin/env python3
"""Run neuroscience-wide NER on a local PDF via an OpenRouter model.

Uses the OpenAI SDK pointed at OpenRouter, sending the PDF as a base64 data
URL and configuring PDF parsing through the OpenRouter file-parser plugin.

Usage:
    python neuroscience_ner_openrouter.py --file paper.pdf --model openai/gpt-5.5
"""
import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GrobidClient:
    """
    Thin wrapper around the Grobid HTTP API for full-text PDF parsing.

    Grobid is expected to be running as a service (Docker image
    ``lfoppiano/grobid`` on port 8070 by default). This client POSTs a PDF
    to the ``processFulltextDocument`` endpoint and returns the resulting
    TEI XML string.

    Parameters
    ----------
    base_url:
        Root URL of the Grobid service, e.g. ``http://localhost:8070``.
    timeout:
        HTTP timeout in seconds. Large papers (>30 pages) can take 60+ seconds
        to parse so the default is set generously.
    consolidate_citations:
        Grobid flag — set to 0 (off), 1 (consolidate via CrossRef), or 2
        (consolidate against local biblio-glutton service).
    """

    FULLTEXT_ENDPOINT = "/api/processFulltextDocument"
    ISALIVE_ENDPOINT = "/api/isalive"

    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: float = 300.0,
        consolidate_citations: int = 0,
    ) -> None:
        try:
            import requests as requests_mod
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `requests` package is required for GrobidClient. "
                "Install with: uv add requests"
            ) from exc

        self._requests: Any = requests_mod
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.consolidate_citations = consolidate_citations

    def is_alive(self) -> bool:
        """Return True if the Grobid server responds at /api/isalive."""
        req = self._requests
        try:
            r = req.get(self.base_url + self.ISALIVE_ENDPOINT, timeout=5)
            return r.status_code == 200
        except req.RequestException:
            return False

    def process_fulltext(self, pdf_path: Path) -> str:
        """
        POST *pdf_path* to Grobid's full-text endpoint and return TEI XML.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF on disk.

        Returns
        -------
        str
            UTF-8 TEI XML string as returned by Grobid.

        Raises
        ------
        FileNotFoundError
            If *pdf_path* does not exist.
        RuntimeError
            If Grobid returns a non-2xx status, or the request fails.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        url = self.base_url + self.FULLTEXT_ENDPOINT
        req = self._requests
        try:
            with pdf_path.open("rb") as fh:
                resp = req.post(
                    url,
                    files={"input": (pdf_path.name, fh, "application/pdf")},
                    data={"consolidateCitations": str(self.consolidate_citations)},
                    timeout=self.timeout,
                )
        except req.RequestException as exc:
            raise RuntimeError(
                f"Grobid request to {url} failed: {exc}. "
                f"Is the Grobid server running and reachable?"
            ) from exc

        if not resp.ok:
            raise RuntimeError(
                f"Grobid returned HTTP {resp.status_code} for {pdf_path.name}: "
                f"{resp.text[:200]}"
            )
        return resp.text

# Optional attribution headers for openrouter.ai rankings.
EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/sensein/structsense",
    "X-Title": "structsense-ner-eval",
}

# Default system/extractor prompt shipped alongside this script. Override with
# --prompt-file to test alternative prompts. Shared with the OpenAI variant.
DEFAULT_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompts",
    "extractor_neuroscience_ner.txt",
)

# Used when the PDF is sent as an attachment and parsed by OpenRouter.
USER_PROMPT_ATTACHED = """\
INPUT TEXT:
The text to process is the attached file. Treat the full extracted text of
the attached document as the INPUT TEXT.

METADATA (paper_title / doi / source_path) — populate `source_metadata` from
this; do NOT repeat on every entity:
{metadata_json}
"""

# Used when pre-parsed text (e.g. XML) is inlined directly into the prompt.
USER_PROMPT_INLINE = """\
INPUT TEXT:
<<<
{input_text}
>>>

METADATA (paper_title / doi / source_path) — populate `source_metadata` from
this; do NOT repeat on every entity:
{metadata_json}
"""


def encode_pdf_to_data_url(pdf_path):
    """Read a local PDF and return it as a base64 data URL."""
    with open(pdf_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def main():
    parser = argparse.ArgumentParser(
        description="Run neuroscience-wide NER on a local PDF via an OpenRouter model."
    )
    parser.add_argument("--file", "-f", required=True, help="Path to the PDF to process.")
    parser.add_argument(
        "--model", "-m", required=True, help="OpenRouter model, e.g. openai/gpt-5.5."
    )
    parser.add_argument(
        "--prompt-file",
        "-p",
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to the system/extractor prompt file. Default: {DEFAULT_PROMPT_FILE}",
    )
    parser.add_argument(
        "--pdf-engine",
        default="cloudflare-ai",
        choices=["mistral-ocr", "cloudflare-ai", "native"],
        help="OpenRouter file-parser PDF engine. Default: cloudflare-ai. "
        "Ignored when --grobid is set.",
    )
    parser.add_argument(
        "--grobid",
        action="store_true",
        help="Parse the PDF locally with Grobid to TEI XML before sending to "
        "the LLM (inlined as text), instead of using OpenRouter's PDF parser.",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL of the Grobid service. Default: http://localhost:8070.",
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
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="Sampling temperature for reproducible runs (e.g. 0). "
        "Omitted by default; some reasoning models reject this.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed for reproducible runs. Omitted by default; "
        "some reasoning models reject this.",
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

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is not set (check your .env).")

    vprint(f"Loading system prompt from: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as fh:
        system_prompt = fh.read()

    vprint("Initializing OpenRouter client...")
    # Generous read timeout for very long generations; fail rather than hang forever.
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=1800.0,
        max_retries=2,
    )

    # Three input modes:
    #   - PDF + --grobid : parse locally to TEI XML, then inline as text.
    #   - PDF (default)   : send as attachment, parsed by OpenRouter's plugin.
    #   - non-PDF         : pre-parsed text (XML, plain text) inlined directly.
    is_pdf = args.file.lower().endswith(".pdf")
    metadata_json = json.dumps(metadata)

    if is_pdf and args.grobid:
        vprint(f"Parsing PDF with Grobid at {args.grobid_url} ...")
        grobid = GrobidClient(base_url=args.grobid_url)
        if not grobid.is_alive():
            parser.error(
                f"Grobid service is not reachable at {args.grobid_url}. "
                "Start it (e.g. `docker run -p 8070:8070 lfoppiano/grobid`) "
                "or pass --grobid-url."
            )
        input_text = grobid.process_fulltext(Path(args.file))
        vprint(f"  Grobid returned {len(input_text)} chars of TEI XML.")
        input_mode = "grobid_tei"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": USER_PROMPT_INLINE.format(
                    input_text=input_text, metadata_json=metadata_json
                ),
            },
        ]
        plugins = None
    elif is_pdf:
        vprint(f"Encoding PDF: {args.file} ...")
        data_url = encode_pdf_to_data_url(args.file)
        input_mode = "pdf_attachment"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT_ATTACHED.format(metadata_json=metadata_json),
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(args.file),
                            "file_data": data_url,
                        },
                    },
                ],
            },
        ]
        plugins = [{"id": "file-parser", "pdf": {"engine": args.pdf_engine}}]
    else:
        vprint(f"Reading pre-parsed text file: {args.file} ...")
        with open(args.file, "r", encoding="utf-8") as fh:
            input_text = fh.read()
        vprint(f"  read {len(input_text)} chars.")
        input_mode = "inline_text"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": USER_PROMPT_INLINE.format(
                    input_text=input_text, metadata_json=metadata_json
                ),
            },
        ]
        plugins = None

    # Only send sampling controls when explicitly requested; some reasoning
    # models reject temperature/seed outright.
    request_kwargs = {}
    if args.temperature is not None:
        request_kwargs["temperature"] = args.temperature
    if args.seed is not None:
        request_kwargs["seed"] = args.seed
    if request_kwargs:
        vprint(f"  sampling controls: {request_kwargs}")

    # The file-parser plugin only applies to PDF attachments.
    if plugins is not None:
        request_kwargs["extra_body"] = {"plugins": plugins}

    if input_mode == "pdf_attachment":
        source_desc = f"pdf-engine={args.pdf_engine}"
    elif input_mode == "grobid_tei":
        source_desc = "grobid TEI"
    else:
        source_desc = "inline text"
    vprint(
        f"Sending request to model '{args.model}' "
        f"({source_desc}, streaming; this may take a while)..."
    )
    chunks = []
    chars = 0
    next_report = 2000  # print a progress line every ~2000 chars
    finish_reason = None
    stream = client.chat.completions.create(
        model=args.model,
        messages=messages,
        extra_headers=EXTRA_HEADERS,
        stream=True,
        **request_kwargs,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = getattr(choice.delta, "content", None)
        if delta:
            chunks.append(delta)
            chars += len(delta)
            if chars >= next_report:
                vprint(f"  ...streaming, {chars} chars received so far")
                next_report += 2000
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    output_text = "".join(chunks)
    vprint(f"  response complete — {chars} chars total (finish_reason={finish_reason}).")

    # Detect truncation: a low entity count is often an output-token cutoff
    # rather than the model deciding it was done.
    if finish_reason == "length":
        print(
            "WARNING: response was truncated (finish_reason='length'). "
            "Entity count is an undercount. "
            "Consider raising the model's output-token limit or chunking the input."
        )

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
            "prompt_file": args.prompt_file,
            "input_mode": input_mode,
            "pdf_engine": args.pdf_engine if input_mode == "pdf_attachment" else None,
            "grobid_url": args.grobid_url if input_mode == "grobid_tei" else None,
            "temperature": args.temperature,
            "seed": args.seed,
            "finish_reason": finish_reason,
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
