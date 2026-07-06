#!/usr/bin/env python3
"""Evaluate structsense entity extraction against the CNS/MeSH ground-truth XML.

For every annotation row in the ground-truth BioC XML, this checks whether
structsense extracted that entity *in the same sentence*. Only the entity text is
compared -- annotation labels/types are ignored for the match itself (though they
are reported for breakdowns).

Matching rules (see the module CLI flags to change them):
  * Entity text  : normalized whole-string equality (case-insensitive, collapsed
                   whitespace, stripped surrounding punctuation).
  * Scope        : same-sentence. A ground-truth annotation counts as extracted
                   only if structsense has a matching-entity occurrence whose
                   sentence corresponds to the annotation's sentence. Because the
                   ground truth is a clean curated excerpt while structsense
                   processes PDFs (which inject reference numbers mid-sentence,
                   merge captions, etc.), sentence correspondence is measured by
                   token overlap rather than exact substring.
  * Duplicates   : every ground-truth annotation row is scored independently.

Usage:
    python evaluate_extraction.py \
        --xml NLM-CellLink_CNS_MeSH_train_val.xml \
        --output-dir structsense_extraction_output \
        --detail-csv evaluation_detail.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

_PUNCT = " .,;:!?()[]{}\"'`"


def normalize_entity(text: str) -> str:
    """Normalize an entity string for whole-string equality matching."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(_PUNCT)


def sentence_tokens(text: str, min_len: int) -> set[str]:
    """Alphanumeric word tokens of a sentence, for overlap-based comparison."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= min_len}


def normalize_blob(text: str) -> str:
    """Collapse to alphanumeric-only, for exact-substring sentence matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def sentence_containing(passage: str, offset: int, length: int) -> str:
    """Return the sentence of ``passage`` that contains the [offset, offset+length) span."""
    cursor = 0
    for sent in split_sentences(passage):
        start = passage.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        if start <= offset < end or (offset < end and offset + length > start):
            return sent
        cursor = end
    return passage  # fallback: whole passage


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class GTAnnotation:
    pmid: str
    passage_id: str
    entity_text: str
    entity_type: str
    offset: int
    length: int
    passage_text: str


@dataclass
class SSEntity:
    text: str
    norm: str
    sentences: list[str] = field(default_factory=list)  # occurrence sentences (raw)


def load_ground_truth(xml_path: str) -> list[GTAnnotation]:
    root = ET.parse(xml_path).getroot()
    rows: list[GTAnnotation] = []
    for doc in root.findall("document"):
        passage = doc.find("passage")
        if passage is None:
            continue
        pmid = passage_id = None
        for infon in passage.findall("infon"):
            key = infon.get("key")
            if key == "article-id_pmid":
                pmid = infon.text
            elif key == "passage_id":
                passage_id = infon.text
        passage_text = passage.findtext("text") or ""
        for ann in passage.findall("annotation"):
            loc = ann.find("location")
            if loc is None:
                continue
            atype = None
            for infon in ann.findall("infon"):
                if infon.get("key") == "type":
                    atype = infon.text
            rows.append(
                GTAnnotation(
                    pmid=pmid or "",
                    passage_id=passage_id or (doc.findtext("id") or ""),
                    entity_text=ann.findtext("text") or "",
                    entity_type=atype or "",
                    offset=int(loc.get("offset", 0)),
                    length=int(loc.get("length", 0)),
                    passage_text=passage_text,
                )
            )
    return rows


def load_structsense(output_dir: str) -> dict[str, dict[str, SSEntity]]:
    """Return {pmid: {normalized_entity_text: SSEntity}}."""
    index: dict[str, dict[str, SSEntity]] = {}
    for path in glob.glob(os.path.join(output_dir, "*.json")):
        pmid = os.path.basename(path).split("_")[0]
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        by_norm = index.setdefault(pmid, {})
        for ent in data.get("entities", []):
            raw = ent.get("entity", "")
            norm = normalize_entity(raw)
            if not norm:
                continue
            ss = by_norm.get(norm)
            if ss is None:
                ss = by_norm[norm] = SSEntity(text=raw, norm=norm)
            occurrences = ent.get("occurrences") or [{}]
            for occ in occurrences:
                sent = occ.get("sentence", "")
                if sent:
                    ss.sentences.append(sent)
    return index


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def sentence_matches(gt_sentence: str, ss_sentences: list[str], mode: str, threshold: float, min_token_len: int) -> bool:
    """Does any structsense occurrence sentence correspond to the GT sentence?"""
    if mode == "exact":
        gt_blob = normalize_blob(gt_sentence)
        if not gt_blob:
            return False
        for sent in ss_sentences:
            blob = normalize_blob(sent)
            if gt_blob in blob or (len(blob) > 20 and blob in gt_blob):
                return True
        return False
    # token-overlap (default): fraction of GT-sentence tokens present in an SS sentence
    gt_toks = sentence_tokens(gt_sentence, min_token_len)
    if not gt_toks:
        return False
    for sent in ss_sentences:
        ss_toks = sentence_tokens(sent, min_token_len)
        if ss_toks and len(gt_toks & ss_toks) / len(gt_toks) >= threshold:
            return True
    return False


@dataclass
class Result:
    ann: GTAnnotation
    covered: bool            # pmid has structsense output
    matched_anywhere: bool   # entity extracted somewhere in the paper
    matched_sentence: bool   # entity extracted in the same sentence as the annotation


def evaluate(
    gt: list[GTAnnotation],
    ss_index: dict[str, dict[str, SSEntity]],
    sent_mode: str,
    threshold: float,
    min_token_len: int,
) -> list[Result]:
    """Score every annotation on BOTH criteria (anywhere-in-paper and same-sentence)."""
    results: list[Result] = []
    for ann in gt:
        by_norm = ss_index.get(ann.pmid)
        covered = by_norm is not None
        norm = normalize_entity(ann.entity_text)
        ss = by_norm.get(norm) if covered else None
        matched_anywhere = ss is not None
        matched_sentence = False
        if matched_anywhere:
            gt_sent = sentence_containing(ann.passage_text, ann.offset, ann.length)
            matched_sentence = sentence_matches(gt_sent, ss.sentences, sent_mode, threshold, min_token_len)
        results.append(
            Result(ann=ann, covered=covered, matched_anywhere=matched_anywhere, matched_sentence=matched_sentence)
        )
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "n/a"


def report(results: list[Result], out=None) -> None:
    def emit(line: str = "") -> None:
        print(line)
        if out is not None:
            out.write(line + "\n")

    # Evaluate only over papers that have structsense output; uncovered papers are ignored.
    covered = [r for r in results if r.covered]
    n_cov = len(covered)
    n_papers = len({r.ann.pmid for r in covered})
    sent_cov = sum(r.matched_sentence for r in covered)
    any_cov = sum(r.matched_anywhere for r in covered)

    emit("# StructSense Extraction Evaluation")
    emit()
    emit(f"Evaluated over the {n_papers} papers that have structsense output "
         f"({n_cov} ground-truth annotation rows). Entity match: normalized whole-string "
         "(labels ignored); every GT row scored. Two recall metrics are reported side by side:")
    emit()
    emit("- **SAME-SENTENCE** — entity extracted in the same sentence the GT annotated it in.")
    emit("- **ANYWHERE** — entity extracted anywhere in the paper (location ignored).")
    emit()
    emit("## Recall (headline)")
    emit()
    emit("| Metric | Extracted | Recall |")
    emit("| --- | ---: | ---: |")
    emit(f"| Same-sentence | {sent_cov}/{n_cov} | {_pct(sent_cov, n_cov)} |")
    emit(f"| Anywhere | {any_cov}/{n_cov} | {_pct(any_cov, n_cov)} |")
    emit()

    # Per entity-type breakdown
    emit("## By entity type")
    emit()
    _breakdown(emit, covered, key=lambda r: r.ann.entity_type, label="Type")
    emit()

    # Per-PMID breakdown, ranked by same-sentence recall then volume
    emit("## By PMID (ranked by same-sentence recall)")
    emit()
    _breakdown(emit, covered, key=lambda r: r.ann.pmid, label="PMID")


def _breakdown(emit, rows: list[Result], key, label: str) -> None:
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    emit(f"| {label} | Total | Same-sentence | Anywhere |")
    emit("| --- | ---: | ---: | ---: |")
    ranked = sorted(
        groups.items(),
        key=lambda kv: (-(sum(r.matched_sentence for r in kv[1]) / len(kv[1])), -len(kv[1])),
    )
    for name, rs in ranked:
        n = len(rs)
        s = sum(r.matched_sentence for r in rs)
        a = sum(r.matched_anywhere for r in rs)
        emit(f"| {name} | {n} | {s} ({_pct(s, n)}) | {a} ({_pct(a, n)}) |")


def write_detail_csv(results: list[Result], path: str) -> None:
    # Only covered papers (those with structsense output); uncovered papers are ignored.
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["pmid", "passage_id", "entity_type", "entity_text", "offset", "length",
             "matched_anywhere", "matched_same_sentence"]
        )
        for r in results:
            if not r.covered:
                continue
            writer.writerow(
                [r.ann.pmid, r.ann.passage_id, r.ann.entity_type, r.ann.entity_text,
                 r.ann.offset, r.ann.length,
                 int(r.matched_anywhere), int(r.matched_sentence)]
            )
    print(f"Wrote per-annotation detail: {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", default=os.path.join(here, "NLM-CellLink_CNS_MeSH_train_val.xml"),
                        help="Ground-truth BioC XML file.")
    parser.add_argument("--output-dir", default=os.path.join(here, "structsense_extraction_output"),
                        help="Directory of structsense per-PMID JSON outputs.")
    parser.add_argument("--sentence-mode", choices=["overlap", "exact"], default="overlap",
                        help="Sentence correspondence test: token overlap (default, tolerant of PDF "
                             "reference-number insertions) or exact alphanumeric substring.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Token-overlap threshold for sentence correspondence (overlap mode). Default 0.6.")
    parser.add_argument("--min-token-len", type=int, default=3,
                        help="Minimum token length for sentence overlap comparison. Default 3.")
    parser.add_argument("--detail-csv", default=None, help="Optional path to write a per-annotation CSV.")
    parser.add_argument("--summary-out", default=os.path.join(here, "evaluation_summary.md"),
                        help="Path to write the Markdown summary report (also printed to stdout). "
                             "Pass '' to disable.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.xml):
        parser.error(f"XML not found: {args.xml}")
    if not os.path.isdir(args.output_dir):
        parser.error(f"Output dir not found: {args.output_dir}")

    gt = load_ground_truth(args.xml)
    ss_index = load_structsense(args.output_dir)
    results = evaluate(gt, ss_index, args.sentence_mode, args.threshold, args.min_token_len)
    if args.summary_out:
        with open(args.summary_out, "w") as fh:
            report(results, out=fh)
        print(f"Wrote summary report: {args.summary_out}")
    else:
        report(results)
    if args.detail_csv:
        write_detail_csv(results, args.detail_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
