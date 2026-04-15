#!/usr/bin/env python3
"""
Benchmark NER Evaluation for StructSense.

Evaluates entity extraction recall against standard benchmark datasets
(NCBI Disease, S800 Species) with JSONL ground truth files.

The ground truth defines the MINIMUM entities to extract. StructSense
typically extracts more — that's its advantage. So the primary metric
is recall (did we find the ground truth entities?), and extra entities
are counted as a positive signal.

Usage:
    python benchmark_eval.py                              # All datasets + all models
    python benchmark_eval.py --dataset ncbi               # One dataset
    python benchmark_eval.py --gt f.jsonl --result f.json # Explicit pair
    python benchmark_eval.py -o report.json               # Save JSON report
    python benchmark_eval.py --verbose                    # Show all missed entities
"""

import argparse
import json
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for matching.

    Handles the spaced-hyphen convention from BIO tokenization:
    "ataxia - telangiectasia" -> "ataxia-telangiectasia"
    """
    text = text.strip().lower()
    text = re.sub(r"\s*-\s*", "-", text)  # collapse spaced hyphens
    text = re.sub(r"\s*\.\s*", ".", text)  # collapse spaced periods (S . ratti -> S.ratti)
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text


# =============================================================================
# SPAN OVERLAP
# =============================================================================


def span_overlap_ratio(gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
    """Compute overlap ratio relative to ground truth span length."""
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    if overlap_start >= overlap_end:
        return 0.0
    gt_len = gt_end - gt_start
    if gt_len <= 0:
        return 0.0
    return (overlap_end - overlap_start) / gt_len


# =============================================================================
# SOURCE MODEL FILTER
# =============================================================================


def is_only_en_core_web_sm(entity_data: dict) -> bool:
    """Check if entity comes exclusively from en_core_web_sm."""
    sources = set()
    for p in entity_data.get("provenance", []):
        for s in p.get("sources", []):
            sources.add(s.get("source_model", ""))
    return sources == {"en_core_web_sm"}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_ground_truth(jsonl_path: str) -> list[dict]:
    """Load ground truth from JSONL. Returns flat list of entity mentions.

    Each mention: {text, normalized, char_start, char_end, sentence_id, sentence_text}
    Uses entities_global for character offsets relative to the full text.
    """
    mentions = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sent = json.loads(line)
            for ent in sent.get("entities_global", []):
                mentions.append(
                    {
                        "text": ent["text"],
                        "normalized": normalize_entity_text(ent["text"]),
                        "char_start": ent["char_start"],
                        "char_end": ent["char_end"],
                        "sentence_id": sent["sentence_id"],
                        "sentence_text": sent["sentence_text"],
                    }
                )
    return mentions


def load_results(json_path: str) -> tuple[list[dict], dict[str, list[dict]], list[tuple]]:
    """Load StructSense results. Returns:
    - entities: raw entity list (after filtering)
    - text_index: normalized_text -> [entity_data, ...]
    - intervals: sorted list of (global_start, global_end, entity_text, entity_data)
    """
    with open(json_path) as f:
        data = json.load(f)

    raw_entities = data.get("entities", [])
    total_count = len(raw_entities)

    # Filter en_core_web_sm-only
    entities = [e for e in raw_entities if not is_only_en_core_web_sm(e)]
    filtered_count = total_count - len(entities)

    # Build text index
    text_index: dict[str, list[dict]] = defaultdict(list)
    for ent in entities:
        norm = normalize_entity_text(ent["entity"])
        text_index[norm].append(ent)

    # Build interval list from occurrences
    intervals = []
    for ent in entities:
        for occ in ent.get("occurrences", []):
            gs = occ.get("global_start")
            ge = occ.get("global_end")
            if gs is not None and ge is not None:
                intervals.append((gs, ge, ent["entity"], ent))
        # Also use top-level start/end as fallback
        if not ent.get("occurrences"):
            s = ent.get("start")
            e = ent.get("end")
            if s is not None and e is not None:
                intervals.append((s, e, ent["entity"], ent))

    intervals.sort(key=lambda x: x[0])

    return entities, dict(text_index), intervals, total_count, filtered_count


# =============================================================================
# MATCHING
# =============================================================================


def match_ground_truth(
    gt_mentions: list[dict],
    text_index: dict[str, list[dict]],
    intervals: list[tuple],
    overlap_threshold: float = 0.5,
) -> dict:
    """Match GT mentions against result entities using 3-tier cascade.

    Returns a report dict with match counts and details.
    """
    exact_matches = []
    span_matches = []
    partial_matches = []
    missed = []

    # Pre-compute all normalized result texts for substring matching
    all_result_norms = set(text_index.keys())

    for gt in gt_mentions:
        gt_norm = gt["normalized"]
        gt_start = gt["char_start"]
        gt_end = gt["char_end"]

        # Tier 1: Normalized text match
        if gt_norm in text_index:
            exact_matches.append({**gt, "match_type": "exact", "matched_text": gt_norm})
            continue

        # Tier 2: Span overlap
        matched_span = False
        for rs, re_, rt, rd in intervals:
            ratio = span_overlap_ratio(gt_start, gt_end, rs, re_)
            if ratio >= overlap_threshold:
                span_matches.append(
                    {
                        **gt,
                        "match_type": "span",
                        "matched_text": rt,
                        "overlap_ratio": ratio,
                    }
                )
                matched_span = True
                break
        if matched_span:
            continue

        # Tier 3: Substring/containment match
        matched_partial = False
        for result_norm in all_result_norms:
            if gt_norm in result_norm or result_norm in gt_norm:
                # Avoid trivially short substring matches
                shorter = min(len(gt_norm), len(result_norm))
                longer = max(len(gt_norm), len(result_norm))
                if shorter >= 3 and shorter / longer >= 0.3:
                    partial_matches.append(
                        {
                            **gt,
                            "match_type": "partial",
                            "matched_text": result_norm,
                        }
                    )
                    matched_partial = True
                    break
        if matched_partial:
            continue

        # Not matched
        missed.append({**gt, "match_type": "missed"})

    return {
        "exact_matches": exact_matches,
        "span_matches": span_matches,
        "partial_matches": partial_matches,
        "missed": missed,
    }


def compute_extra_entities(
    entities: list[dict],
    gt_mentions: list[dict],
    text_index: dict[str, list[dict]],
) -> tuple[int, Counter]:
    """Count result entities that don't match any GT entity.

    Returns (count, label_distribution).
    """
    # Build set of normalized GT texts
    gt_norms = {m["normalized"] for m in gt_mentions}

    extra_count = 0
    extra_labels = Counter()

    # Check each unique result entity text
    for norm_text, ent_list in text_index.items():
        # Check if this result entity matches any GT (text or substring)
        matched = False
        if norm_text in gt_norms:
            matched = True
        else:
            for gt_norm in gt_norms:
                if gt_norm in norm_text or norm_text in gt_norm:
                    matched = True
                    break

        if not matched:
            for ent in ent_list:
                extra_count += 1
                extra_labels[ent.get("label", "UNKNOWN")] += 1

    return extra_count, extra_labels


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(gt_path: str, result_path: str, overlap_threshold: float = 0.5) -> dict:
    """Evaluate one result file against one ground truth file."""
    gt_mentions = load_ground_truth(gt_path)
    entities, text_index, intervals, total_count, filtered_count = load_results(result_path)

    match_report = match_ground_truth(gt_mentions, text_index, intervals, overlap_threshold)

    gt_unique = len({m["normalized"] for m in gt_mentions})

    n_exact = len(match_report["exact_matches"])
    n_span = len(match_report["span_matches"])
    n_partial = len(match_report["partial_matches"])
    n_missed = len(match_report["missed"])
    gt_total = len(gt_mentions)

    extra_count, extra_labels = compute_extra_entities(entities, gt_mentions, text_index)

    report = {
        "gt_file": gt_path,
        "result_file": result_path,
        "gt_total": gt_total,
        "gt_unique": gt_unique,
        "result_total": total_count,
        "result_filtered": filtered_count,
        "result_evaluated": len(entities),
        "exact_matches": n_exact,
        "span_matches": n_span,
        "partial_matches": n_partial,
        "missed": n_missed,
        "recall_strict": n_exact / gt_total if gt_total > 0 else 0.0,
        "recall_relaxed": (n_exact + n_span + n_partial) / gt_total if gt_total > 0 else 0.0,
        "extra_entities": extra_count,
        "extra_entity_labels": dict(extra_labels.most_common()),
        "missed_entities": _summarize_missed(match_report["missed"]),
    }
    return report


def _summarize_missed(missed: list[dict]) -> list[dict]:
    """Summarize missed GT entities, grouped by text."""
    by_text = defaultdict(list)
    for m in missed:
        by_text[m["text"]].append(m)

    summary = []
    for text, mentions in sorted(by_text.items(), key=lambda x: -len(x[1])):
        summary.append(
            {
                "text": text,
                "normalized": mentions[0]["normalized"],
                "count": len(mentions),
                "example_sentence": mentions[0]["sentence_text"][:120],
            }
        )
    return summary


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================


def find_benchmark_files(benchmark_dir: str, dataset_filter: str = None) -> list[dict]:
    """Auto-discover GT + result file pairs in the benchmark directory."""
    pairs = []
    benchmark_path = Path(benchmark_dir)

    for dataset_dir in sorted(benchmark_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        if dataset_name in ("script", "evaluation"):
            continue
        if dataset_filter and dataset_name != dataset_filter:
            continue

        # Find ground truth JSONL
        jsonl_files = list(dataset_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        gt_path = str(jsonl_files[0])

        # Find result files in results-*/ directories
        for result_dir in sorted(dataset_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            if not result_dir.name.startswith("results"):
                continue

            model_name = result_dir.name.replace("results-", "").replace("results_", "")

            # Find final result JSONs (not in staged_*)
            for json_file in sorted(result_dir.glob("*.json")):
                basename = json_file.name.lower()
                # Determine variant
                if any(x in basename for x in ["nhil", "no_hil", "non_hil", "without_hil"]):
                    variant = "nhil"
                elif "hil" in basename:
                    variant = "hil"
                else:
                    variant = "unknown"

                pairs.append(
                    {
                        "dataset": dataset_name,
                        "gt_path": gt_path,
                        "result_path": str(json_file),
                        "model": model_name,
                        "variant": variant,
                    }
                )

    return pairs


# =============================================================================
# REPORTING
# =============================================================================


def print_report(reports_by_dataset: dict[str, list[dict]], verbose: bool = False) -> None:
    """Print human-readable report grouped by dataset."""
    for dataset, reports in reports_by_dataset.items():
        print(f"\n{'='*70}")
        print(f"  Benchmark Evaluation: {dataset}")
        print(f"{'='*70}")

        # GT stats (same for all reports in this dataset)
        gt_total = reports[0]["gt_total"]
        gt_unique = reports[0]["gt_unique"]
        print(f"\n  Ground truth: {gt_total} mentions ({gt_unique} unique entities)")

        for r in reports:
            model = os.path.basename(os.path.dirname(r["result_file"]))
            variant = (
                "nhil"
                if "nhil" in os.path.basename(r["result_file"]).lower()
                or "no_hil" in os.path.basename(r["result_file"]).lower()
                or "non_hil" in os.path.basename(r["result_file"]).lower()
                or "without_hil" in os.path.basename(r["result_file"]).lower()
                else "hil"
            )
            fname = os.path.basename(r["result_file"])

            print(f"\n  {model} ({variant}) — {fname}")
            print(f"  {'─'*60}")
            print(f"  Result entities:     {r['result_total']:5d} (filtered {r['result_filtered']} en_core_web_sm)")
            print(f"  Evaluated:           {r['result_evaluated']:5d}")
            print()
            print(f"  Recall (strict):     {r['exact_matches']:4d}/{gt_total}  ({r['recall_strict']*100:5.1f}%)  [exact text match]")
            print(
                f"  Recall (relaxed):    {r['exact_matches']+r['span_matches']+r['partial_matches']:4d}/{gt_total}  ({r['recall_relaxed']*100:5.1f}%)  [+ span + partial]"
            )
            print(
                f"    Exact:  {r['exact_matches']:4d}  |  Span: {r['span_matches']:3d}  |  Partial: {r['partial_matches']:3d}  |  Missed: {r['missed']:4d}"
            )
            print()
            print(f"  Extra entities:      {r['extra_entities']:5d}  (beyond ground truth)")

            if r["extra_entity_labels"]:
                top_labels = list(r["extra_entity_labels"].items())[:8]
                label_str = ", ".join(f"{l}({c})" for l, c in top_labels)
                print(f"    Top labels: {label_str}")

            if r["missed_entities"]:
                n_show = 30 if verbose else 10
                print(f"\n  Missed GT entities (top {min(n_show, len(r['missed_entities']))}):")
                for m in r["missed_entities"][:n_show]:
                    print(f"    \"{m['text']}\" ({m['count']} mentions)")
                if len(r["missed_entities"]) > n_show:
                    print(f"    ... and {len(r['missed_entities']) - n_show} more unique entities")

    # Summary table
    if sum(len(v) for v in reports_by_dataset.values()) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"\n  {'Dataset':<12s} {'Model':<30s} {'Var':<5s} {'GT':>5s} {'Strict':>8s} {'Relaxed':>8s} {'Extra':>6s}")
        print(f"  {'-'*80}")
        for dataset, reports in reports_by_dataset.items():
            for r in reports:
                model = os.path.basename(os.path.dirname(r["result_file"]))
                variant = (
                    "nhil"
                    if "nhil" in os.path.basename(r["result_file"]).lower()
                    or "no_hil" in os.path.basename(r["result_file"]).lower()
                    or "non_hil" in os.path.basename(r["result_file"]).lower()
                    or "without_hil" in os.path.basename(r["result_file"]).lower()
                    else "hil"
                )
                print(
                    f"  {dataset:<12s} {model:<30s} {variant:<5s} "
                    f"{r['gt_total']:>5d} "
                    f"{r['recall_strict']*100:>6.1f}% "
                    f"{r['recall_relaxed']*100:>6.1f}% "
                    f"{r['extra_entities']:>6d}"
                )
        print()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER extraction recall against benchmark ground truth")
    parser.add_argument(
        "--dataset",
        help="Evaluate only this dataset (e.g., ncbi, s800)",
    )
    parser.add_argument(
        "--gt",
        help="Path to ground truth JSONL file (requires --result)",
    )
    parser.add_argument(
        "--result",
        help="Path to StructSense result JSON file (requires --gt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Save detailed report as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all missed entities (not just top 10)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Minimum span overlap ratio for tier-2 matching (default: 0.5)",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Path to benchmark directory (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Determine benchmark directory
    if args.benchmark_dir:
        benchmark_dir = args.benchmark_dir
    else:
        # Auto-detect: script is in evaluation/benchmark/analysis/
        script_dir = Path(__file__).resolve().parent
        benchmark_dir = str(script_dir.parent)

    # Explicit GT + result pair
    if args.gt and args.result:
        report = evaluate(args.gt, args.result, args.overlap_threshold)
        dataset_name = Path(args.gt).parent.name
        reports_by_dataset = {dataset_name: [report]}
    elif args.gt or args.result:
        print("Error: --gt and --result must be used together")
        sys.exit(1)
    else:
        # Auto-discover
        pairs = find_benchmark_files(benchmark_dir, args.dataset)
        if not pairs:
            print(f"No benchmark files found in {benchmark_dir}")
            if args.dataset:
                print(f"  (filtered by dataset: {args.dataset})")
            sys.exit(1)

        print(f"Found {len(pairs)} result files across {len(set(p['dataset'] for p in pairs))} datasets")

        reports_by_dataset = defaultdict(list)
        for pair in pairs:
            report = evaluate(pair["gt_path"], pair["result_path"], args.overlap_threshold)
            reports_by_dataset[pair["dataset"]].append(report)

    # Print
    print_report(dict(reports_by_dataset), verbose=args.verbose)

    # Save JSON
    if args.output:
        all_reports = {}
        for dataset, reports in reports_by_dataset.items():
            all_reports[dataset] = reports
        with open(args.output, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
