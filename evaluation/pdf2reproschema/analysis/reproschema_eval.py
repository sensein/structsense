#!/usr/bin/env python3
"""
PDF-to-ReproSchema Evaluation for StructSense.

Evaluates the quality of questionnaire extraction from PDF to ReproSchema
format. Uses hardcoded ground truth from the MFQ (Mood and Feelings
Questionnaire) Long Version.

Evaluates across dimensions:
  1. Item completeness — all 33 questions extracted?
  2. Question text accuracy — fuzzy match against PDF text
  3. Response options — correct values and labels?
  4. Scoring — per-item scoring and computed total formula?
  5. Activity metadata — title, preamble, citation?
  6. Item ordering — correct sequence?
  7. Input type — all items radio?

Usage:
    python reproschema_eval.py                         # All result files
    python reproschema_eval.py <result.json>           # Single file
    python reproschema_eval.py -o report.json          # Save JSON report
    python reproschema_eval.py --verbose               # Show per-item details
"""

import argparse
import json
import glob
import os
import re
import sys
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path


# =============================================================================
# GROUND TRUTH — MFQ Long Version (33 items)
# =============================================================================

GT_ACTIVITY = {
    "prefLabel_keywords": ["mood", "feelings", "questionnaire", "long"],
    "description_keywords": ["feeling", "acting", "recently", "past two weeks"],
    "preamble_keywords": [
        "not true",
        "sometimes",
        "true",
        "past two weeks",
        "feeling or acting",
    ],
    "citation_keywords": ["angold", "costello", "1987", "duke"],
}

GT_QUESTIONS = [
    "I felt miserable or unhappy.",
    "I didn\u2019t enjoy anything at all.",
    "I was less hungry than usual.",
    "I ate more than usual.",
    "I felt so tired I just sat around and did nothing.",
    "I was moving and walking more slowly than usual.",
    "I was very restless.",
    "I felt I was no good anymore.",
    "I blamed myself for things that weren\u2019t my fault.",
    "It was hard for me to make up my mind.",
    "I felt grumpy and cross with other people.",
    "I felt like talking less than usual.",
    "I was talking more slowly than usual.",
    "I cried a lot.",
    "I thought there was nothing good for me in the future.",
    "I thought that life wasn\u2019t worth living.",
    "I thought about death or dying.",
    "I thought my family would be better off without me.",
    "I thought about killing myself.",
    "I didn\u2019t want to see my friends.",
    "I found it hard to think properly or concentrate.",
    "I thought bad things would happen to me.",
    "I hated myself.",
    "I felt I was a bad person.",
    "I thought I looked ugly.",
    "I worried about aches and pains.",
    "I felt lonely.",
    "I thought nobody really loved me.",
    "I didn\u2019t have any fun in any of my activities.",
    "I thought I could never be as good as other people.",
    "I did everything wrong.",
    "I didn\u2019t sleep as well as I usually sleep.",
    "I slept a lot more than usual.",
]

GT_RESPONSE_OPTIONS = [
    {"value": 0, "name": "NOT TRUE"},
    {"value": 1, "name": "SOMETIMES"},
    {"value": 2, "name": "TRUE"},
]

GT_SCORING = {"NOT TRUE": 0, "SOMETIMES": 1, "TRUE": 2}

GT_NUM_ITEMS = 33
GT_INPUT_TYPE = "radio"


# =============================================================================
# TEXT SIMILARITY
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip()
    # Normalize smart quotes to straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings (0-1)."""
    a_norm = normalize_text(a).lower()
    b_norm = normalize_text(b).lower()
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def contains_keywords(text: str, keywords: list[str]) -> tuple[int, int]:
    """Check how many keywords appear in text. Returns (found, total)."""
    text_lower = normalize_text(text).lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found, len(keywords)


# =============================================================================
# EVALUATION DIMENSIONS
# =============================================================================


def eval_item_completeness(data: dict) -> dict:
    """Check if all 33 items were extracted."""
    items = data.get("items", [])
    n = len(items)
    return {
        "score": n / GT_NUM_ITEMS if GT_NUM_ITEMS > 0 else 0,
        "extracted": n,
        "expected": GT_NUM_ITEMS,
        "pass": n == GT_NUM_ITEMS,
    }


def eval_question_text(data: dict) -> dict:
    """Check question text accuracy against ground truth."""
    items = data.get("items", [])
    results = []
    total_sim = 0.0

    for i, gt_q in enumerate(GT_QUESTIONS):
        if i < len(items):
            extracted_q = items[i].get("question", "")
            sim = text_similarity(gt_q, extracted_q)
            total_sim += sim
            results.append(
                {
                    "item_num": i + 1,
                    "gt": gt_q,
                    "extracted": extracted_q,
                    "similarity": round(sim, 3),
                    "pass": sim >= 0.85,
                }
            )
        else:
            results.append(
                {
                    "item_num": i + 1,
                    "gt": gt_q,
                    "extracted": "",
                    "similarity": 0.0,
                    "pass": False,
                }
            )

    n_pass = sum(1 for r in results if r["pass"])
    avg_sim = total_sim / len(GT_QUESTIONS) if GT_QUESTIONS else 0

    return {
        "score": n_pass / GT_NUM_ITEMS,
        "correct": n_pass,
        "total": GT_NUM_ITEMS,
        "average_similarity": round(avg_sim, 3),
        "mismatches": [r for r in results if not r["pass"]],
    }


def eval_response_options(data: dict) -> dict:
    """Check response options for each item."""
    items = data.get("items", [])
    correct = 0
    issues = []

    for i, item in enumerate(items):
        opts = item.get("responseOptions", [])
        # Normalize to list of (value, name) tuples
        extracted = [(o.get("value"), normalize_text(str(o.get("name", ""))).upper()) for o in opts]
        expected = [(o["value"], o["name"]) for o in GT_RESPONSE_OPTIONS]

        if extracted == expected:
            correct += 1
        else:
            # Check if values/names are correct even if format differs
            ext_values = {o.get("value") for o in opts}
            ext_names = {normalize_text(str(o.get("name", ""))).upper() for o in opts}
            gt_values = {o["value"] for o in GT_RESPONSE_OPTIONS}
            gt_names = {o["name"] for o in GT_RESPONSE_OPTIONS}

            if ext_values == gt_values and ext_names == gt_names:
                correct += 1  # Same content, maybe different order
            else:
                issues.append(
                    {
                        "item_id": item.get("id", f"item_{i}"),
                        "expected": expected,
                        "extracted": extracted,
                    }
                )

    n_items = len(items) if items else GT_NUM_ITEMS
    return {
        "score": correct / n_items if n_items > 0 else 0,
        "correct": correct,
        "total": n_items,
        "issues": issues,
    }


def eval_scoring(data: dict) -> dict:
    """Check per-item scoring maps and computed total formula."""
    items = data.get("items", [])
    results = {"per_item_scoring": {}, "computed_score": {}}

    # Per-item scoring
    items_with_scoring = 0
    correct_scoring = 0
    for item in items:
        scoring = item.get("scoring", {})
        if scoring:
            items_with_scoring += 1
            # Normalize and compare
            norm_scoring = {normalize_text(str(k)).upper(): v for k, v in scoring.items()}
            if norm_scoring == GT_SCORING:
                correct_scoring += 1

    results["per_item_scoring"] = {
        "items_with_scoring": items_with_scoring,
        "correct": correct_scoring,
        "total": len(items),
        "present": items_with_scoring > 0,
    }

    # Computed total score
    computed = data.get("computedScores", [])
    if computed:
        formula = computed[0].get("jsExpression", "")
        # Extract variable names from formula
        vars_in_formula = set(re.findall(r"[A-Za-z_]\w*", formula))
        # Check all items are referenced (by any ID scheme)
        n_vars = len(vars_in_formula)
        results["computed_score"] = {
            "present": True,
            "formula": formula,
            "variables_count": n_vars,
            "correct_count": n_vars == GT_NUM_ITEMS,
            "label": computed[0].get("label", ""),
        }
    else:
        results["computed_score"] = {
            "present": False,
            "formula": "",
            "variables_count": 0,
            "correct_count": False,
        }

    # Overall score
    has_formula = results["computed_score"]["present"] and results["computed_score"]["correct_count"]
    has_item_scoring = results["per_item_scoring"]["present"]
    score = 0.0
    if has_formula:
        score += 0.5
    if has_item_scoring and correct_scoring == len(items):
        score += 0.5
    elif has_item_scoring:
        score += 0.25

    results["score"] = score
    return results


def eval_activity_metadata(data: dict) -> dict:
    """Check activity-level metadata extraction."""
    activity = data.get("activity", {})
    results = {}

    for field, keywords in GT_ACTIVITY.items():
        field_name = field.replace("_keywords", "")
        value = activity.get(field_name, "")
        if value:
            found, total = contains_keywords(value, keywords)
            results[field_name] = {
                "present": True,
                "value": value[:120],
                "keywords_found": found,
                "keywords_total": total,
                "score": found / total if total > 0 else 0,
            }
        else:
            results[field_name] = {
                "present": False,
                "value": "",
                "keywords_found": 0,
                "keywords_total": len(keywords),
                "score": 0.0,
            }

    # Overall score: average of all field scores
    scores = [r["score"] for r in results.values()]
    overall = sum(scores) / len(scores) if scores else 0

    return {
        "score": overall,
        "fields": results,
    }


def eval_item_order(data: dict) -> dict:
    """Check if items are in correct order."""
    items = data.get("items", [])
    order_field = data.get("order", [])

    # Check items array order by question text
    items_in_order = True
    for i, item in enumerate(items):
        if i < len(GT_QUESTIONS):
            sim = text_similarity(item.get("question", ""), GT_QUESTIONS[i])
            if sim < 0.85:
                items_in_order = False
                break

    # Check order field
    order_correct = False
    if order_field:
        # Order should have exactly 33 entries matching item IDs
        item_ids = [item.get("id") for item in items]
        order_correct = order_field == item_ids and len(order_field) == GT_NUM_ITEMS

    return {
        "score": 1.0 if items_in_order else 0.0,
        "items_in_order": items_in_order,
        "order_field_present": len(order_field) > 0,
        "order_field_correct": order_correct,
        "order_field_count": len(order_field),
    }


def eval_input_type(data: dict) -> dict:
    """Check if all items have correct input type."""
    items = data.get("items", [])
    correct = 0
    issues = []

    for item in items:
        input_type = item.get("inputType", "").lower()
        if input_type == GT_INPUT_TYPE:
            correct += 1
        else:
            issues.append(
                {
                    "item_id": item.get("id"),
                    "expected": GT_INPUT_TYPE,
                    "extracted": input_type,
                }
            )

    n = len(items) if items else GT_NUM_ITEMS
    return {
        "score": correct / n if n > 0 else 0,
        "correct": correct,
        "total": n,
        "issues": issues,
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def evaluate_file(filepath: str) -> dict:
    """Evaluate a single result file."""
    with open(filepath) as f:
        data = json.load(f)

    # Check if extraction failed (Qwen timeout case)
    errors = data.get("errors", [])
    items = data.get("items", [])
    if not items and errors:
        return {
            "file": filepath,
            "status": "failed",
            "error": errors[0].get("error", "Unknown error")[:200],
            "dimensions": {},
            "overall_score": 0.0,
        }

    dimensions = {
        "item_completeness": eval_item_completeness(data),
        "question_text": eval_question_text(data),
        "response_options": eval_response_options(data),
        "scoring": eval_scoring(data),
        "activity_metadata": eval_activity_metadata(data),
        "item_order": eval_item_order(data),
        "input_type": eval_input_type(data),
    }

    # Overall score: weighted average
    weights = {
        "item_completeness": 0.20,
        "question_text": 0.25,
        "response_options": 0.15,
        "scoring": 0.15,
        "activity_metadata": 0.10,
        "item_order": 0.10,
        "input_type": 0.05,
    }

    overall = sum(dimensions[dim]["score"] * weights[dim] for dim in weights)

    return {
        "file": filepath,
        "status": "success",
        "dimensions": dimensions,
        "overall_score": round(overall, 3),
    }


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================


def find_result_files(base_dir: str) -> list[dict]:
    """Find all result JSON files."""
    files = []
    base = Path(base_dir)

    for result_dir in sorted(base.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.startswith("results"):
            continue
        model = result_dir.name.replace("results_", "").replace("results-", "")

        for json_file in sorted(result_dir.glob("*.json")):
            basename = json_file.name.lower()
            if "nhil" in basename or "no_hil" in basename or "non_hil" in basename:
                variant = "nhil"
            elif "hil" in basename:
                variant = "hil"
            else:
                variant = "unknown"

            files.append(
                {
                    "path": str(json_file),
                    "model": model,
                    "variant": variant,
                }
            )

    return files


# =============================================================================
# REPORTING
# =============================================================================


def print_report(reports: list[dict], verbose: bool = False) -> None:
    """Print human-readable report."""
    for r in reports:
        fname = os.path.basename(r["file"])
        print(f"\n{'='*70}")
        print(f"  {fname}")
        print(f"{'='*70}")

        if r["status"] == "failed":
            print(f"\n  STATUS: FAILED")
            print(f"  Error: {r['error']}")
            print(f"  Overall Score: 0.0%")
            continue

        dims = r["dimensions"]
        print(f"\n  Overall Score: {r['overall_score']*100:.1f}%\n")

        # Item completeness
        ic = dims["item_completeness"]
        mark = "PASS" if ic["pass"] else "FAIL"
        print(f"  Item Completeness:    {ic['extracted']}/{ic['expected']}  ({ic['score']*100:.0f}%)  [{mark}]")

        # Question text
        qt = dims["question_text"]
        print(
            f"  Question Text:        {qt['correct']}/{qt['total']}  ({qt['score']*100:.0f}%)  avg similarity: {qt['average_similarity']:.3f}"
        )
        if qt["mismatches"] and verbose:
            for m in qt["mismatches"]:
                print(f"    Q{m['item_num']}: sim={m['similarity']:.3f}")
                print(f"      GT:  {m['gt'][:70]}")
                print(f"      Got: {m['extracted'][:70]}")

        # Response options
        ro = dims["response_options"]
        print(f"  Response Options:     {ro['correct']}/{ro['total']}  ({ro['score']*100:.0f}%)")
        if ro["issues"] and verbose:
            for issue in ro["issues"][:5]:
                print(f"    {issue['item_id']}: expected {issue['expected']}, got {issue['extracted']}")

        # Scoring
        sc = dims["scoring"]
        pis = sc["per_item_scoring"]
        cs = sc["computed_score"]
        scoring_parts = []
        if pis["present"]:
            scoring_parts.append(f"per-item: {pis['correct']}/{pis['total']}")
        else:
            scoring_parts.append("per-item: missing")
        if cs["present"]:
            count_ok = "correct" if cs["correct_count"] else f"wrong count ({cs['variables_count']})"
            scoring_parts.append(f"total formula: {count_ok}")
        else:
            scoring_parts.append("total formula: missing")
        print(f"  Scoring:              {sc['score']*100:.0f}%  ({'; '.join(scoring_parts)})")

        # Activity metadata
        am = dims["activity_metadata"]
        fields = am["fields"]
        field_parts = []
        for name, info in fields.items():
            status = f"{info['keywords_found']}/{info['keywords_total']}" if info["present"] else "missing"
            field_parts.append(f"{name}: {status}")
        print(f"  Activity Metadata:    {am['score']*100:.0f}%  ({'; '.join(field_parts)})")

        # Item order
        io = dims["item_order"]
        order_status = "correct" if io["items_in_order"] else "wrong"
        field_status = "present & correct" if io["order_field_correct"] else "present but wrong" if io["order_field_present"] else "missing"
        print(f"  Item Order:           {io['score']*100:.0f}%  (sequence: {order_status}; order field: {field_status})")

        # Input type
        it = dims["input_type"]
        print(f"  Input Type:           {it['correct']}/{it['total']}  ({it['score']*100:.0f}%)")

    # Summary table
    if len(reports) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(
            f"\n  {'File':<45s} {'Status':<8s} {'Items':>5s} {'QText':>6s} {'Resp':>5s} {'Score':>6s} {'Meta':>5s} {'Order':>5s} {'Overall':>8s}"
        )
        print(f"  {'-'*90}")
        for r in reports:
            fname = os.path.basename(r["file"])[:45]
            if r["status"] == "failed":
                print(f"  {fname:<45s} {'FAIL':<8s} {'—':>5s} {'—':>6s} {'—':>5s} {'—':>6s} {'—':>5s} {'—':>5s} {'0.0%':>8s}")
            else:
                d = r["dimensions"]
                print(
                    f"  {fname:<45s} {'OK':<8s} "
                    f"{d['item_completeness']['score']*100:>4.0f}% "
                    f"{d['question_text']['score']*100:>5.0f}% "
                    f"{d['response_options']['score']*100:>4.0f}% "
                    f"{d['scoring']['score']*100:>5.0f}% "
                    f"{d['activity_metadata']['score']*100:>4.0f}% "
                    f"{d['item_order']['score']*100:>4.0f}% "
                    f"{r['overall_score']*100:>6.1f}%"
                )
        print()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate PDF-to-ReproSchema extraction quality")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to result JSON file (or auto-detect from pdf2reproschema dir)",
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
        help="Show per-item details for mismatches",
    )

    args = parser.parse_args()

    if args.input and os.path.isfile(args.input):
        reports = [evaluate_file(args.input)]
    else:
        # Auto-detect
        if args.input:
            base_dir = args.input
        else:
            script_dir = Path(__file__).resolve().parent
            base_dir = str(script_dir.parent)

        files = find_result_files(base_dir)
        if not files:
            print(f"No result files found in {base_dir}")
            sys.exit(1)

        print(f"Found {len(files)} result files")
        reports = [evaluate_file(f["path"]) for f in files]

    print_report(reports, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
