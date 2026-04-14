#!/usr/bin/env python3
"""
Resource Extraction Evaluation for StructSense.

Evaluates the quality of scientific resource metadata extraction from
research papers. Uses hardcoded ground truth for ViTPose and DeepLabCut
papers, built from cross-model consensus and paper content.

Evaluates across dimensions:
  1. Primary resource fields — name, type, category, target, URL
  2. Mentions recall — datasets, models, benchmarks, tools found
  3. Ontology mapping quality — are mappings sensible?

Usage:
    python resource_eval.py                           # All papers + models
    python resource_eval.py <result.json>             # Single file
    python resource_eval.py -o report.json            # Save JSON report
    python resource_eval.py --verbose                 # Show per-field details
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
# TEXT UTILITIES
# =============================================================================

def normalize(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def fuzzy_match(a: str, b: str) -> float:
    """Fuzzy match score between two strings (0-1)."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def name_in_list(name: str, items: list, threshold: float = 0.7) -> bool:
    """Check if a name fuzzy-matches any item in list."""
    name_norm = normalize(name)
    for item in items:
        if isinstance(item, dict):
            item_text = normalize(item.get("name", ""))
        else:
            item_text = normalize(str(item))
        if not item_text:
            continue
        # Exact substring or high fuzzy match
        if name_norm in item_text or item_text in name_norm:
            return True
        if fuzzy_match(name_norm, item_text) >= threshold:
            return True
    return False


# =============================================================================
# GROUND TRUTH — built from cross-model consensus + paper content
# =============================================================================

GROUND_TRUTH = {
    "vitpose": {
        "primary": {
            "name": "ViTPose",
            "type": "Model",
            "category": "Pose Estimation",
            "target": "Human",
            "url_keywords": ["vitpose", "github"],
            # specific_target varies but should mention "keypoint" or "pose"
            "specific_target_keywords": ["keypoint", "pose", "human"],
        },
        # Mentions ground truth: entities that appear in ALL or MOST model outputs
        "mentions": {
            "related_models": {
                # Core models mentioned in the paper (consensus across 3+ outputs)
                "required": [
                    "PRTR", "TokenPose", "TransPose", "HRFormer", "MAE", "ViT",
                ],
                "optional": [
                    "ViT-B", "ViT-L", "ViT-H", "HRNet", "SimpleBaseline",
                    "Swin", "MLP-Mixer", "ResNet",
                ],
            },
            "datasets": {
                "required": [
                    "MS COCO", "ImageNet", "AI Challenger", "MPII",
                    "CrowdPose", "OCHuman",
                ],
                "optional": [
                    "AIC", "ImageNet-1K", "ImageNet-22K",
                ],
            },
            "benchmarks": {
                "required": ["COCO"],
                "optional": ["COCO test-dev", "COCO val", "MPII"],
            },
            "tools": {
                "required": ["mmpose"],
                "optional": [],
            },
        },
    },
    "deeplabcut": {
        "primary": {
            "name": "DeepLabCut",
            "type": "Tool",
            "category": "Pose Estimation",
            "target": "Animal",
            "url_keywords": ["deeplabcut", "github"],
            "specific_target_keywords": ["mice", "marmoset", "fish", "animal", "multi"],
        },
        "mentions": {
            "related_models": {
                "required": [
                    "DLCRNet", "OpenPose", "HRNet", "ResNet",
                ],
                "optional": [
                    "EfficientNet", "DLCRNet_ms5", "MobileNet",
                    "ResNet-AE", "HRNet-AE", "Transformer",
                ],
            },
            "datasets": {
                "required": [
                    "tri-mouse", "parenting", "marmoset", "fish",
                ],
                "optional": [
                    "COCO", "pups",
                ],
            },
            "benchmarks": {
                "required": ["COCO", "MOTA"],
                "optional": ["mAP", "benchmark.deeplabcut.org"],
            },
            "tools": {
                "required": ["idtracker.ai"],
                "optional": [
                    "DeepLabCut-Live", "NetworkX", "py-motmetrics", "MMPose",
                ],
            },
        },
    },
}


# =============================================================================
# EVALUATION DIMENSIONS
# =============================================================================

def eval_primary_fields(resource: dict, gt: dict) -> dict:
    """Evaluate primary resource metadata fields."""
    gt_primary = gt["primary"]
    results = {}

    # Name
    name = resource.get("name", "")
    name_sim = fuzzy_match(name, gt_primary["name"])
    results["name"] = {
        "extracted": name,
        "expected": gt_primary["name"],
        "score": 1.0 if name_sim >= 0.8 else name_sim,
        "pass": name_sim >= 0.8,
    }

    # Type
    rtype = normalize(resource.get("type", ""))
    expected_type = normalize(gt_primary["type"])
    type_match = rtype == expected_type or expected_type in rtype
    results["type"] = {
        "extracted": resource.get("type", ""),
        "expected": gt_primary["type"],
        "score": 1.0 if type_match else 0.0,
        "pass": type_match,
    }

    # Category
    cat = normalize(resource.get("category", ""))
    expected_cat = normalize(gt_primary["category"])
    cat_match = expected_cat in cat or fuzzy_match(cat, expected_cat) >= 0.7
    results["category"] = {
        "extracted": resource.get("category", ""),
        "expected": gt_primary["category"],
        "score": 1.0 if cat_match else fuzzy_match(cat, expected_cat),
        "pass": cat_match,
    }

    # Target
    target = normalize(resource.get("target", ""))
    expected_target = normalize(gt_primary["target"])
    target_match = expected_target in target or target in expected_target
    results["target"] = {
        "extracted": resource.get("target", ""),
        "expected": gt_primary["target"],
        "score": 1.0 if target_match else 0.0,
        "pass": target_match,
    }

    # Specific target (keyword check)
    specific = normalize(resource.get("specific_target", ""))
    kw_found = sum(1 for kw in gt_primary["specific_target_keywords"] if kw in specific)
    kw_total = len(gt_primary["specific_target_keywords"])
    results["specific_target"] = {
        "extracted": resource.get("specific_target", ""),
        "keywords_found": kw_found,
        "keywords_total": kw_total,
        "score": kw_found / kw_total if kw_total > 0 else 0,
        "pass": kw_found >= 1,  # At least one keyword
    }

    # URL
    url = normalize(resource.get("url") or "")
    url_kws = gt_primary["url_keywords"]
    url_found = sum(1 for kw in url_kws if kw in url)
    has_url = bool(url) and url not in ("none", "missing", "null")
    results["url"] = {
        "extracted": resource.get("url", ""),
        "keywords_found": url_found,
        "keywords_total": len(url_kws),
        "score": 1.0 if url_found == len(url_kws) else (0.5 if has_url else 0.0),
        "pass": url_found >= 1,
    }

    # Overall
    scores = [r["score"] for r in results.values()]
    results["overall_score"] = sum(scores) / len(scores) if scores else 0
    results["fields_passed"] = sum(1 for r in results.values() if isinstance(r, dict) and r.get("pass", False))
    results["fields_total"] = len([r for r in results.values() if isinstance(r, dict) and "pass" in r])

    return results


def eval_mentions(resource: dict, gt: dict) -> dict:
    """Evaluate mentions (datasets, models, benchmarks, tools) recall."""
    gt_mentions = gt["mentions"]
    mentions = resource.get("mentions", {})
    results = {}

    for category, gt_lists in gt_mentions.items():
        extracted = mentions.get(category, [])
        required = gt_lists["required"]
        optional = gt_lists["optional"]

        # Check required items
        required_found = []
        required_missed = []
        for req in required:
            if name_in_list(req, extracted):
                required_found.append(req)
            else:
                required_missed.append(req)

        # Check optional items
        optional_found = []
        for opt in optional:
            if name_in_list(opt, extracted):
                optional_found.append(opt)

        # Extra items (not in required or optional)
        all_gt = required + optional
        extra = []
        for item in extracted:
            item_name = item.get("name", item) if isinstance(item, dict) else str(item)
            is_known = any(
                normalize(gt_name) in normalize(item_name) or normalize(item_name) in normalize(gt_name)
                for gt_name in all_gt
            )
            if not is_known:
                extra.append(item_name)

        req_score = len(required_found) / len(required) if required else 1.0

        results[category] = {
            "required_found": required_found,
            "required_missed": required_missed,
            "required_recall": req_score,
            "optional_found": optional_found,
            "extra_items": extra[:10],  # cap for readability
            "total_extracted": len(extracted),
        }

    # Overall mentions score (average of required recall across categories)
    cat_scores = [r["required_recall"] for r in results.values()]
    results["overall_score"] = sum(cat_scores) / len(cat_scores) if cat_scores else 0

    return results


def eval_ontology_mappings(resource: dict) -> dict:
    """Evaluate quality of ontology mappings (flag nonsensical ones)."""
    issues = []
    total_mappings = 0
    sensible_mappings = 0

    # Check primary field mappings
    for field in ["mapped_target_concept", "mapped_name_concept",
                  "mapped_type_concept", "mapped_category_concept",
                  "mapped_specific_target_concept"]:
        mappings = resource.get(field, [])
        if not mappings:
            continue

        if isinstance(mappings, dict):
            mappings = [mappings]

        for m in mappings:
            if isinstance(m, dict) and "mapped_target_concept" in m:
                # Nested structure for specific_target
                m = m["mapped_target_concept"]
            if not isinstance(m, dict):
                continue

            total_mappings += 1
            ontology_label = m.get("label", "")
            ontology = m.get("ontology", "")
            ont_id = m.get("id", "")

            # Heuristic: flag pharmaceutical/drug mappings for CS/ML resources
            is_pharma = any(kw in normalize(ontology_label) for kw in [
                "drug", "pharmaceutical", "tablet", "capsule", "injection",
                "susp,", "mg/ml", "vaccine", "therapy", "dosage",
                "calfactant", "antigen",
            ])
            is_plant = any(kw in normalize(ontology_label) for kw in [
                "magnolia", "plant", "flower",
            ])
            is_genetic_disease = any(kw in normalize(ontology_label) for kw in [
                "amelogenesis", "imperfecta",
            ])

            if is_pharma or is_plant or is_genetic_disease:
                issues.append({
                    "field": field,
                    "label": ontology_label[:80],
                    "ontology": ontology,
                    "issue": "nonsensical mapping",
                })
            else:
                sensible_mappings += 1

    # Check mention mappings
    for cat_key in ["related_models", "datasets", "benchmarks", "tools"]:
        cat_items = resource.get("mentions", {}).get(cat_key, [])
        for item in cat_items:
            if not isinstance(item, dict):
                continue
            ont_label = item.get("ontology_label", "")
            if not ont_label:
                continue
            total_mappings += 1
            is_bad = any(kw in normalize(ont_label) for kw in [
                "calfactant", "magnolia", "amelogenesis", "mg/ml",
                "tablet", "capsule", "injection", "vaccine",
                "pharmaceutical", "drug product",
            ])
            if is_bad:
                issues.append({
                    "field": f"mentions.{cat_key}",
                    "label": ont_label[:80],
                    "issue": "nonsensical mapping",
                })
            else:
                sensible_mappings += 1

    score = sensible_mappings / total_mappings if total_mappings > 0 else 0.0

    return {
        "total_mappings": total_mappings,
        "sensible_mappings": sensible_mappings,
        "nonsensical_count": len(issues),
        "score": round(score, 3),
        "issues": issues[:15],  # cap for readability
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def detect_paper(filepath: str) -> str | None:
    """Detect which paper a result file belongs to."""
    path_lower = filepath.lower()
    for paper in GROUND_TRUTH:
        if paper in path_lower:
            return paper
    return None


def evaluate_file(filepath: str) -> dict:
    """Evaluate a single result file."""
    with open(filepath) as f:
        data = json.load(f)

    resources = data.get("resources", [])
    errors = data.get("errors", [])

    if not resources:
        return {
            "file": filepath,
            "status": "failed",
            "error": errors[0].get("error", "No resources extracted")[:200] if errors else "No resources",
            "dimensions": {},
            "overall_score": 0.0,
        }

    paper = detect_paper(filepath)
    if not paper:
        return {
            "file": filepath,
            "status": "error",
            "error": f"Cannot detect paper from path: {filepath}",
            "dimensions": {},
            "overall_score": 0.0,
        }

    gt = GROUND_TRUTH[paper]
    resource = resources[0]  # Primary resource

    dimensions = {
        "primary_fields": eval_primary_fields(resource, gt),
        "mentions": eval_mentions(resource, gt),
        "ontology_mappings": eval_ontology_mappings(resource),
    }

    # Overall score: weighted
    weights = {"primary_fields": 0.40, "mentions": 0.40, "ontology_mappings": 0.20}
    overall = sum(
        dimensions[dim]["overall_score"] * weights[dim]
        if "overall_score" in dimensions[dim]
        else dimensions[dim]["score"] * weights[dim]
        for dim in weights
    )

    return {
        "file": filepath,
        "paper": paper,
        "status": "success",
        "dimensions": dimensions,
        "overall_score": round(overall, 3),
        "judge_score": resource.get("judge_score"),
    }


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================

def find_result_files(base_dir: str) -> list[str]:
    """Find all final result JSON files."""
    files = []
    for paper_dir in sorted(Path(base_dir).iterdir()):
        if not paper_dir.is_dir() or paper_dir.name in ("evaluation", "notebook", "script"):
            continue
        for result_dir in sorted(paper_dir.iterdir()):
            if not result_dir.is_dir() or not result_dir.name.startswith("results"):
                continue
            for json_file in sorted(result_dir.glob("*.json")):
                if "staged" not in str(json_file):
                    files.append(str(json_file))
    return files


# =============================================================================
# REPORTING
# =============================================================================

def print_report(reports: list[dict], verbose: bool = False) -> None:
    """Print human-readable report."""
    # Group by paper
    by_paper = defaultdict(list)
    for r in reports:
        by_paper[r.get("paper", "unknown")].append(r)

    for paper, paper_reports in by_paper.items():
        print(f"\n{'='*70}")
        print(f"  Resource Extraction: {paper}")
        print(f"{'='*70}")

        for r in paper_reports:
            fname = os.path.basename(r["file"])
            print(f"\n  {fname}")
            print(f"  {'─'*60}")

            if r["status"] != "success":
                print(f"  STATUS: {r['status'].upper()}")
                print(f"  Error: {r.get('error', '')}")
                continue

            d = r["dimensions"]
            print(f"  Overall Score: {r['overall_score']*100:.1f}%  (judge_score: {r.get('judge_score', 'N/A')})")

            # Primary fields
            pf = d["primary_fields"]
            print(f"\n  Primary Fields: {pf['fields_passed']}/{pf['fields_total']} passed ({pf['overall_score']*100:.0f}%)")
            for field in ["name", "type", "category", "target", "specific_target", "url"]:
                info = pf[field]
                mark = "PASS" if info["pass"] else "FAIL"
                extracted = info.get("extracted", "")
                if len(str(extracted)) > 50:
                    extracted = str(extracted)[:50] + "..."
                print(f"    {field:20s} [{mark}]  {extracted}")

            # Mentions
            mn = d["mentions"]
            print(f"\n  Mentions Recall: {mn['overall_score']*100:.0f}%")
            for cat in ["related_models", "datasets", "benchmarks", "tools"]:
                if cat not in mn:
                    continue
                info = mn[cat]
                n_found = len(info["required_found"])
                n_req = n_found + len(info["required_missed"])
                print(
                    f"    {cat:20s} {n_found}/{n_req} required  "
                    f"(+{len(info['optional_found'])} optional, "
                    f"+{len(info['extra_items'])} extra)"
                )
                if info["required_missed"] and verbose:
                    print(f"      missed: {info['required_missed']}")

            # Ontology mappings
            om = d["ontology_mappings"]
            print(
                f"\n  Ontology Mappings: {om['sensible_mappings']}/{om['total_mappings']} sensible "
                f"({om['score']*100:.0f}%), {om['nonsensical_count']} nonsensical"
            )
            if om["issues"] and verbose:
                for issue in om["issues"][:5]:
                    print(f"    {issue['field']}: \"{issue['label']}\" ({issue['issue']})")

    # Summary table
    if len(reports) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"\n  {'Paper':<14s} {'File':<35s} {'Primary':>8s} {'Mentions':>9s} {'Ontology':>9s} {'Overall':>8s}")
        print(f"  {'-'*85}")
        for r in reports:
            fname = os.path.basename(r["file"])[:35]
            paper = r.get("paper", "?")[:14]
            if r["status"] != "success":
                print(f"  {paper:<14s} {fname:<35s} {'FAIL':>8s} {'—':>9s} {'—':>9s} {'0.0%':>8s}")
            else:
                d = r["dimensions"]
                pf_score = d["primary_fields"]["overall_score"]
                mn_score = d["mentions"]["overall_score"]
                om_score = d["ontology_mappings"]["score"]
                print(
                    f"  {paper:<14s} {fname:<35s} "
                    f"{pf_score*100:>6.0f}% "
                    f"{mn_score*100:>7.0f}% "
                    f"{om_score*100:>7.0f}% "
                    f"{r['overall_score']*100:>6.1f}%"
                )
        print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate resource extraction quality"
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to result JSON file (or auto-detect from resource_extraction dir)",
    )
    parser.add_argument("-o", "--output", help="Save detailed report as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-field details")

    args = parser.parse_args()

    if args.input and os.path.isfile(args.input):
        reports = [evaluate_file(args.input)]
    else:
        base_dir = args.input if args.input else str(Path(__file__).resolve().parent.parent)
        files = find_result_files(base_dir)
        if not files:
            print(f"No result files found in {base_dir}")
            sys.exit(1)
        print(f"Found {len(files)} result files")
        reports = [evaluate_file(f) for f in files]

    print_report(reports, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
