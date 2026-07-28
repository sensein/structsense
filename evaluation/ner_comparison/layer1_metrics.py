"""
Layer 1 evaluation metrics: entity span and label comparison.

Both systems assign free-form labels. Label agreement is computed by canonicalizing
both labels through ner_eval._canonicalize_label() and checking for equality.
Entity filtering (noise, stopwords, generics) uses ner_eval.is_excluded().

Layer 1A — pre-alignment: StructSense extractor output vs. direct API call
Layer 1B — post-alignment: full StructSense output (judge_score filtered) vs. direct API call
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Type alias for sentence-grounded span keys: (normalized_entity, sentence_fingerprint)
SpanKey = tuple[str, str]

# Import shared heuristics from the existing ner_eval module
sys.path.insert(0, str(Path(__file__).parent.parent / "ner" / "analysis"))
from ner_eval import _canonicalize_label, is_excluded  # noqa: E402

JUDGE_SCORE_THRESHOLD = 0.8

# Common neuroscience abbreviation expansions applied before span matching.
# Two spans are compared after normalization so "PFC" and "prefrontal cortex" are treated as the same entity.
ABBREVIATIONS: dict[str, str] = {
    "PFC": "prefrontal cortex",
    "ACC": "anterior cingulate cortex",
    "mPFC": "medial prefrontal cortex",
    "dlPFC": "dorsolateral prefrontal cortex",
    "vmPFC": "ventromedial prefrontal cortex",
    "OFC": "orbitofrontal cortex",
    "HPC": "hippocampus",
    "BLA": "basolateral amygdala",
    "NAc": "nucleus accumbens",
    "VTA": "ventral tegmental area",
    "LC": "locus coeruleus",
    "DRN": "dorsal raphe nucleus",
    "SNc": "substantia nigra pars compacta",
    "SNr": "substantia nigra pars reticulata",
    "STN": "subthalamic nucleus",
    "DA": "dopamine",
    "5-HT": "serotonin",
    "NE": "norepinephrine",
    "ACh": "acetylcholine",
    "BDNF": "brain-derived neurotrophic factor",
    "NGF": "nerve growth factor",
    "fMRI": "functional magnetic resonance imaging",
    "EEG": "electroencephalography",
    "LFP": "local field potential",
    "TMS": "transcranial magnetic stimulation",
    "PD": "parkinson's disease",
    "AD": "alzheimer's disease",
    "MS": "multiple sclerosis",
    "MDD": "major depressive disorder",
    "PTSD": "post-traumatic stress disorder",
    "OCD": "obsessive-compulsive disorder",
    "ADHD": "attention-deficit/hyperactivity disorder",
}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)


def normalize_span(text: str) -> str:
    """Lowercase, expand known abbreviations, and collapse whitespace."""
    expanded = _ABBREV_PATTERN.sub(lambda m: ABBREVIATIONS[m.group(0)], text)
    return " ".join(expanded.lower().split())


def normalize_sentence(text: str) -> str:
    """Normalize a sentence for comparison.

    Strips StructSense chunk-prefix artifacts of the form "Section Name\\n['" that
    appear in the occurrences[].sentence field, then lowercases and collapses whitespace.
    API sentences are plain text and pass through unchanged.
    """
    if "\n" in text:
        text = re.sub(r"^[^\n]*\n\[?'?", "", text)
        text = re.sub(r"'?\]?\s*$", "", text)
    return " ".join(text.lower().split())


def sentence_fingerprint(sentence: str, n: int = 8) -> str:
    """First n words of a normalized sentence — cheap, stable position proxy."""
    return " ".join(sentence.split()[:n])


# ---------------------------------------------------------------------------
# Layer 1A — pre-alignment
# ---------------------------------------------------------------------------


@dataclass
class Layer1AResult:
    paper_id: str
    jaccard: float
    structsense_only: list[SpanKey]
    api_only: list[SpanKey]
    shared_count: int
    label_agreement_rate: float     # fraction of shared spans where canonical labels agree
    label_disagreements: list[dict] # shared spans where canonical labels differ


def compute_layer1a(
    paper_id: str,
    structsense_entities: list[dict],
    api_entities: list[dict],
) -> Layer1AResult:
    """Compare entity spans and labels before the StructSense alignment stage (Layer 1A).

    Both systems assign free-form labels. Label agreement is determined by running both
    labels through _canonicalize_label() and comparing the results.

    Entity filtering: spans that pass ner_eval.is_excluded() are dropped before comparison.

    Args:
        paper_id: Identifier for the paper being evaluated.
        structsense_entities: Extractor-stage output. Each dict needs {"entity", "label"}.
        api_entities: Direct API output. Each dict needs {"entity", "label"}.

    Returns:
        Layer1AResult with Jaccard overlap, label agreement rate, and disagreement records.
    """
    ss_map = _build_entity_map(structsense_entities)
    api_map = _build_entity_map(api_entities)

    ss_spans = set(ss_map)
    api_spans = set(api_map)
    intersection = ss_spans & api_spans
    union = ss_spans | api_spans

    jaccard = len(intersection) / len(union) if union else 0.0

    agreements = 0
    disagreements: list[dict] = []
    for span in intersection:
        ss_canon = _canonicalize_label(ss_map[span])
        api_canon = _canonicalize_label(api_map[span])
        if ss_canon == api_canon:
            agreements += 1
        else:
            disagreements.append(
                {
                    "paper_id": paper_id,
                    "entity_normalized": span[0],
                    "sentence_fingerprint": span[1],
                    "structsense_label": ss_map[span],
                    "structsense_canonical": ss_canon,
                    "api_label": api_map[span],
                    "api_canonical": api_canon,
                }
            )

    label_agreement_rate = agreements / len(intersection) if intersection else 0.0

    return Layer1AResult(
        paper_id=paper_id,
        jaccard=jaccard,
        structsense_only=sorted(ss_spans - api_spans),
        api_only=sorted(api_spans - ss_spans),
        shared_count=len(intersection),
        label_agreement_rate=label_agreement_rate,
        label_disagreements=disagreements,
    )


# ---------------------------------------------------------------------------
# Layer 1B — post-alignment
# ---------------------------------------------------------------------------


@dataclass
class Layer1BResult:
    paper_id: str
    jaccard_high_conf: float
    structsense_only: list[SpanKey]
    api_only: list[SpanKey]
    shared_count: int
    label_agreement_rate: float
    label_disagreements: list[dict]
    low_conf_count: int      # StructSense entities excluded due to judge_score < threshold
    total_structsense_count: int
    low_conf_rate: float     # low_conf_count / total_structsense_count


def compute_layer1b(
    paper_id: str,
    structsense_entities: list[dict],
    api_entities: list[dict],
    judge_threshold: float = JUDGE_SCORE_THRESHOLD,
) -> Layer1BResult:
    """Compare entity spans and labels using the full StructSense post-alignment output (Layer 1B).

    Entities with judge_score < threshold are excluded from span comparison and counted separately.
    Label agreement uses the same canonical normalization as Layer 1A.

    For StructSense entities the label used is the extractor label (result["label"]).
    The ontology field is available but not used here — this layer compares label agreement
    in the StructSense label space, not ontology space, keeping it directly comparable to Layer 1A.

    Args:
        paper_id: Identifier for the paper being evaluated.
        structsense_entities: Full pipeline output. Each dict needs {"entity", "label", "judge_score"}.
        api_entities: Direct API output. Each dict needs {"entity", "label"}.
        judge_threshold: Minimum judge_score to include in span comparison (default 0.8).

    Returns:
        Layer1BResult with high-confidence Jaccard, label agreement, and low-confidence rate.
    """
    total = len(structsense_entities)
    high_conf = [e for e in structsense_entities if (e.get("judge_score") or 0) >= judge_threshold]
    low_conf_count = total - len(high_conf)

    ss_map = _build_entity_map(high_conf)
    api_map = _build_entity_map(api_entities)

    ss_spans = set(ss_map)
    api_spans = set(api_map)
    intersection = ss_spans & api_spans
    union = ss_spans | api_spans

    jaccard = len(intersection) / len(union) if union else 0.0

    agreements = 0
    disagreements: list[dict] = []
    for span in intersection:
        ss_canon = _canonicalize_label(ss_map[span])
        api_canon = _canonicalize_label(api_map[span])
        if ss_canon == api_canon:
            agreements += 1
        else:
            disagreements.append(
                {
                    "paper_id": paper_id,
                    "entity_normalized": span[0],
                    "sentence_fingerprint": span[1],
                    "structsense_label": ss_map[span],
                    "structsense_canonical": ss_canon,
                    "api_label": api_map[span],
                    "api_canonical": api_canon,
                }
            )

    label_agreement_rate = agreements / len(intersection) if intersection else 0.0
    low_conf_rate = low_conf_count / total if total else 0.0

    return Layer1BResult(
        paper_id=paper_id,
        jaccard_high_conf=jaccard,
        structsense_only=sorted(ss_spans - api_spans),
        api_only=sorted(api_spans - ss_spans),
        shared_count=len(intersection),
        label_agreement_rate=label_agreement_rate,
        label_disagreements=disagreements,
        low_conf_count=low_conf_count,
        total_structsense_count=total,
        low_conf_rate=low_conf_rate,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_entity_map(entities: list[dict]) -> dict[SpanKey, str]:
    """Build a (normalized_span, sentence_fingerprint) → label map.

    Keying on sentence fingerprint grounds each entity to a specific mention in the
    paper rather than just its surface form, enabling position-aware comparison.

    StructSense entities carry an occurrences list; each occurrence (potentially from
    a different sentence) becomes its own key so duplicate mentions in different
    sentences are preserved rather than collapsed.

    API entities carry a top-level sentence field and produce one key per entity dict.

    NOTE: if the same (entity, sentence) pair appears more than once the last label
    wins. True within-sentence duplicates are a known issue tracked separately.
    """
    result: dict[SpanKey, str] = {}
    for e in entities:
        text = e.get("entity", "")
        label = e.get("label", "")
        excluded, _ = is_excluded(text)
        if excluded:
            continue
        norm = normalize_span(text)
        if not norm:
            continue

        occurrences = e.get("occurrences")
        if occurrences:
            # StructSense: expand each occurrence into its own (entity, sentence) key
            for occ in occurrences:
                fp = sentence_fingerprint(normalize_sentence(occ.get("sentence", "")))
                result[(norm, fp)] = label
        else:
            # Direct API: use top-level sentence field
            fp = sentence_fingerprint(normalize_sentence(e.get("sentence", "")))
            result[(norm, fp)] = label

    return result
