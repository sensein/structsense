#!/usr/bin/env python3
"""
NER Evaluation Script for StructSense.

Evaluates entity-label correctness in NER extraction results.
Fully automated: filters junk entities, then scores label correctness
using a baked-in heuristic dictionary built from cross-model consensus.

Usage:
    python ner_eval.py <input.json>                  # Evaluate single file
    python ner_eval.py <input.json> -o report.json   # Save detailed report
    python ner_eval.py <dir>                          # Evaluate all nhil JSONs in dir tree
    python ner_eval.py <input.json> --verbose         # Show per-entity decisions
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
# EXCLUDE RULES — entities that should never be evaluated
# =============================================================================

# Single characters and Greek letters
_SINGLE_CHARS = set("abcdefghijklmnopqrstuvwxyz") | {
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "ℝ", "ẋ", "ẏ",
}

# 2-char entities that are noise (not valid gene/region abbreviations)
_NOISE_2CHAR = {
    "rn", "de", "rt", "ns", "ms", "mt", "se", "rf", "or", "as",
    "sc", "f1", "bi", "pc", "un", "im", "so", "pl", "go", "gr",
    "ca", "eq", "nn", "ij", "fs", "fm", "no", "it",
}

# Stopwords and generic terms
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "we", "our", "they", "their",
    "each", "all", "both", "other", "such", "than", "also", "only",
    "can", "may", "will", "would", "should", "could", "here", "there",
    "where", "when", "how", "what", "which", "who", "whom",
    "not", "nor", "but", "and", "or", "if", "then", "so", "as",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "between", "under", "same", "different", "new", "old",
    "well", "e.g", "i.e", "etc", "vs", "using", "used", "based",
    "use", "given", "show", "shown", "see", "found", "first", "second",
    "across", "within", "however", "therefore", "thus", "specifically",
    "respectively", "additionally", "overall", "including",
    "corresponding", "particular", "unique",
}

# Generic adjectives/quantifiers
_GENERIC_ADJ = {
    "single", "multiple", "several", "various", "specific", "general",
    "total", "full", "complete", "partial", "main", "major", "minor",
    "key", "important", "significant", "similar", "distinct", "common",
    "rare", "standard", "alternative", "additional", "potential",
    "possible", "true", "false", "positive", "negative", "final",
    "initial", "primary", "secondary", "relative", "absolute",
    "average", "mean", "maximum", "minimum", "optimal",
    "high", "low", "large", "small",
}

# Generic meta/document terms (not named entities)
_GENERIC_META = {
    "data", "model", "models", "method", "methods", "approach",
    "analysis", "results", "study", "studies", "work", "paper",
    "figure", "table", "section", "level", "levels", "type", "types",
    "set", "sets", "group", "groups", "case", "cases", "step", "steps",
    "number", "value", "values", "sample", "samples", "feature",
    "features", "pattern", "patterns", "structure", "structures",
    "system", "systems", "process", "function", "input", "output",
    "task", "note", "example", "comparison", "effect", "effects",
    "performance", "quality", "resolution", "information", "details",
    "framework", "pipeline", "network", "parameter", "parameters",
    "setting", "settings", "condition", "conditions", "state", "states",
    "component", "components", "version", "test", "fig", "fig.",
    "supplementary", "et al", "et al.", "types",
}

# Figure/table reference patterns
_FIGURE_RE = re.compile(
    r"^(fig|figure|table|tab|suppl|supplementary|extended data fig|"
    r"supplementary fig|supplementary figure|supplementary table)"
    r"[s.]?\s*[\d\w.,\-–—()/ ]*$",
    re.IGNORECASE,
)

# Pure numeric values (integers, floats, percentages, ranges, scientific notation)
_NUMERIC_RE = re.compile(
    r"^[\d.,\s%eE+\-–—/×]+$"
)

# Citation-like numbers (comma-separated reference numbers)
_CITATION_RE = re.compile(
    r"^\d[\d,\s]+$"
)

# Equation references
_EQUATION_RE = re.compile(
    r"^(eq|eqs)\.?\s*[\(\[\d,\s\)–\-and ]*$",
    re.IGNORECASE,
)

# Punctuation-only artifacts
_PUNCT_RE = re.compile(
    r"^[\[\]\(\)\{\}\-_.,;:!?\s/]+$"
)

# URL patterns
_URL_RE = re.compile(r"^https?:", re.IGNORECASE)


def is_excluded(entity: str) -> tuple[bool, str]:
    """Check if an entity should be excluded. Returns (excluded, reason)."""
    ent = entity.strip()
    ent_lower = ent.lower()

    if len(ent) == 0:
        return True, "empty"
    if ent_lower in _SINGLE_CHARS:
        return True, "single_char"
    if len(ent_lower) == 2 and ent_lower in _NOISE_2CHAR:
        return True, "noise_2char"
    if _PUNCT_RE.match(ent):
        return True, "punctuation"
    if _NUMERIC_RE.match(ent):
        return True, "numeric"
    if _CITATION_RE.match(ent):
        return True, "citation_number"
    if _FIGURE_RE.match(ent_lower):
        return True, "figure_table_ref"
    if _EQUATION_RE.match(ent_lower):
        return True, "equation_ref"
    if _URL_RE.match(ent):
        return True, "url"
    if ent_lower in _STOPWORDS:
        return True, "stopword"
    if ent_lower in _GENERIC_ADJ:
        return True, "generic_adjective"
    if ent_lower in _GENERIC_META:
        return True, "generic_meta_term"

    return False, ""


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
# LABEL BLACKLIST — labels that are almost never correct for neuroscience NER
# =============================================================================

_LABEL_BLACKLIST = {
    # Clinical NER labels from d4data/biomedical-ner-all, misapplied
    "Diagnostic_procedure",
    "Sign_symptom",
    "Coreference",
    "Detailed_description",
    "Lab_value",
    "Disease_disorder",
    "Severity",
    "Diagnostic_info",
    "Quantitative_concept",
    "Duration",
    # Vague catch-all labels
    "ENTITY",
    "MISC",
    "OTHER",
    "O",
    "BIO",
    "bio",
    "UNKNOWN",
    "NOUN",
    "Miscellaneous",
    "Other",
}


# =============================================================================
# HEURISTIC DICTIONARY — entity → expected label
# Built from cross-model consensus (70%+ agreement, seen 2+ times)
# =============================================================================

# Exact entity→label mappings (case-insensitive lookup)
_ENTITY_LABEL_MAP: dict[str, str] = {
    # --- CELL_TYPE ---
    "cell types": "CELL_TYPE",
    "cell type": "CELL_TYPE",
    "msns": "CELL_TYPE",
    "astrocytes": "CELL_TYPE",
    "astrocyte": "CELL_TYPE",
    "interneurons": "CELL_TYPE",
    "gabaergic neurons": "CELL_TYPE",
    "glial": "CELL_TYPE",
    "lamp5 interneurons": "CELL_TYPE",
    "oligodendrocyte lineage": "CELL_TYPE",
    "cop": "CELL_TYPE",
    "imoligo": "CELL_TYPE",
    "hematopoietic stem cells": "CELL_TYPE",
    "cholinergic interneurons": "CELL_TYPE",
    "astrocyte subtypes": "CELL_TYPE",
    "target cell types": "CELL_TYPE",
    "opc-oligo": "CELL_TYPE",
    "vasoactive intestinal polypeptide-expressing inhibitory neurons": "CELL_TYPE",
    "parvalbumin-expressing inhibitory neurons": "CELL_TYPE",
    "layer 4 excitatory neurons": "CELL_TYPE",
    "hillock-like": "CELL_TYPE",
    "d1": "CELL_TYPE",
    "d2": "CELL_TYPE",
    "l4": "CELL_TYPE",
    "l5": "CELL_TYPE",
    "l3": "CELL_TYPE",
    # Specific cell subtypes from papers
    "gpe meis2-sox6": "CELL_TYPE",
    "cn lhx6-gbx1": "CELL_TYPE",
    "oligo plekhg1": "CELL_TYPE",
    "oligo opalin": "CELL_TYPE",
    "str d1d2 hybrid": "CELL_TYPE",
    "strd d2 hybrid": "CELL_TYPE",
    "gpin-bf cholinergic": "CELL_TYPE",
    "gpe sox6-ctxnd1": "CELL_TYPE",
    "tac3-plpp4 interneurons": "CELL_TYPE",

    # --- BRAIN_REGION ---
    "putamen": "BRAIN_REGION",
    "caudate nucleus": "BRAIN_REGION",
    "nucleus accumbens": "BRAIN_REGION",
    "nac": "BRAIN_REGION",
    "nac shell": "BRAIN_REGION",
    "nac core": "BRAIN_REGION",
    "prefrontal cortex": "BRAIN_REGION",
    "white matter": "BRAIN_REGION",
    "striatopallidal fiber tracts": "BRAIN_REGION",
    "dorsolateral": "BRAIN_REGION",
    "dorsal striatum": "BRAIN_REGION",
    "internal capsule white matter": "BRAIN_REGION",
    "gpi nucleus": "BRAIN_REGION",
    "caudateputamen": "BRAIN_REGION",
    "ventromedial": "BRAIN_REGION",
    "brain areas": "BRAIN_REGION",
    "gp": "BRAIN_REGION",
    "basal ganglia": "BRAIN_REGION",
    "striatum": "BRAIN_REGION",
    "globus pallidus": "BRAIN_REGION",
    "cortex": "BRAIN_REGION",
    "hippocampus": "BRAIN_REGION",
    "thalamus": "BRAIN_REGION",
    "amygdala": "BRAIN_REGION",
    "cerebellum": "BRAIN_REGION",

    # --- GENE ---
    "gene": "GENE",
    "drd1": "GENE",
    "drd2": "GENE",
    "tac1": "GENE",
    "tac3": "GENE",
    "plpp4": "GENE",
    "aqp4": "GENE",
    "chi3l1": "GENE",
    "kirrel3": "GENE",
    "pde10a": "GENE",
    "nrgn": "GENE",
    "nnat": "GENE",
    "crym": "GENE",
    "rap1gap": "GENE",
    "sox6": "GENE",
    "fam83d": "GENE",
    "opalin": "GENE",
    "penk": "GENE",
    "gata3": "GENE",
    "tcf7l2": "GENE",
    "dopamine receptor d1": "GENE",
    "dopamine receptor d2": "GENE",
    "tachykinin precursor 1": "GENE",
    "proenkephalin": "GENE",
    "th": "GENE",
    "pv": "GENE",

    # --- CHEMICAL ---
    "methanol": "CHEMICAL",
    "ethanol": "CHEMICAL",
    "gaba": "CHEMICAL",
    "pbs": "CHEMICAL",
    "pfa": "CHEMICAL",
    "sds": "CHEMICAL",
    "naoh": "CHEMICAL",
    "edta": "CHEMICAL",
    "hcl": "CHEMICAL",
    "triethylamine": "CHEMICAL",
    "allyl trichlorosilane": "CHEMICAL",
    "chloroform": "CHEMICAL",
    "poly-d-lysine": "CHEMICAL",
    "oct": "CHEMICAL",
    "tween-20": "CHEMICAL",
    "dopamine": "CHEMICAL",
    "glutamate": "CHEMICAL",
    "serotonin": "CHEMICAL",
    "acetylcholine": "CHEMICAL",

    # --- DISEASE ---
    "parkinsons disease": "DISEASE",
    "huntingtons disease": "DISEASE",
    "neurodegenerative disease": "DISEASE",
    "alzheimers disease": "DISEASE",

    # --- SPECIES ---
    "non-human primate": "SPECIES",
    "primate": "SPECIES",
    "human": "SPECIES",
    "mouse": "SPECIES",
    "macaque": "SPECIES",
    "rat": "SPECIES",
    "mice": "SPECIES",

    # --- ORGANIZATION ---
    "sigmaaldrich": "ORGANIZATION",
    "scipy": "ORGANIZATION",
    "thermo fisher scientific": "ORGANIZATION",
    "zymo research": "ORGANIZATION",

    # --- METHOD ---
    "umap": "METHOD",
    "ns-forest workflow": "METHOD",
    "next-generation sequencing": "METHOD",
    "spatially resolved transcriptomics": "METHOD",
    "uniform manifold approximation and projection": "METHOD",
    "immunohistochemistry": "METHOD",
    "in situ sequencing": "METHOD",
    "leiden clustering": "METHOD",
    "leiden clustering algorithm": "METHOD",
    "spatial module analysis": "METHOD",
    "wilcoxon rank sum method": "METHOD",
    "probe-based multiplexed fluorescent in situ hybridization": "METHOD",
    "radius_neighbors_graph": "METHOD",
    "unsupervised methods": "METHOD",
    "correlation-based methods": "METHOD",
    "metropolis hastings optimization algorithm": "METHOD",
    "wieners algorithm": "METHOD",
    "testing method": "METHOD",
    "in vivo electrophysiological": "METHOD",
    "connectivity assays": "METHOD",

    # --- METRIC ---
    "positive predictive value": "METRIC",
    "enrichment score": "METRIC",
    "ppv": "METRIC",
    "tp": "METRIC",
    "fp": "METRIC",
    "tn": "METRIC",
    "fn": "METRIC",
    "f-beta score": "METRIC",
    "on-target fraction": "METRIC",
    "binary expression score": "METRIC",
    "recall": "METRIC",
    "precision": "METRIC",
    "f1 score": "METRIC",
    "accuracy": "METRIC",
    "sensitivity": "METRIC",
    "specificity": "METRIC",

    # --- ANATOMICAL_STRUCTURE (→ BRAIN_REGION canonical) ---
    "anatomical domains": "BRAIN_REGION",
    "white matter tracts": "BRAIN_REGION",
    "human basal ganglia": "BRAIN_REGION",
    "gpi": "BRAIN_REGION",
    "gpe": "BRAIN_REGION",
    "pfc": "BRAIN_REGION",
    "mtg": "BRAIN_REGION",
    "lung": "BRAIN_REGION",
    "kidney": "BRAIN_REGION",
    "striosome": "BRAIN_REGION",

    # --- MODEL/COMPUTATIONAL ---
    "rnn": "MODEL",
    "rnns": "MODEL",
    "latent circuit model": "MODEL",
    "latent circuit": "MODEL",
    "random forest": "METHOD",

    # --- SOFTWARE/TOOL ---
    "ns-forest": "METHOD",  # acceptable as both METHOD and SOFTWARE
    "ns-forest v4.0": "SOFTWARE",
    "binaryfirst": "METHOD",  # acceptable as both METHOD and SOFTWARE
    "binaryfirst_high": "METHOD",
    "binaryfirst_moderate": "METHOD",
    "cellref": "SOFTWARE",
    "azimuth": "SOFTWARE",
    "merfish": "METHOD",
    "stereo-seq": "METHOD",
    "scrna-seq": "METHOD",
    "scrna seq": "METHOD",

    # --- GENE (additional) ---
    "genes": "GENE",
    "marker genes": "GENE",
    "marker gene": "GENE",
    "vip": "GENE",
    "pvalb": "GENE",

    # --- DATASET ---
    "hlca": "DATASET",
    "asctb": "DATASET",
    "human cell atlas": "DATASET",
    "allen human brain atlas": "DATASET",

    # --- CONCEPT/PROCESS ---
    "connectivity": "CONCEPT",
    "circuit mechanisms": "CONCEPT",
    "context-dependent decision-making": "CONCEPT",
    "monkeys": "SPECIES",

    # --- Entities that are both BRAIN_REGION and SPECIES-qualified ---
    "human brain": "BRAIN_REGION",
    "mouse brain": "BRAIN_REGION",
}

# Entities where multiple labels are acceptable (entity_lower → set of canonical labels)
_ENTITY_MULTI_LABELS: dict[str, set[str]] = {
    "ns-forest": {"METHOD", "SOFTWARE"},
    "ns-forest v4.0": {"METHOD", "SOFTWARE"},
    "binaryfirst": {"METHOD", "SOFTWARE"},
    "binaryfirst_high": {"METHOD", "SOFTWARE", "PARAMETER"},
    "binaryfirst_moderate": {"METHOD", "SOFTWARE", "PARAMETER"},
    "vip": {"GENE", "CELL_TYPE"},
    "pvalb": {"GENE", "CELL_TYPE"},
    "d1": {"CELL_TYPE", "GENE"},
    "d2": {"CELL_TYPE", "GENE"},
    "human cell atlas": {"ORGANIZATION", "DATASET"},
    "human biomolecular atlas program": {"ORGANIZATION", "DATASET"},
    "allen human brain atlas": {"ORGANIZATION", "DATASET"},
    "asctb": {"DATABASE", "DATASET"},
    "hlca": {"DATABASE", "DATASET"},
    "lung": {"BRAIN_REGION", "ORGAN", "TISSUE"},
    "kidney": {"BRAIN_REGION", "ORGAN", "TISSUE"},
    "basal ganglia": {"BRAIN_REGION", "BIOLOGICAL_STRUCTURE"},
    "brain": {"BRAIN_REGION", "BIOLOGICAL_STRUCTURE"},
    "human brain": {"BRAIN_REGION", "BIOLOGICAL_STRUCTURE", "SPECIES"},
    "gene": {"GENE", "CONCEPT"},
    "genes": {"GENE", "CONCEPT"},
    "marker genes": {"GENE", "CONCEPT"},
    "marker gene": {"GENE", "CONCEPT"},
    "cell types": {"CELL_TYPE", "CONCEPT"},
    "cell type": {"CELL_TYPE", "CONCEPT"},
    "random forest": {"METHOD", "MODEL"},
    "rnn": {"MODEL", "METHOD"},
    "rnns": {"MODEL", "METHOD"},
    "umap": {"METHOD", "SOFTWARE"},
    "leiden clustering": {"METHOD", "SOFTWARE"},
    "gaba": {"CHEMICAL", "GENE"},
    "dopamine": {"CHEMICAL", "GENE"},
    "serotonin": {"CHEMICAL", "GENE"},
    "acetylcholine": {"CHEMICAL", "GENE"},
    "merfish": {"METHOD", "SOFTWARE"},
    "stereo-seq": {"METHOD", "SOFTWARE"},
    "scrna-seq": {"METHOD", "SOFTWARE"},
    "azimuth": {"SOFTWARE", "DATASET"},
    "cellref": {"SOFTWARE", "DATASET"},
    "scipy": {"ORGANIZATION", "SOFTWARE"},
}

# =============================================================================
# KEYWORD-BASED RULES — ordered by specificity (most specific first)
# Each rule: (keywords_or_pattern, expected_label)
# =============================================================================

_KEYWORD_RULES: list[tuple[re.Pattern | list[str], str]] = [
    # CELL_TYPE — keyword patterns
    (["neuron"], "CELL_TYPE"),
    (["interneuron"], "CELL_TYPE"),
    (["astrocyte"], "CELL_TYPE"),
    (["oligodendrocyte"], "CELL_TYPE"),
    (["microglia"], "CELL_TYPE"),
    (["inhibitory"], "CELL_TYPE"),
    (["excitatory"], "CELL_TYPE"),
    (["gabaergic"], "CELL_TYPE"),
    (["glutamatergic"], "CELL_TYPE"),
    (["cholinergic"], "CELL_TYPE"),
    (["dopaminergic"], "CELL_TYPE"),
    (["serotonergic"], "CELL_TYPE"),
    (["msn"], "CELL_TYPE"),
    (["stem cell"], "CELL_TYPE"),
    (["progenitor"], "CELL_TYPE"),
    (["endothelial"], "CELL_TYPE"),
    (["pericyte"], "CELL_TYPE"),
    (["macrophage"], "CELL_TYPE"),
    (["lymphocyte"], "CELL_TYPE"),
    (["fibroblast"], "CELL_TYPE"),
    (["epithelial"], "CELL_TYPE"),

    # BRAIN_REGION — keyword patterns
    (["cortex"], "BRAIN_REGION"),
    (["cortical"], "BRAIN_REGION"),
    (["striatum"], "BRAIN_REGION"),
    (["striatal"], "BRAIN_REGION"),
    (["hippocamp"], "BRAIN_REGION"),
    (["thalam"], "BRAIN_REGION"),
    (["amygdala"], "BRAIN_REGION"),
    (["cerebell"], "BRAIN_REGION"),
    (["nucleus"], "BRAIN_REGION"),
    (["pallidus"], "BRAIN_REGION"),
    (["putamen"], "BRAIN_REGION"),
    (["caudate"], "BRAIN_REGION"),
    (["accumbens"], "BRAIN_REGION"),
    (["prefrontal"], "BRAIN_REGION"),
    (["ventral"], "BRAIN_REGION"),
    (["dorsal"], "BRAIN_REGION"),

    # GENE — common gene name patterns (uppercase 2-6 chars often genes)
    (["receptor"], "GENE"),
    (["kinase"], "GENE"),
    (["transcription factor"], "GENE"),
    (["transporter"], "GENE"),

    # DISEASE
    (["disease"], "DISEASE"),
    (["disorder"], "DISEASE"),
    (["syndrome"], "DISEASE"),

    # SPECIES
    (["primate"], "SPECIES"),
    (["macaque"], "SPECIES"),

    # METHOD
    (["sequencing"], "METHOD"),
    (["clustering"], "METHOD"),
    (["hybridization"], "METHOD"),
    (["immunohistochemistry"], "METHOD"),
    (["staining"], "METHOD"),
    (["imaging"], "METHOD"),
    (["microscopy"], "METHOD"),
    (["transcriptomics"], "METHOD"),
    (["proteomics"], "METHOD"),
    (["algorithm"], "METHOD"),

    # SOFTWARE
    (["python"], "SOFTWARE"),
    (["scanpy"], "SOFTWARE"),
    (["seurat"], "SOFTWARE"),

    # CHEMICAL
    (["buffer"], "CHEMICAL"),
    (["solution"], "CHEMICAL"),
    (["reagent"], "CHEMICAL"),

    # METRIC
    (["score"], "METRIC"),
    (["fraction"], "METRIC"),
    (["coefficient"], "METRIC"),
    (["accuracy"], "METRIC"),
    (["precision"], "METRIC"),
    (["recall"], "METRIC"),
    (["f1"], "METRIC"),

    # DATASET
    (["atlas"], "DATASET"),
    (["database"], "DATASET"),
    (["dataset"], "DATASET"),

    # SPECIES
    (["mouse"], "SPECIES"),
    (["mice"], "SPECIES"),
    (["human"], "SPECIES"),
    (["rat"], "SPECIES"),
]


def _match_keyword_rule(entity_lower: str) -> str | None:
    """Try to match entity against keyword rules. Returns expected label or None."""
    for keywords_or_pattern, label in _KEYWORD_RULES:
        if isinstance(keywords_or_pattern, list):
            for kw in keywords_or_pattern:
                # Use word boundary matching to avoid substring false positives
                # e.g., "rat" shouldn't match "curation"
                if re.search(r'(?:^|[\s\-_/])' + re.escape(kw), entity_lower):
                    return label
        else:
            if keywords_or_pattern.search(entity_lower):
                return label
    return None


# =============================================================================
# SCORING LOGIC
# =============================================================================

def _normalize_label(label: str) -> str:
    """Normalize label for comparison (uppercase, underscores)."""
    return label.strip().upper().replace(" ", "_").replace("-", "_")


# Label aliases — different names for the same concept
_LABEL_ALIASES: dict[str, str] = {
    # CELL_TYPE variants
    "CELL": "CELL_TYPE",
    "CELLTYPE": "CELL_TYPE",
    "CELL_TYPE_MARKER": "CELL_TYPE",
    "CELL_TYPE_SUBGROUP": "CELL_TYPE",
    "CELL_CLUSTER": "CELL_TYPE",
    "CELL_SUBCLASS": "CELL_TYPE",
    "CELL_TYPE_": "CELL_TYPE",
    "BIOLOGICAL_UNIT": "CELL_TYPE",
    # GENE variants
    "GENE_OR_GENE_PRODUCT": "GENE",
    "GENE_OR_PROTEIN": "GENE",
    "GENE_MARKER": "GENE",
    "GENE_CATEGORY": "GENE",
    "GENETIC": "GENE",
    "GENETIC_MARKER": "GENE",
    "GENETIC_TECH": "GENE",
    "GENETIC_METRIC": "GENE",
    "PROTEIN": "GENE",
    "GENE_SET": "GENE",
    "GENE_EXPRESSION_PATTERN": "GENE",
    "BIOLOGICAL_MARKER": "GENE",
    "BIOLOGICAL_MOLECULE": "GENE",
    # BRAIN_REGION variants
    "BRAIN_AREA": "BRAIN_REGION",
    "BRAIN_REGION_": "BRAIN_REGION",
    "ANATOMY": "BRAIN_REGION",
    "ANATOMICAL_STRUCTURE": "BRAIN_REGION",
    "BIOLOGICAL_STRUCTURE": "BRAIN_REGION",
    "ORGAN": "BRAIN_REGION",
    "TISSUE": "BRAIN_REGION",
    # METHOD variants
    "TECHNIQUE": "METHOD",
    "METHODOLOGY": "METHOD",
    "EXPERIMENTAL_TECHNIQUE": "METHOD",
    "STATISTICAL_METHOD": "METHOD",
    "ALGORITHM": "METHOD",
    "ALGORITHM_VERSION": "METHOD",
    "ALGORITHM_STEP": "METHOD",
    "ALGORITHM_STRATEGY": "METHOD",
    "ALGORITHM_MODULE": "METHOD",
    "STRATEGY": "METHOD",
    "EXPERIMENTAL_CONFIGURATION": "METHOD",
    "THRESHOLD_METHOD": "METHOD",
    "STATISTICAL_TEST": "METHOD",
    "DIAGNOSTIC_PROCEDURE": "METHOD",
    "DIAGNOSTIC_METRIC": "METHOD",
    "DIAGNOSTIC_VISUALIZATION": "METHOD",
    # SOFTWARE variants
    "PROGRAMMING_LANGUAGE": "SOFTWARE",
    "SOFTWARE_TOOL": "SOFTWARE",
    "SOFTWARE_LIBRARY": "SOFTWARE",
    "SOFTWARE_VERSION": "SOFTWARE",
    "TOOL": "SOFTWARE",
    "COMPUTATIONAL_SYSTEM": "SOFTWARE",
    # SPECIES variants
    "ORGANISM": "SPECIES",
    # ORGANIZATION variants
    "ORGANIZATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    # DATASET/DATABASE
    "DATASET": "DATASET",
    "DATASET_LEVEL": "DATASET",
    "DATABASE": "DATABASE",
    "RESOURCE": "DATASET",
    # CHEMICAL
    "CHEMICAL": "CHEMICAL",
    # DISEASE
    "DISEASE": "DISEASE",
    # METRIC variants
    "METRIC": "METRIC",
    "MEASUREMENT": "METRIC",
    "STATISTICAL_METRIC": "METRIC",
    "STATISTICAL_DESCRIPTOR": "METRIC",
    "STATISTICAL_MEASURE": "METRIC",
    "QUANTITATIVE_VALUE": "METRIC",
    # SPECIES
    "SPECIES": "SPECIES",
    # DATA_TYPE / DATA_STRUCTURE
    "DATA_TYPE": "DATA_TYPE",
    "DATA_STRUCTURE": "DATA_TYPE",
    "DATA_ATTR": "DATA_TYPE",
    "DAT_STRUCTURE": "DATA_TYPE",
    "FILE_FORMAT": "DATA_TYPE",
    # BIOLOGICAL_PROCESS
    "BIOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
    "PROCESS": "BIOLOGICAL_PROCESS",
    # MODEL
    "MODEL": "MODEL",
    "MODEL_ARCHITECTURE": "MODEL",
    "MODEL_COMPONENT": "MODEL",
    # PARAMETER/CONFIGURATION
    "PARAMETER": "PARAMETER",
    "PARAMETER_LEVEL": "PARAMETER",
    "METHOD_PARAMETER": "PARAMETER",
    "CONFIGURATION": "PARAMETER",
    "THRESHOLD_SETTING": "PARAMETER",
    "THRESHOLD": "PARAMETER",
    # PERSON
    "PERSON": "PERSON",
    # FIELD_OF_STUDY
    "FIELD_OF_STUDY": "FIELD_OF_STUDY",
    "KEY_TERM": "CONCEPT",
    "CONCEPT": "CONCEPT",
    # TECHNOLOGY
    "TECHNOLOGY": "METHOD",
    # VERSION
    "VERSION": "VERSION",
    # FIGURE/VISUALIZATION
    "FIGURE": "FIGURE",
    "FIGURE_REF": "FIGURE",
    "VISUALIZATION": "FIGURE",
    "DOCUMENT_REFERENCE": "FIGURE",
    # CITATION
    "CITATION": "CITATION",
    "REFERENCE": "CITATION",
    # CONCEPT (broad)
    "CONCEPT": "CONCEPT",
    "KEY_TERM": "CONCEPT",
    "SCIENTIFIC_TERM": "CONCEPT",
    "SCIENTIFICTERM": "CONCEPT",
    "BIOMEDICAL_ENTITY": "CONCEPT",
    "BIOLOGICALENTITY": "CONCEPT",
    "BIOENTITY": "CONCEPT",
    # BIOLOGICAL_PROCESS
    "BIOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
    "PROCESS": "BIOLOGICAL_PROCESS",
    "BIOLOGICALPROCESS": "BIOLOGICAL_PROCESS",
    # PRODUCT/MISC → ambiguous, leave as-is for now
    "PRODUCT": "PRODUCT",
    "TASK": "CONCEPT",
    "EVENT": "CONCEPT",
    "STIMULUS": "STIMULUS",
    "BEHAVIOR": "BEHAVIOR",
    "COLOR": "CONCEPT",
    "MATERIAL": "MATERIAL",
    "DEVICE": "DEVICE",
    "MOLECULE": "MOLECULE",
    "MOLECULAR_TOOL": "MOLECULE",
    "CELLULAR_COMPONENT": "CELLULAR_COMPONENT",
    "NEURAL_COMPONENT": "NEURAL_COMPONENT",
    "MATHEMATICAL_OBJECT": "MATHEMATICAL_OBJECT",
    "NEUROANATOMY": "BRAIN_REGION",
    "ANATOMICAL_ENTITY": "BRAIN_REGION",
    "SYSTEM": "SYSTEM",
    "SECTION": "CONCEPT",
    "DATA": "DATA_TYPE",
    "MEASURE": "METRIC",
    "VARIABLE": "VARIABLE",
    "PERSON": "PERSON",
}

# Canonical labels that we consider "valid" for neuroscience NER
# If a label canonicalizes to one of these, and we have no conflicting evidence,
# we trust it as "likely correct"
_VALID_CANONICAL_LABELS = {
    "CELL_TYPE", "BRAIN_REGION", "GENE", "CHEMICAL", "DISEASE", "SPECIES",
    "METHOD", "SOFTWARE", "ORGANIZATION", "DATASET", "DATABASE", "METRIC",
    "DATA_TYPE", "BIOLOGICAL_PROCESS", "MODEL", "PARAMETER",
    "FIELD_OF_STUDY", "CONCEPT", "PERSON", "VARIABLE",
    "STIMULUS", "BEHAVIOR", "MATERIAL", "DEVICE",
    "MOLECULE", "CELLULAR_COMPONENT", "NEURAL_COMPONENT",
    "MATHEMATICAL_OBJECT", "NEUROANATOMY", "SYSTEM",
    "FIGURE", "CITATION", "PRODUCT",
}


# Prefix-based fallback rules: if normalized label starts with prefix → canonical
_LABEL_PREFIX_RULES: list[tuple[str, str]] = [
    ("NEURAL_", "CONCEPT"),
    ("NEURO", "CONCEPT"),
    ("BRAIN_", "BRAIN_REGION"),
    ("ANATOMICAL", "BRAIN_REGION"),
    ("CELL_TYPE", "CELL_TYPE"),
    ("CELL_", "CELL_TYPE"),
    ("GENE_", "GENE"),
    ("GENETIC", "GENE"),
    ("BIOLOGICAL_", "BIOLOGICAL_PROCESS"),
    ("BIOMEDICAL_", "CONCEPT"),
    ("BIOLINK:", "CONCEPT"),
    ("STATISTICAL_", "METRIC"),
    ("MATHEMATICAL_", "CONCEPT"),
    ("MATH_", "CONCEPT"),
    ("COMPUTATIONAL_", "METHOD"),
    ("EXPERIMENTAL_", "METHOD"),
    ("SOFTWARE_", "SOFTWARE"),
    ("BEHAVIORAL_", "CONCEPT"),
    ("COGNITIVE_", "CONCEPT"),
    ("MOLECULAR_", "MOLECULE"),
    ("DATA_", "DATA_TYPE"),
    ("MODEL_", "MODEL"),
    ("ALGORITHM_", "METHOD"),
]


def _canonicalize_label(label: str) -> str:
    """Map a label to its canonical form."""
    norm = _normalize_label(label)
    # Exact alias match
    canon = _LABEL_ALIASES.get(norm)
    if canon:
        return canon
    # Prefix-based fallback
    for prefix, canon_label in _LABEL_PREFIX_RULES:
        if norm.startswith(prefix):
            return canon_label
    return norm


def score_entity(entity_text: str, assigned_label: str) -> tuple[str, str, str]:
    """
    Score an entity-label pair.

    Returns: (verdict, expected_label, reason)
        verdict: "correct", "incorrect", "unknown"
        expected_label: what we think the label should be (or "" if unknown)
        reason: explanation
    """
    ent_lower = entity_text.strip().lower()
    assigned_canon = _canonicalize_label(assigned_label)

    # Check multi-label entities first (these accept multiple correct labels)
    multi = _ENTITY_MULTI_LABELS.get(ent_lower)
    if multi:
        if assigned_canon in multi:
            return "correct", assigned_label, f"multi-label match (acceptable: {multi})"
        # Check if label blacklisted
        if assigned_label in _LABEL_BLACKLIST:
            best = sorted(multi)[0]
            return "incorrect", best, f"blacklisted label '{assigned_label}', expected one of {multi}"
        best = sorted(multi)[0]
        return "incorrect", best, f"expected one of {multi}, got '{assigned_label}'"

    # Check label blacklist
    if assigned_label in _LABEL_BLACKLIST:
        # The label itself is from a clinical model, almost certainly wrong
        # But we can still try to find the correct label
        expected = _ENTITY_LABEL_MAP.get(ent_lower)
        if expected:
            return "incorrect", expected, f"blacklisted label '{assigned_label}', expected '{expected}'"
        kw_label = _match_keyword_rule(ent_lower)
        if kw_label:
            return "incorrect", kw_label, f"blacklisted label '{assigned_label}', keyword suggests '{kw_label}'"
        return "incorrect", "", f"blacklisted label '{assigned_label}'"

    # Check exact dictionary match
    expected = _ENTITY_LABEL_MAP.get(ent_lower)
    if expected:
        expected_canon = _canonicalize_label(expected)
        if assigned_canon == expected_canon:
            return "correct", expected, "exact dictionary match"
        else:
            return "incorrect", expected, f"dictionary says '{expected}', got '{assigned_label}'"

    # Check keyword rules
    kw_label = _match_keyword_rule(ent_lower)
    if kw_label:
        kw_canon = _canonicalize_label(kw_label)
        if assigned_canon == kw_canon:
            return "correct", kw_label, "keyword rule match"
        else:
            return "incorrect", kw_label, f"keyword rule suggests '{kw_label}', got '{assigned_label}'"

    # Step 4: Trusted label fallback
    # If the assigned label canonicalizes to a known valid neuroscience category,
    # and we have no conflicting evidence, trust it as likely correct.
    if assigned_canon in _VALID_CANONICAL_LABELS:
        return "correct", assigned_label, f"trusted canonical label '{assigned_canon}'"

    # No heuristic matches — can't determine
    return "unknown", "", "no heuristic match"


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_file(filepath: str, verbose: bool = False) -> dict:
    """Evaluate a single NER result JSON file. Returns a report dict."""
    with open(filepath) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    report = {
        "file": filepath,
        "total_entities": len(entities),
        "excluded": {"count": 0, "reasons": Counter(), "entities": []},
        "source_filtered": {"count": 0, "entities": []},
        "evaluated": {
            "count": 0,
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "details": [],
        },
        "label_distribution": Counter(),
        "entity_label_conflicts": {},
    }

    # Track unique entities and their labels for conflict detection
    entity_labels = defaultdict(set)

    evaluated_entities = []

    for ent in entities:
        entity_text = ent["entity"]
        label = ent["label"]
        ent_lower = entity_text.strip().lower()

        report["label_distribution"][label] += 1
        entity_labels[ent_lower].add(label)

        # Step 1: Filter en_core_web_sm-only entities
        if is_only_en_core_web_sm(ent):
            report["source_filtered"]["count"] += 1
            report["source_filtered"]["entities"].append(entity_text)
            continue

        # Step 2: Apply exclude rules
        excluded, reason = is_excluded(entity_text)
        if excluded:
            report["excluded"]["count"] += 1
            report["excluded"]["reasons"][reason] += 1
            report["excluded"]["entities"].append(
                {"entity": entity_text, "reason": reason}
            )
            continue

        evaluated_entities.append(ent)

    # Step 3: Score remaining entities
    for ent in evaluated_entities:
        entity_text = ent["entity"]
        label = ent["label"]

        verdict, expected, reason = score_entity(entity_text, label)

        report["evaluated"]["count"] += 1
        report["evaluated"][verdict] += 1
        report["evaluated"]["details"].append({
            "entity": entity_text,
            "assigned_label": label,
            "verdict": verdict,
            "expected_label": expected,
            "reason": reason,
            "judge_score": ent.get("judge_score"),
            "ontology_id": ent.get("ontology_id", ""),
        })

    # Detect label conflicts (same entity, different labels)
    for ent_key, labels in entity_labels.items():
        if len(labels) > 1:
            report["entity_label_conflicts"][ent_key] = sorted(labels)

    return report


def print_report(report: dict, verbose: bool = False) -> None:
    """Print a human-readable report."""
    print(f"\n{'='*70}")
    print(f"  NER Evaluation: {os.path.basename(report['file'])}")
    print(f"{'='*70}")

    total = report["total_entities"]
    src_filt = report["source_filtered"]["count"]
    excluded = report["excluded"]["count"]
    evaled = report["evaluated"]["count"]
    correct = report["evaluated"]["correct"]
    incorrect = report["evaluated"]["incorrect"]
    unknown = report["evaluated"]["unknown"]

    print(f"\n  Total entities in file:     {total}")
    print(f"  Filtered (en_core_web_sm): -{src_filt}")
    print(f"  Excluded (junk/generic):   -{excluded}")
    print(f"  ─────────────────────────────────")
    print(f"  Evaluated:                  {evaled}")
    print(f"")

    if evaled > 0:
        print(f"  ✓ Correct:    {correct:4d}  ({correct/evaled*100:5.1f}%)")
        print(f"  ✗ Incorrect:  {incorrect:4d}  ({incorrect/evaled*100:5.1f}%)")
        print(f"  ? Unknown:    {unknown:4d}  ({unknown/evaled*100:5.1f}%)")

        scored = correct + incorrect
        if scored > 0:
            print(f"\n  Accuracy (excl. unknown): {correct/scored*100:.1f}% ({correct}/{scored})")

    # Exclude reasons breakdown
    if report["excluded"]["reasons"]:
        print(f"\n  Exclude reasons:")
        for reason, count in report["excluded"]["reasons"].most_common():
            print(f"    {reason:25s} {count:4d}")

    # Label conflicts
    conflicts = report["entity_label_conflicts"]
    if conflicts:
        print(f"\n  Entities with conflicting labels: {len(conflicts)}")
        if verbose:
            for ent, labels in sorted(conflicts.items()):
                print(f"    \"{ent}\": {labels}")

    # Incorrect details
    incorrect_items = [
        d for d in report["evaluated"]["details"] if d["verdict"] == "incorrect"
    ]
    if incorrect_items:
        print(f"\n  Incorrect labels ({len(incorrect_items)}):")
        # Group by reason type
        by_reason = defaultdict(list)
        for item in incorrect_items:
            key = "blacklisted" if "blacklisted" in item["reason"] else "heuristic_mismatch"
            by_reason[key].append(item)

        for category, items in by_reason.items():
            print(f"\n    [{category}] ({len(items)} entities):")
            for item in items[:20]:  # show first 20
                expected = item["expected_label"]
                exp_str = f", expected '{expected}'" if expected else ""
                print(
                    f"      \"{item['entity']}\" — "
                    f"got '{item['assigned_label']}'"
                    f"{exp_str}"
                )
            if len(items) > 20:
                print(f"      ... and {len(items)-20} more")

    if verbose:
        unknown_items = [
            d for d in report["evaluated"]["details"] if d["verdict"] == "unknown"
        ]
        if unknown_items:
            print(f"\n  Unknown entities ({len(unknown_items)}):")
            for item in unknown_items:
                print(f"    \"{item['entity']}\" → {item['assigned_label']}")

    print()


def find_nhil_files(directory: str) -> list[str]:
    """Find all non-hil result JSON files in a directory tree."""
    patterns = [
        os.path.join(directory, "**", "results_*", "*.json"),
        os.path.join(directory, "**", "results-*", "*.json"),
    ]
    files = []
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            if "/staged_" in f or "/staged/" in f:
                continue
            basename = os.path.basename(f).lower()
            if any(x in basename for x in ["no_hil", "nhil", "without_hil"]):
                files.append(f)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NER entity-label correctness"
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file or directory containing NER results",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save detailed report as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-entity decisions including unknowns",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary table, skip per-file details",
    )

    args = parser.parse_args()

    # Determine input files
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = find_nhil_files(args.input)
        if not files:
            print(f"No non-hil JSON files found in {args.input}")
            sys.exit(1)
        print(f"Found {len(files)} non-hil result files")
    else:
        print(f"Not found: {args.input}")
        sys.exit(1)

    # Evaluate
    all_reports = []
    for filepath in files:
        report = evaluate_file(filepath, verbose=args.verbose)
        all_reports.append(report)
        if not args.summary_only:
            print_report(report, verbose=args.verbose)

    # Summary table for multiple files
    if len(files) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY ACROSS ALL FILES")
        print(f"{'='*70}")
        print(
            f"\n  {'File':<50s} {'Total':>5s} {'Eval':>5s} "
            f"{'Corr':>5s} {'Incor':>5s} {'Unkn':>5s} {'Acc%':>6s}"
        )
        print(f"  {'-'*82}")

        totals = {"total": 0, "eval": 0, "correct": 0, "incorrect": 0, "unknown": 0}
        for r in all_reports:
            fname = os.path.basename(r["file"])[:50]
            ev = r["evaluated"]
            scored = ev["correct"] + ev["incorrect"]
            acc = f"{ev['correct']/scored*100:.1f}" if scored > 0 else "N/A"
            print(
                f"  {fname:<50s} {r['total_entities']:>5d} {ev['count']:>5d} "
                f"{ev['correct']:>5d} {ev['incorrect']:>5d} {ev['unknown']:>5d} {acc:>6s}"
            )
            totals["total"] += r["total_entities"]
            totals["eval"] += ev["count"]
            totals["correct"] += ev["correct"]
            totals["incorrect"] += ev["incorrect"]
            totals["unknown"] += ev["unknown"]

        print(f"  {'-'*82}")
        scored = totals["correct"] + totals["incorrect"]
        acc = f"{totals['correct']/scored*100:.1f}" if scored > 0 else "N/A"
        print(
            f"  {'TOTAL':<50s} {totals['total']:>5d} {totals['eval']:>5d} "
            f"{totals['correct']:>5d} {totals['incorrect']:>5d} {totals['unknown']:>5d} {acc:>6s}"
        )
        print()

    # Save JSON report
    if args.output:
        # Convert Counters to dicts for JSON serialization
        for r in all_reports:
            r["excluded"]["reasons"] = dict(r["excluded"]["reasons"])
            r["label_distribution"] = dict(r["label_distribution"])

        output_data = all_reports if len(all_reports) > 1 else all_reports[0]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
