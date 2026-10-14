"""Round 3 extraction: scan round-2 masked file for residuals, add domain-relevant entities, terminate if zero-delta."""
import json
import re
from pathlib import Path

SRC = Path("source_text.txt").read_text()
prev = json.loads(Path("agent1_entities_round2.json").read_text())

# Round 3 additions (small set discovered in round-2 masked file)
new_entities_round3 = [
    ("E243b", "GABA", "BiologicalConcept", ["GABA"]),
    ("E515", "AIBS", "Organization", ["AIBS"]),
    ("E516", "PBS", "Reagent", ["PBS"]),
    ("E517", "20X SSC", "Reagent", ["20X SSC", "0.1× SSC", "2xSSC", "2x SSC"]),
    ("E518", "Center for Neural Circuit Mapping", "Organization", ["Center for Neural Circuit Mapping (CNCM)", "CNCM"]),
    ("E519", "amygdala", "BrainRegion", ["AMY"]),
    ("E520", "sublenticular extended amygdala", "BrainRegion", ["SLEA"]),
    ("E521", "MERFISH+ assay", "Method", ["MER1", "MER2", "MER3", "MER4"]),
]

# Round-3 residuals enumerated from reading round-2 masked file
unmasked_candidates_round_3 = [
    {"line_approx": 60, "candidate": "GABA (neurotransmitter)", "decision": "new entity (BiologicalConcept) — captured as E243b"},
    {"line_approx": 350, "candidate": "STAR (STAR Methods)", "decision": "non-entity (section reference)"},
    {"line_approx": 1268, "candidate": "20X SSC / 0.1× SSC / 2xSSC", "decision": "new entity (Reagent) — captured as E517"},
    {"line_approx": 1368, "candidate": "PBS", "decision": "new entity (Reagent) — captured as E516"},
    {"line_approx": 21, "candidate": "Center for Neural Circuit Mapping (CNCM)", "decision": "new entity (Organization) — captured as E518"},
    {"line_approx": 2680, "candidate": "AMY (amygdala)", "decision": "new entity (BrainRegion) — captured as E519"},
    {"line_approx": 2680, "candidate": "SLEA", "decision": "new entity (BrainRegion) — captured as E520"},
    {"line_approx": 1125, "candidate": "MER1 / MER2 / MER3 / MER4", "decision": "new entity (Method) — captured as E521 (MERFISH+ assay labels)"},
    {"line_approx": 105, "candidate": "AIBS", "decision": "new entity (Organization) — captured as E515"},
    {"line_approx": 2080, "candidate": "M33631 / D4006 / Q32854 / Q10212 / SAF (catalog numbers)", "decision": "non-entity (vendor catalog numbers)"},
    {"line_approx": 1620, "candidate": "RESOURCE TABLE header", "decision": "non-entity (heading)"},
    {"line_approx": 84, "candidate": "AP18 / AP19 / AP26 / AP34 / ALL section labels", "decision": "non-entity (sample-section identifiers)"},
    {"line_approx": 2680, "candidate": "JNEUROSCI", "decision": "non-entity (journal abbreviation in references)"},
    {"line_approx": 2680, "candidate": "NIH", "decision": "non-entity (funding agency in acknowledgments)"},
    {"line_approx": 280, "candidate": "Neurochemically Unique Domains in the Accumbens and Putamen", "decision": "non-entity (NUDAP acronym expansion — STRv D1 NUDAP captured)"},
    {"line_approx": 2680, "candidate": "PCR", "decision": "non-entity (generic technique — polymerase chain reaction)"},
    {"line_approx": 2680, "candidate": "NDB", "decision": "subsumed under DNA nanoball E311"},
    {"line_approx": 350, "candidate": "Wilcoxon rank-sum test (visible partly masked)", "decision": "already captured E323"},
    {"line_approx": 2073, "candidate": "PE100 / PE100_50+100 (sequencing config)", "decision": "non-entity (config parameter)"},
    {"line_approx": 2076, "candidate": "FASTQ / fastq", "decision": "non-entity (file format)"},
    {"line_approx": 2080, "candidate": "Q10 / N bases (quality filter)", "decision": "non-entity (QC parameter)"},
]

added = {}
for eid, etext, etype, sforms in new_entities_round3:
    valid = [sf for sf in sforms if sf in SRC]
    if not valid:
        print(f"DROP {eid}: no valid forms")
        continue
    added[eid] = {
        "entity_id": eid,
        "entity_text": etext,
        "entity_type": etype,
        "surface_forms": valid,
        "extraction_round": 3,
    }
print(f"Round 3 new entities added: {len(added)}")

merged = {**prev, **added}

# Build masked round 3
all_forms = []
for ent in merged.values():
    all_forms.extend(ent["surface_forms"])
all_forms_sorted = sorted(set(all_forms), key=lambda x: -len(x))
masked3 = SRC
for sf in all_forms_sorted:
    repl = "".join("*" if c != "\n" else "\n" for c in sf)
    masked3 = masked3.replace(sf, repl)
Path("multiscale_spatial_transcriptomic_masked_round3.txt").write_text(masked3)

Path("agent1_entities_round3.json").write_text(json.dumps(merged, indent=2))
Path("agent1_unmasked_candidates_round3.json").write_text(json.dumps(unmasked_candidates_round_3, indent=2))

total = sum(1 for c in SRC if c.isalpha())
masked_alpha = sum(1 for c in masked3 if c.isalpha())
print(f"Total entities now: {len(merged)}")
print(f"Masked fraction (alpha): {(total - masked_alpha)/total*100:.1f}%")
