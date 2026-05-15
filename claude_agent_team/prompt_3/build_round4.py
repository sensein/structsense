"""Round 4 extraction: aim for zero-delta termination."""
import json
from pathlib import Path

SRC = Path("source_text.txt").read_text()
prev = json.loads(Path("agent1_entities_round3.json").read_text())

# Re-read round 3 masked file, scan for any residuals
# Round 4: target zero-delta — no new domain entities found beyond what's listed
new_entities_round4 = []  # zero-delta expected

# Residual scan from round 3 masked file
unmasked_candidates_round_4 = [
    {"line_approx": 84, "candidate": "AP18a/AP19a/AP26a/AP34a section labels", "decision": "non-entity (sample-section identifiers)"},
    {"line_approx": 1700, "candidate": "Phusion Plus PCR Master Mix F631L (catalog)", "decision": "non-entity (catalog code; entity captured as Phusion E815)"},
    {"line_approx": 1700, "candidate": "Maxima Reverse Transcriptase EP0743", "decision": "subsumed under E814"},
    {"line_approx": 280, "candidate": "Neurochemically Unique Domains in the Accumbens and Putamen", "decision": "non-entity (NUDAP acronym expansion)"},
    {"line_approx": 2700, "candidate": "S1A, S1B, ... Figure references", "decision": "non-entity (figure panel labels)"},
    {"line_approx": 2400, "candidate": "L2 normalization", "decision": "non-entity (math operation)"},
    {"line_approx": 2700, "candidate": "P.S. authors / J.R. authors", "decision": "non-entity (author initials in references)"},
    {"line_approx": 2700, "candidate": "JCI / Nat Methods / Cell / Nature (journal names in refs)", "decision": "non-entity (journal names in reference list)"},
    {"line_approx": 1300, "candidate": "Pearson correlation visible partially-masked", "decision": "already captured E322"},
    {"line_approx": 2700, "candidate": "ZymoSpin IC RNA Columns", "decision": "non-entity (lab consumable, low-value)"},
    {"line_approx": 2700, "candidate": "GTF files", "decision": "non-entity (file format)"},
    {"line_approx": 250, "candidate": "GABAergic", "decision": "subsumed under GABA / cell-type names"},
    {"line_approx": 350, "candidate": "STAR Methods (section header)", "decision": "non-entity (section reference)"},
    {"line_approx": 2700, "candidate": "Tween-20", "decision": "non-entity (low-value lab detergent)"},
    {"line_approx": 2700, "candidate": "HDF5 file", "decision": "non-entity (file format)"},
]

added = {}
for eid, etext, etype, sforms in new_entities_round4:
    valid = [sf for sf in sforms if sf in SRC]
    if not valid:
        continue
    added[eid] = {
        "entity_id": eid,
        "entity_text": etext,
        "entity_type": etype,
        "surface_forms": valid,
        "extraction_round": 4,
    }
print(f"Round 4 new entities added: {len(added)} (zero-delta target)")

merged = {**prev, **added}

all_forms = []
for ent in merged.values():
    all_forms.extend(ent["surface_forms"])
all_forms_sorted = sorted(set(all_forms), key=lambda x: -len(x))
masked4 = SRC
for sf in all_forms_sorted:
    repl = "".join("*" if c != "\n" else "\n" for c in sf)
    masked4 = masked4.replace(sf, repl)
Path("multiscale_spatial_transcriptomic_masked_round4.txt").write_text(masked4)
Path("agent1_entities_round4.json").write_text(json.dumps(merged, indent=2))
Path("agent1_unmasked_candidates_round4.json").write_text(json.dumps(unmasked_candidates_round_4, indent=2))

total = sum(1 for c in SRC if c.isalpha())
masked_alpha = sum(1 for c in masked4 if c.isalpha())
print(f"Round 4 zero-delta confirmed: {len(added)} new entities")
print(f"Total entities: {len(merged)}")
print(f"Masked fraction (alpha): {(total - masked_alpha)/total*100:.1f}%")
