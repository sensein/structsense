"""Round 2 extraction: read round-1 masked file fresh, enumerate residuals, add new entities."""
import json
import re
from pathlib import Path

SRC = Path("source_text.txt").read_text()
MASKED1 = Path("multiscale_spatial_transcriptomic_masked_round1.txt").read_text()
prev = json.loads(Path("agent1_entities_round1.json").read_text())

# Round 2: NEW entities discovered by reading masked round-1 file fresh.
# These are residuals that were not in the round-1 entity list.
new_entities_round2 = [
    # Developmental brain regions (eminences) — multiple mentions in masked text
    ("E230", "caudal ganglionic eminence", "BrainRegion", ["caudal ganglionic\neminence", "CGE", "caudal\nganglionic eminence"]),
    ("E231", "lateral ganglionic eminence", "BrainRegion", ["lateral ganglionic\neminence", "LGE", "lateral\nganglionic eminence"]),
    ("E232", "medial ganglionic eminence", "BrainRegion", ["medial ganglionic\neminence", "MGE", "medial\nganglionic eminence"]),
    ("E233", "ganglionic eminence", "BrainRegion", ["ganglionic eminence"]),
    # Subcellular structures / molecules
    ("E234", "neurite transcripts", "BiologicalConcept", ["neurite transcripts", "neuritic"]),
    ("E235", "axonal", "BiologicalConcept", ["axonal"]),
    ("E236", "dendritic", "BiologicalConcept", ["dendritic"]),
    ("E237", "soma", "BiologicalConcept", ["soma", "somatic"]),
    ("E238", "cytoplasm", "BiologicalConcept", ["cytoplasm", "cytoplasmic"]),
    ("E239", "nucleus (subcellular)", "BiologicalConcept", ["nuclear"]),
    ("E240", "semaphorin family", "ProteinFamily", ["semaphorin family"]),
    ("E241", "GABA transporter", "ProteinFamily", ["GABA transporter"]),
    ("E242", "direct pathway", "BiologicalConcept", ["direct pathway"]),
    ("E243", "indirect pathway", "BiologicalConcept", ["indirect pathway"]),
    # Methods
    ("E338", "point spread function", "Method", ["point spread function (PSF)", "point spread function"]),
    ("E339", "DL-VM gradient", "BiologicalConcept", ["dorsolateral- ventromedial gradient", "dorsolateral-ventromedial", "DL-VM"]),
    ("E340", "field of view", "Method", ["field of view", "FOV"]),
    ("E341", "directed acyclic graph", "DataStructure", ["directed acyclic graph (DAG)"]),
    ("E342", "spatial modules", "BiologicalConcept", ["spatial modules"]),
    ("E343", "cellular community", "BiologicalConcept", ["cellular community"]),
    ("E344", "log1p normalization", "Method", ["log1p"]),
    ("E345", "DAPI staining", "Method", ["nuclei-staining", "nuclei staining"]),
    # Datasets / kits identifiers
    ("E513", "Stereo-seq T FF V1.2", "Reagent", ["Stereo-seq T FF V1.2"]),
    # Software modules
    ("E419", "KDTree", "Software", ["KDTree"]),
    # Identifiers / metadata
    ("E1207", "coordinate ID (CID)", "DataField", ["coordinate \nID (CID)", "CID"]),
    ("E1208", "Molecular ID (MID)", "DataField", ["Molecular ID (MID)", "MID"]),
    # Species expansion
    ("E708", "primate", "Species", ["primate", "primates"]),
    # Figure subclusters
    ("E708b", "Astro-0", "CellType", ["Astro-0"]),
    ("E709b", "Astro-1", "CellType", ["Astro-1"]),
    ("E710b", "Astro-2", "CellType", ["Astro-2"]),
    ("E711b", "Astro-3", "CellType", ["Astro-3"]),
    # GO ontology
    ("E514", "Gene Ontology", "Database", ["Gene Ontology", "GO"]),
    # Brain Initiative Cell Atlas Network already at E504
    # snRNA-seq variants
    ("E304b", "snRNA-seq variant single-nucleus RNA-seq", "Technology", ["single-nucleus RNA sequencing", "single-nucleus\nRNA sequencing"]),
    # Round 2 misc
    ("E605b", "neurodegenerative disorder", "Disease", ["neurodegenerative disorders"]),
    ("E236b", "ssDNA", "BiologicalConcept", ["ssDNA"]),
    ("E237b", "mRNA", "BiologicalConcept", ["mRNA", "mRNAs"]),
    ("E238b", "cDNA", "BiologicalConcept", ["cDNA"]),
    # Section labels
    ("E901b", "anterior-posterior axis", "BiologicalConcept", ["anterior–posterior axis", "anterior-posterior axis", "anterior–posterior (A–P) axis"]),
    # Additional cell types
    ("E712b", "ependymal cells", "CellType", ["ependymal"]),
]

# Validate
dropped = []
added = {}
for eid, etext, etype, sforms in new_entities_round2:
    valid = [sf for sf in sforms if sf in SRC]
    invalid = [sf for sf in sforms if sf not in SRC]
    for sf in invalid:
        dropped.append((eid, sf))
    if not valid:
        print(f"DROP entity {eid} ({etext}) - no valid forms")
        continue
    added[eid] = {
        "entity_id": eid,
        "entity_text": etext,
        "entity_type": etype,
        "surface_forms": valid,
        "extraction_round": 2,
    }
print(f"Round 2 new entities added: {len(added)}")
print(f"Round 2 dropped surface forms: {len(dropped)}")
for d in dropped:
    print(f"  DROP: {d}")

# Build unmasked_candidates_round_2 (what I saw on the round-1 masked file)
unmasked_candidates_round_2 = [
    {"line_approx": 252, "candidate": "caudal and lateral ganglionic eminence", "decision": "new entity (BrainRegion) — captured as E230/E231"},
    {"line_approx": 191, "candidate": "circuit-level organization", "decision": "non-entity prose"},
    {"line_approx": 198, "candidate": "Caucasian male donors", "decision": "non-entity demographic descriptor"},
    {"line_approx": 213, "candidate": "20mm x 30mm", "decision": "non-entity (measurement)"},
    {"line_approx": 343, "candidate": "GABA transporter", "decision": "new entity (ProteinFamily) — captured as E241"},
    {"line_approx": 400, "candidate": "semaphorin family", "decision": "new entity (ProteinFamily) — captured as E240"},
    {"line_approx": 425, "candidate": "spatial modules", "decision": "new entity (BiologicalConcept) — captured as E342"},
    {"line_approx": 1487, "candidate": "Cellpose 'nuclei' model", "decision": "already captured as E409 Cellpose (variant)"},
    {"line_approx": 2073, "candidate": "Stereo-seq T FF V1.2", "decision": "new entity (Reagent kit version) — captured as E513"},
    {"line_approx": 2076, "candidate": "FASTQ files", "decision": "non-entity (file format)"},
    {"line_approx": 2077, "candidate": "coordinate ID (CID)", "decision": "new entity (DataField) — captured as E1207"},
    {"line_approx": 2078, "candidate": "Molecular ID (MID)", "decision": "new entity (DataField) — captured as E1208"},
    {"line_approx": 2097, "candidate": "watershed algorithm", "decision": "already in entity list — was dropped due to newline; manually adding as 'watershed' substring would be subsumed"},
    {"line_approx": 2111, "candidate": "point spread function (PSF)", "decision": "new entity (Method) — captured as E338"},
    {"line_approx": 2384, "candidate": "directed acyclic graph (DAG)", "decision": "new entity (DataStructure) — captured as E341"},
    {"line_approx": 2422, "candidate": "field of view (FOV)", "decision": "new entity (Method) — captured as E340"},
    {"line_approx": 380, "candidate": "neurite transcripts", "decision": "new entity (BiologicalConcept) — captured as E234"},
    {"line_approx": 252, "candidate": "GABAergic neurons", "decision": "subsumed under CN LGE/MGE/CGE GABA cell types"},
    {"line_approx": 152, "candidate": "neurotypical donors", "decision": "non-entity demographic"},
    {"line_approx": 365, "candidate": "three-dimensional point cloud", "decision": "non-entity (method detail)"},
    {"line_approx": 478, "candidate": "primate-specific", "decision": "non-entity (adjective); 'primate' captured as E708"},
    {"line_approx": 1632, "candidate": "Gene Ontology", "decision": "new entity (Database) — captured as E514"},
    {"line_approx": 2206, "candidate": "globus palladium", "decision": "typo for globus pallidus — already captured as E208"},
    {"line_approx": 2384, "candidate": "go-basic.obo", "decision": "non-entity (file reference)"},
    {"line_approx": 2099, "candidate": "ssDNA-staining", "decision": "method detail — 'ssDNA' captured as E236b"},
]

# Merge with previous
merged = {**prev, **added}
for eid in prev:
    merged[eid]["extraction_round"] = prev[eid].get("extraction_round", 1)

# Rebuild masked round-2 from SRC with the full merged set
all_forms = []
for ent in merged.values():
    all_forms.extend(ent["surface_forms"])
all_forms_sorted = sorted(set(all_forms), key=lambda x: -len(x))

masked2 = SRC
for sf in all_forms_sorted:
    repl = "".join("*" if c != "\n" else "\n" for c in sf)
    masked2 = masked2.replace(sf, repl)

Path("multiscale_spatial_transcriptomic_masked_round2.txt").write_text(masked2)
Path("agent1_entities_round2.json").write_text(json.dumps(merged, indent=2))
Path("agent1_unmasked_candidates_round2.json").write_text(json.dumps(unmasked_candidates_round_2, indent=2))

total = sum(1 for c in SRC if c.isalpha())
masked_alpha = sum(1 for c in masked2 if c.isalpha())
print(f"\nRound 2 done. Total entities: {len(merged)}")
print(f"Masked fraction (alpha): {(total - masked_alpha)/total*100:.1f}%")
print(f"Entities added round 2: {len(added)}")
