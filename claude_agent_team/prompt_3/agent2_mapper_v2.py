"""Agent 2 v2 — update mappings: add mappings for the 25 new entities and resolve
the 8 partial mappings the Reviewer flagged."""
import json
from pathlib import Path

agent1 = json.loads(Path("agent1_output_v2.json").read_text())
agent2_v1 = json.loads(Path("agent2_output.json").read_text())
entities = {e["entity_id"]: e for e in agent1["entities"]}
prev_mappings = {m["entity_id"]: m for m in agent2_v1["mappings"]}

# Resolve partial mappings flagged by Reviewer
resolve = {
    "E133": (None, None, None, "none"),  # CTXND1 — drop ontology (no validated HGNC ID)
    "E303": ("RRID", "RRID:SCR_025001", "Stereo-seq Analysis Workflow (SAW) / Stereo-seq platform", "high"),
    "E308": ("OBI", "OBI:0001271", "single-molecule FISH (smFISH)", "medium"),
    "E327": (None, None, "LOWESS regression", "low"),
    "E329": (None, None, "Delaunay triangulation", "low"),
    "E408": (None, None, "pyvista", "low"),
    "E520": (None, None, "sublenticular extended amygdala (SLEA)", "low"),
    "E507": ("NCBIGenome", "GCA_000001405.15", "GRCh38 human reference genome (Ensembl)", "high"),
}
for eid, (ont, oid, lbl, conf) in resolve.items():
    if eid in prev_mappings:
        prev_mappings[eid].update({
            "ontology": ont,
            "ontology_id": oid,
            "ontology_label": lbl,
            "confidence": conf,
            "id_validated": False,
            "mapping_note": "ID recalled / refined per Reviewer iteration 1 feedback"
        })

# New entity mappings (round 5 additions)
new_maps = {
    "E346": ("OBI", "OBI:0002698", "spatial transcriptomics", "high"),
    "E347": (None, None, "in situ sequencing", "low"),
    "E348": ("OBI", "OBI:0600003", "immunohistochemistry", "high"),
    "E349": ("OBI", "OBI:0002628", "immunofluorescence staining", "medium"),
    "E522": ("RRID", "RRID:SCR_002978", "Allen Brain Atlas", "high"),
    "E229": ("UBERON", "UBERON:0002023", "claustrum", "high"),
    "E713": ("CL", "CL:0000129", "microglial cell", "high"),
    "E714": ("CL", "CL:0000576", "monocyte", "high"),
    "E715": ("CL", "CL:0000236", "B cell", "high"),
    "E716": ("CL", "CL:0000084", "T cell", "high"),
    "E717": ("CL", "CL:4023050", "vascular leptomeningeal cell", "high"),
    "E718": ("CL", "CL:0000192", "smooth muscle cell", "high"),
    "E719": (None, None, None, "none"),  # paper-specific cluster name
    "E197": ("HGNC", "HGNC:8632", "PCP4", "high"),
    "E198": ("HGNC", "HGNC:14005", "PEG10", "high"),
    "E199": ("HGNC", None, "SRRIT1 (uncertain symbol)", "low"),
    "E200": ("MGI", "MGI:1913150", "Trem2 (mouse)", "high"),
    "E606": ("MONDO", "MONDO:0005329", "Tourette syndrome", "high"),
    "E607": ("MONDO", "MONDO:0005090", "schizophrenia", "high"),
    "E608": ("MONDO", "MONDO:0005260", "autism spectrum disorder", "high"),
    "E609": ("MONDO", "MONDO:0011110", "developmental language disorder", "medium"),
    "E720": ("NCBITaxon", "NCBITaxon:9499", "Callithrix jacchus (marmoset)", "high"),
    "E828": ("CHEBI", "CHEBI:42191", "EDTA", "high"),
    "E829": ("CHEBI", "CHEBI:32145", "sodium hydroxide", "high"),
    "E830": (None, None, "Tris-EDTA (TE) buffer", "low"),
}
for eid, (ont, oid, lbl, conf) in new_maps.items():
    prev_mappings[eid] = {
        "entity_id": eid,
        "entity_text": entities[eid]["entity_text"] if eid in entities else "",
        "ontology": ont,
        "ontology_id": oid,
        "ontology_label": lbl,
        "confidence": conf,
        "id_validated": False,
        "mapping_note": "ID recalled from training data; round 5 addition (Reviewer triggered)"
    }

# Recompute unmapped set
unmapped = [eid for eid, m in prev_mappings.items() if not m.get("ontology_id")]

agent2_out = {
    "agent": "OntologyMapper",
    "model_used": "claude-opus-4-7 (orchestrated, single-model context)",
    "id_validation_method": "none (recalled, not validated)",
    "mappings": list(prev_mappings.values()),
    "unmapped": unmapped,
}
Path("agent2_output_v2.json").write_text(json.dumps(agent2_out, indent=2))
mapped = len(prev_mappings) - len(unmapped)
print(f"Agent 2 v2: total {len(prev_mappings)}, mapped {mapped} ({mapped/len(prev_mappings)*100:.1f}%), unmapped {len(unmapped)}")
