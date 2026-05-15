"""Build the final merged entity+mapping+provenance JSON deliverable."""
import json
from datetime import datetime, timezone
from pathlib import Path

agent1 = json.loads(Path("agent1_output_v2.json").read_text())
agent2 = json.loads(Path("agent2_output_v2.json").read_text())
agent3 = json.loads(Path("agent3_output_v2.json").read_text())

entities = {e["entity_id"]: e for e in agent1["entities"]}
mappings = {m["entity_id"]: m for m in agent2["mappings"]}

merged_entities = {}
for eid, ent in entities.items():
    m = mappings.get(eid, {})
    merged_entities[eid] = {
        "entity_id": eid,
        "entity_text": ent["entity_text"],
        "entity_type": ent["entity_type"],
        "surface_forms": ent["surface_forms"],
        "source_sentences": ent.get("source_sentences", []),
        "occurrence_count": ent.get("occurrence_count", 0),
        "indirect_references": ent.get("indirect_references", []),
        "extraction_round": ent.get("extraction_round"),
        # Mapping
        "ontology": m.get("ontology"),
        "ontology_id": m.get("ontology_id"),
        "ontology_label": m.get("ontology_label"),
        "confidence": m.get("confidence", "none"),
        "id_validated": m.get("id_validated", False),
        "mapping_note": m.get("mapping_note", ""),
    }

deliverable = {
    "source_paper": {
        "title": "Multiscale Spatial Transcriptomic Atlas of Human Basal Ganglia Cell-Type and Cellular Community Organization",
        "authors_short": "Berackey BT, Tan Z, Wu G, Das SC, Li R, Esser B, Ye Q, Nafisi M, Park SS, Sequeira Mendieta PA, Berry J, Mamdani F, Zhu Q, Holmes TC, Li D, Wang T, Behrens MM, Ren B, Ecker JR, Bintu B, Xu X",
        "venue": "bioRxiv preprint",
        "posted_date": "2025-12-05",
        "doi": "10.64898/2025.12.02.691876",
        "source_file": "/Users/pujatrivedi/Desktop/MIT/structsense/evaluation/ner/multiscale_spatial_transcriptomic/2025.12.02.691876v1.full.pdf",
        "keywords": ["MERFISH+", "Stereo-seq", "single-cell atlas", "cellular communities",
                     "transcript topography", "striosome–matrix organization", "cross-species comparison"]
    },
    "pipeline_run": {
        "completion_time_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "reviewer_status": agent3.get("status"),
        "reviewer_iterations": agent3.get("iteration"),
        "extraction_certified_exhaustive": agent3.get("extraction_certified_exhaustive"),
        "models_used": {
            "extractor": "claude-opus-4-7",
            "mapper": "claude-opus-4-7",
            "reviewer": "claude-opus-4-7 (1M context) - spawned fresh agent for iteration 1, sent message for iteration 2",
        },
        "isolation_limitation": agent3.get("reviewer_isolation"),
        "total_entities": len(merged_entities),
        "total_mapped_with_id": sum(1 for e in merged_entities.values() if e.get("ontology_id")),
    },
    "extraction_audit": {
        "round_log": agent1["round_log"],
        "masking_applied": agent1["masking_applied"],
        "termination_reason": agent1["termination_reason"],
        "coreference_pass_completed": agent1["coreference_pass_completed"],
        "coreference_summary": {
            "entities_with_indirect_refs": sum(1 for e in merged_entities.values() if e["indirect_references"]),
            "total_indirect_refs": sum(len(e["indirect_references"]) for e in merged_entities.values()),
        },
        "reviewer_checklist": agent3.get("checklist"),
        "reviewer_independent_candidates_file": "reviewer_independent_candidates.txt",
        "reviewer_independent_diff_missing_count": agent3.get("independent_diff_missing_count"),
        "id_validation_method": agent2.get("id_validation_method"),
        "ontology_mapping_caveat": agent3.get("notes", "")[:1000],
    },
    "entities": merged_entities,
}

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
fname = f"multiscale_spatial_transcriptomic_{stamp}.json"
Path(fname).write_text(json.dumps(deliverable, indent=2))
print(f"Final deliverable: {fname}")
print(f"Entities: {len(merged_entities)}, mapped-with-id: {sum(1 for e in merged_entities.values() if e.get('ontology_id'))}")
