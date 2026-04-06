"""Unit tests for parallel downstream chunking correctness.

Tests cover:
  - unify_ontology_across_entities   (postprocessing.py)
  - split_structured_payload         (downstream_agent_helper.py)
  - merge_structured_chunk_results   (downstream_agent_helper.py)
  - data-loss guard: merged count vs pre-split count

All tests are fully offline — no API key, no LLM, no network.
"""
import sys
import os
import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.postprocessing import unify_ontology_across_entities
from utils.downstream_agent_helper import (
    split_structured_payload,
    merge_structured_chunk_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entities(n: int, label: str = "brain region") -> list:
    return [
        {
            "entity": f"entity_{i}",
            "label": label,
            "sentence": f"Sentence {i}.",
            "start": i * 10,
            "end": i * 10 + 5,
        }
        for i in range(n)
    ]


def _make_payload(n_entities: int, n_key_terms: int = 0) -> dict:
    return {
        "entities": _make_entities(n_entities),
        "key_terms": [{"term": f"term_{i}"} for i in range(n_key_terms)],
        "task_type": "ner",
    }



# ---------------------------------------------------------------------------
# unify_ontology_across_entities
# ---------------------------------------------------------------------------

class TestUnifyOntologyAcrossEntities:

    def test_single_entity_unchanged(self):
        entities = [{"entity": "hippocampus", "label": "brain region",
                     "ontology_id": "UBERON:0001954", "concept_mapping_provenance": "tool"}]
        result = unify_ontology_across_entities(entities)
        assert result[0]["ontology_id"] == "UBERON:0001954"

    def test_tool_beats_llm_knowledge(self):
        """Two occurrences of same entity: one has tool mapping, one has llm_knowledge.
        After unification both should carry the tool mapping."""
        entities = [
            {"entity": "hippocampus", "label": "brain region",
             "ontology_id": "UBERON:0001954", "concept_mapping_provenance": "tool"},
            {"entity": "hippocampus", "label": "brain region",
             "ontology_id": "UBERON:0002421", "concept_mapping_provenance": "llm_knowledge"},
        ]
        result = unify_ontology_across_entities(entities)
        # Both occurrences must now carry the tool-backed ID
        assert all(e["ontology_id"] == "UBERON:0001954" for e in result)
        assert all(e["concept_mapping_provenance"] == "tool" for e in result)

    def test_real_iri_beats_na(self):
        """One occurrence has a real IRI, another has N/A."""
        entities = [
            {"entity": "CA1", "label": "brain region",
             "ontology_id": "UBERON:0003881", "concept_mapping_provenance": "tool"},
            {"entity": "CA1", "label": "brain region",
             "ontology_id": "N/A", "concept_mapping_provenance": "llm_knowledge"},
        ]
        result = unify_ontology_across_entities(entities)
        assert all(e["ontology_id"] == "UBERON:0003881" for e in result)

    def test_preserves_all_instances(self):
        """Unification must not deduplicate entity instances — all rows are kept."""
        entities = [
            {"entity": "IL-6", "label": "cytokine",
             "sentence": "IL-6 was elevated.", "ontology_id": "PR:000001164",
             "concept_mapping_provenance": "tool"},
            {"entity": "IL-6", "label": "cytokine",
             "sentence": "IL-6 mediates signalling.", "ontology_id": "N/A",
             "concept_mapping_provenance": "llm_knowledge"},
        ]
        result = unify_ontology_across_entities(entities)
        # Both rows preserved
        assert len(result) == 2
        # Both carry the real IRI
        assert all(e["ontology_id"] == "PR:000001164" for e in result)
        # Sentence context is not modified
        sentences = {e["sentence"] for e in result}
        assert "IL-6 was elevated." in sentences
        assert "IL-6 mediates signalling." in sentences

    def test_different_labels_not_unified(self):
        """Entities with the same text but different labels are treated independently."""
        entities = [
            {"entity": "CD4", "label": "cell surface marker",
             "ontology_id": "PR:000001004", "concept_mapping_provenance": "tool"},
            {"entity": "CD4", "label": "t-cell subtype",
             "ontology_id": "CL:0000624", "concept_mapping_provenance": "tool"},
        ]
        result = unify_ontology_across_entities(entities)
        ids = {e["ontology_id"] for e in result}
        # Both IDs must be retained — they are for different labels
        assert "PR:000001004" in ids
        assert "CL:0000624" in ids

    def test_empty_list(self):
        assert unify_ontology_across_entities([]) == []

    def test_non_dict_items_skipped(self):
        entities = [None, "string_item",
                    {"entity": "p53", "label": "gene",
                     "ontology_id": "NCBIGene:7157", "concept_mapping_provenance": "tool"}]
        result = unify_ontology_across_entities(entities)
        # Non-dict items pass through unchanged, dict item is processed
        assert result[2]["ontology_id"] == "NCBIGene:7157"

    def test_missing_ontology_fields_filled_from_best(self):
        """Entity with no ontology_id receives it from a sibling with the same text."""
        entities = [
            {"entity": "neocortex", "label": "brain region",
             "ontology_id": "UBERON:0001950", "concept_mapping_provenance": "tool"},
            {"entity": "neocortex", "label": "brain region"},  # no ontology fields at all
        ]
        result = unify_ontology_across_entities(entities)
        # Second entity should now have the ontology_id from the first
        assert result[1].get("ontology_id") == "UBERON:0001950"
        assert result[1].get("concept_mapping_provenance") == "tool"


# ---------------------------------------------------------------------------
# split_structured_payload + merge_structured_chunk_results
# ---------------------------------------------------------------------------

class TestSplitAndMerge:

    def test_no_data_loss_small(self):
        """After split → merge, entity count must equal original count."""
        n = 50
        payload = _make_payload(n_entities=n)
        chunks = split_structured_payload(payload, max_entities_per_chunk=10)
        # Each chunk is a dict containing a slice of entities
        total_in_chunks = sum(len(c.get("entities") or []) for c in chunks)
        assert total_in_chunks == n, f"split lost entities: {total_in_chunks} != {n}"

    def test_no_data_loss_large(self):
        """500 entities split across many chunks must all survive the round-trip."""
        n = 500
        payload = _make_payload(n_entities=n, n_key_terms=100)
        chunks = split_structured_payload(payload, max_entities_per_chunk=50)
        total_in_chunks = sum(len(c.get("entities") or []) for c in chunks)
        assert total_in_chunks == n

        # Simulate merge (chunks already contain the split entity slices;
        # merge_structured_chunk_results expects raw result dicts from agent runs,
        # so we pass the chunk dicts directly as mock results)
        merged = merge_structured_chunk_results(chunks)
        assert len(merged.get("entities") or []) == n, (
            f"merge lost entities: {len(merged.get('entities', []))} != {n}"
        )

    def test_single_chunk_no_split(self):
        """A payload with fewer entities than the cap stays as one chunk."""
        payload = _make_payload(n_entities=5)
        chunks = split_structured_payload(payload, max_entities_per_chunk=50)
        assert len(chunks) == 1

    def test_chunk_count_matches_entity_count(self):
        """With max_entities_per_chunk=10 and 100 entities we expect 10 chunks."""
        payload = _make_payload(n_entities=100)
        chunks = split_structured_payload(payload, max_entities_per_chunk=10)
        assert len(chunks) == 10

    def test_chunk_metadata_present(self):
        """Each chunk must carry _chunk_index and _chunk_total."""
        payload = _make_payload(n_entities=30)
        chunks = split_structured_payload(payload, max_entities_per_chunk=10)
        for i, chunk in enumerate(chunks):
            assert chunk["_chunk_index"] == i
            assert chunk["_chunk_total"] == len(chunks)

    def test_key_terms_deduped_after_merge(self):
        """merge_structured_chunk_results deduplicates key_terms by term value."""
        chunk1 = {"entities": _make_entities(5), "key_terms": [{"term": "alpha"}, {"term": "beta"}]}
        chunk2 = {"entities": _make_entities(5, label="cell"), "key_terms": [{"term": "beta"}, {"term": "gamma"}]}
        merged = merge_structured_chunk_results([chunk1, chunk2])
        kts = [t["term"] for t in merged.get("key_terms", [])]
        # "beta" appeared in both → only one copy after dedup
        assert kts.count("beta") == 1
        assert set(kts) == {"alpha", "beta", "gamma"}

    def test_resources_preserved(self):
        """Resources are concatenated without loss."""
        chunk1 = {"resources": [{"name": "tool_A", "type": "software"}]}
        chunk2 = {"resources": [{"name": "tool_B", "type": "dataset"}]}
        merged = merge_structured_chunk_results([chunk1, chunk2])
        names = [r["name"] for r in merged.get("resources", [])]
        assert "tool_A" in names
        assert "tool_B" in names

    def test_key_terms_do_not_inflate_chunk_count(self):
        """key_terms must never drive n_chunks above what entities require.

        Regression for a production bug: with 2130 entities (max_entities_per_chunk=645
        → 4 entity chunks) and 1286 key_terms (max_key_terms_per_chunk=215 → 6 chunks),
        the old code set n_chunks=6, producing 2 chunks with entities=[] — wasted LLM
        calls where the judge had nothing to score.

        Fix: only entities (and resources/aligned/judge_resource) drive n_chunks.
        key_terms are reference data distributed across entity-driven chunks.
        """
        # Reproduce exact production numbers: 2130 entities, 1286 key_terms
        payload = _make_payload(n_entities=2130, n_key_terms=1286)
        chunks = split_structured_payload(
            payload,
            max_entities_per_chunk=645,
            max_key_terms_per_chunk=215,
        )
        # Entity-driven: ceil(2130/645) = 4 chunks
        assert len(chunks) == 4, (
            f"Expected 4 entity-driven chunks, got {len(chunks)}. "
            "key_terms must not inflate chunk count."
        )
        # Every chunk must have at least one entity
        for i, chunk in enumerate(chunks):
            assert len(chunk.get("entities") or []) > 0, (
                f"Chunk {i} has empty entities — key_terms inflated chunk count"
            )


# ---------------------------------------------------------------------------
# Integration: split → unify ontology
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_round_trip_ontology_consistency(self):
        """Simulate parallel alignment producing inconsistent ontology IDs for the same
        entity text, then verify unify_ontology_across_entities fixes them."""
        # Two 'chunks' processed by different alignment LLM calls
        chunk_a_entities = [
            {"entity": "hippocampus", "label": "brain region",
             "ontology_id": "UBERON:0001954", "concept_mapping_provenance": "tool"},
            {"entity": "CA1", "label": "brain region",
             "ontology_id": "UBERON:0003881", "concept_mapping_provenance": "tool"},
        ]
        chunk_b_entities = [
            {"entity": "hippocampus", "label": "brain region",
             "ontology_id": "UBERON:0002421", "concept_mapping_provenance": "llm_knowledge"},
            {"entity": "dentate gyrus", "label": "brain region",
             "ontology_id": "UBERON:0001885", "concept_mapping_provenance": "tool"},
        ]

        merged_chunk_a = {"entities": chunk_a_entities, "key_terms": []}
        merged_chunk_b = {"entities": chunk_b_entities, "key_terms": []}
        merged = merge_structured_chunk_results([merged_chunk_a, merged_chunk_b])

        # Before unification hippocampus has two different IDs
        hippo_ids = {e["ontology_id"] for e in merged["entities"]
                     if e["entity"] == "hippocampus"}
        assert len(hippo_ids) == 2

        # After unification all occurrences must share the tool-backed ID
        merged["entities"] = unify_ontology_across_entities(merged["entities"])
        hippo_ids_after = {e["ontology_id"] for e in merged["entities"]
                           if e["entity"] == "hippocampus"}
        assert hippo_ids_after == {"UBERON:0001954"}  # tool-backed wins

        # Total entity count preserved
        assert len(merged["entities"]) == 4
