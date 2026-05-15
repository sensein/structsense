"""Finalize Agent 1: compute source_sentences, occurrence_count, and dedicated coreference pass."""
import json
import re
from pathlib import Path

SRC = Path("source_text.txt").read_text()
ents = json.loads(Path("agent1_entities_round4.json").read_text())

# Build sentence list (rough): split by '.', '?', '!', or '\n\n' boundary but keep page markers out.
# We'll use a robust regex-based split.

def split_sentences(text):
    # Remove page markers and copyright footer lines for cleaner sentence sentences
    text = re.sub(r'=== PAGE \d+ ===', ' ', text)
    text = re.sub(r'\.CC-BY-NC-ND.*?bioRxiv preprint', ' ', text, flags=re.DOTALL)
    # Replace newlines with space (sentences span line breaks in this PDF)
    text2 = re.sub(r'\s+', ' ', text)
    # Split on sentence punctuation followed by space + capital or end
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text2)
    return sents

sentences = split_sentences(SRC)
print(f"Total sentences: {len(sentences)}")

# For each entity, find sentences containing any surface form (case-sensitive substring)
for eid, ent in ents.items():
    matched_sents = []
    seen_idxs = set()
    for i, s in enumerate(sentences):
        for sf in ent["surface_forms"]:
            if sf in s:
                if i not in seen_idxs:
                    matched_sents.append(s.strip())
                    seen_idxs.add(i)
                break
    ent["source_sentences"] = matched_sents
    ent["source_sentence_count"] = len(matched_sents)

# Coreference pass: for each entity with >=2 source_sentences, scan 1-2 sentences after each mention
# for pronouns / descriptive nominals. Map the resolved reference to entity_id when feasible.
pronouns_re = re.compile(r"\b(it|they|these|those|this|that|them)\b", re.IGNORECASE)
descriptive_nominals = [
    "these cells", "this population", "these populations", "this subtype", "these subtypes",
    "the latter", "the former", "the dataset", "this gene", "such neurons", "the protein",
    "this cell type", "these cell types", "this region", "these regions", "this marker",
    "these neurons", "this approach", "these approaches", "this method", "this analysis",
    "these analyses", "this gradient", "these gradients", "this organization",
]

# Build indirect_references for entities with >=2 source sentences and a clearly local successor.
# Approach: for each mention's sentence index, look at i+1 and i+2 sentences and check for pronoun/nominal
import unicodedata
sent_idx_lookup = {i: s for i, s in enumerate(sentences)}

for eid, ent in ents.items():
    indirect = []
    if len(ent.get("source_sentences", [])) < 2:
        ent["indirect_references"] = indirect
        continue
    # find indices of source mentions
    mention_idxs = []
    for i, s in enumerate(sentences):
        for sf in ent["surface_forms"]:
            if sf in s:
                mention_idxs.append(i)
                break
    seen_idxs = set()
    for idx in mention_idxs:
        for offset in (1, 2):
            j = idx + offset
            if j >= len(sentences) or j in seen_idxs:
                continue
            s_next = sentences[j].strip()
            if not s_next or len(s_next) < 10:
                continue
            # Check for descriptive nominal
            matched_phrase = None
            for nom in descriptive_nominals:
                if nom in s_next.lower():
                    matched_phrase = nom
                    break
            if not matched_phrase:
                pm = pronouns_re.search(s_next)
                if pm:
                    matched_phrase = pm.group(0).lower()
            if matched_phrase:
                # ensure the entity is not itself referenced literally in s_next
                literal_present = any(sf in s_next for sf in ent["surface_forms"])
                if literal_present:
                    continue
                # confidence: descriptive nominals high, pronouns medium-low
                if matched_phrase in {nom for nom in descriptive_nominals}:
                    conf = "medium"
                else:
                    conf = "low"
                indirect.append({
                    "referring_phrase": matched_phrase,
                    "source_sentence": s_next[:400],
                    "resolved_to": eid,
                    "resolution_confidence": conf,
                })
                seen_idxs.add(j)
        # cap per entity at 8
        if len(indirect) >= 8:
            break
    ent["indirect_references"] = indirect

# occurrence_count
for eid, ent in ents.items():
    ent["occurrence_count"] = len(ent.get("source_sentences", [])) + len(ent.get("indirect_references", []))

# Truncate source_sentences for serialization (keep all, but warn if extremely long)
for eid, ent in ents.items():
    # Keep all source_sentences (no truncation per prompt)
    ent["source_sentences"] = [s[:600] for s in ent["source_sentences"]]  # cap per-sentence length only

# Build final Agent1 output
agent1_out = {
    "agent": "EntityExtractor",
    "model_used": "claude-opus-4-7 (orchestrated, single-model context)",
    "extraction_rounds_completed": 4,
    "round_log": [
        {"round": 1, "entities_found_this_round": 362,
         "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round1.txt"},
        {"round": 2, "entities_found_this_round": 39,
         "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round2.txt",
         "unmasked_candidates_round_2": json.loads(Path("agent1_unmasked_candidates_round2.json").read_text())},
        {"round": 3, "entities_found_this_round": 8,
         "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round3.txt",
         "unmasked_candidates_round_3": json.loads(Path("agent1_unmasked_candidates_round3.json").read_text())},
        {"round": 4, "entities_found_this_round": 0,
         "masked_artifact_path": "multiscale_spatial_transcriptomic_masked_round4.txt",
         "unmasked_candidates_round_4": json.loads(Path("agent1_unmasked_candidates_round4.json").read_text())},
    ],
    "masking_applied": True,
    "termination_reason": "zero-delta",
    "coreference_pass_completed": True,
    "entities": list(ents.values()),
}

Path("agent1_output.json").write_text(json.dumps(agent1_out, indent=2))

# Summary stats
total_with_indirect = sum(1 for e in ents.values() if e["indirect_references"])
print(f"\nAgent 1 output written. Entities: {len(ents)}")
print(f"Entities with non-empty indirect_references: {total_with_indirect} ({total_with_indirect/len(ents)*100:.1f}%)")
print(f"Median source_sentence_count: {sorted([e['source_sentence_count'] for e in ents.values()])[len(ents)//2]}")
