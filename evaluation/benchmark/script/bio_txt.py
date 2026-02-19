#!/usr/bin/env python3
"""
Convert a (T)SV with BIO tags into:
  1) one continuous text file (all sentences concatenated into one text block)
  2) one entities/mapping file (JSONL)

Supports:
- Delimiter auto-detect per line: TAB or COMMA (token<TAB>tag OR token,tag)
- Tags can be either:
    * O / B / I   (no type)  -> entity type will be "ENTITY"
    * O / B-XXX / I-XXX      -> entity type will be XXX
- Sentence boundaries:
    * blank line
    * a line that is just "," (common in some corpora exports)

Outputs:
- OUT_TEXT: single continuous text block (sentences separated by one space)
- OUT_ENTS_JSONL: JSONL (one record per sentence) with entities and global offsets
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional



IN_FILE = Path("JNLPBA_gene_protein_test.tsv")
OUT_TEXT = Path("JNLPBA_gene_protein_test_text.txt")
OUT_ENTS_JSONL = Path("JNLPBA_gene_protein_test_entities_mapping.jsonl")

# Column indices after splitting line into fields
TOKEN_COL = 0
TAG_COL = 1


def split_fields(line: str) -> List[str]:
    """
    Auto-detect delimiter: prefer TAB if present, else comma.
    """
    if "\t" in line:
        return line.split("\t")
    return line.split(",")


def is_sentence_break(parts: List[str], raw_line: str) -> bool:
    """
    Decide if this line indicates sentence boundary.
    """
    if not raw_line.strip():
        return True
    stripped = raw_line.strip()
    if stripped == ",":
        return True
    # If only one column and it's just a comma, treat as break
    if len(parts) == 1 and parts[0].strip() == ",":
        return True
    return False


def parse_tag(tag: str) -> Tuple[str, str]:
    """
    Returns (bio, etype)
      - bio in {"B","I","O"}
      - etype is "" if unknown
    Handles:
      O
      B / I
      B-XXX / I-XXX
    Note: we just mark every entity as Entity
    """
    tag = tag.strip()
    if not tag or tag == "O":
        return "O", ""

    if tag == "B":
        return "B", "ENTITY"
    if tag == "I":
        return "I", "ENTITY"

    if tag.startswith("B-"):
        return "B", tag.split("-", 1)[1] or "ENTITY"
    if tag.startswith("I-"):
        return "I", tag.split("-", 1)[1] or "ENTITY"

    # Unknown format -> treat as O
    return "O", ""


def read_bio_file(path: Path) -> Tuple[List[List[str]], List[List[Tuple[str, str]]]]:
    """
    Returns:
      sentences_tokens: List[List[token]]
      sentences_tags:   List[List[(bio, etype)]]
    """
    sentences_tokens: List[List[str]] = []
    sentences_tags: List[List[Tuple[str, str]]] = []

    cur_tokens: List[str] = []
    cur_tags: List[Tuple[str, str]] = []

    def flush():
        nonlocal cur_tokens, cur_tags
        if cur_tokens:
            sentences_tokens.append(cur_tokens)
            sentences_tags.append(cur_tags)
            cur_tokens = []
            cur_tags = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n")
            parts = split_fields(raw)

            if is_sentence_break(parts, raw):
                flush()
                continue

            # Need at least token + tag
            if len(parts) <= max(TOKEN_COL, TAG_COL):
                # If the line is just punctuation delimiter in its own column,
                # you can decide to flush. Otherwise it is malformed.
                only = parts[0].strip() if parts else ""
                if only == ",":
                    flush()
                    continue
                raise ValueError(
                    f"Line {lineno}: not enough columns for TOKEN_COL={TOKEN_COL}, TAG_COL={TAG_COL}: {raw!r}"
                )

            token = parts[TOKEN_COL].strip()
            tag_raw = parts[TAG_COL].strip()

            bio, etype = parse_tag(tag_raw)

            cur_tokens.append(token)
            cur_tags.append((bio, etype))

    flush()
    return sentences_tokens, sentences_tags


def token_starts_in_joined(tokens: List[str]) -> List[int]:
    """
    For sentence_text = " ".join(tokens), compute start char for each token.
    """
    starts: List[int] = []
    pos = 0
    for i, t in enumerate(tokens):
        starts.append(pos)
        pos += len(t)
        if i != len(tokens) - 1:
            pos += 1  # space
    return starts


def bio_to_entities(tokens: List[str], tags: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Build entity spans (relative to sentence text).
    """
    sentence = " ".join(tokens)
    starts = token_starts_in_joined(tokens)
    entities: List[Dict[str, Any]] = []

    i = 0
    while i < len(tokens):
        bio, etype = tags[i]
        if bio == "O":
            i += 1
            continue

        # Start entity on B, or treat stray I as B
        if bio in ("B", "I"):
            start_tok = i
            end_tok = i
            cur_type = etype or "ENTITY"

            j = i + 1
            while j < len(tokens):
                bio_j, etype_j = tags[j]
                if bio_j != "I":
                    break
                # If types exist and mismatch, stop; if no types, keep going
                next_type = etype_j or cur_type
                if etype and etype_j and next_type != cur_type:
                    break
                end_tok = j
                j += 1

            char_start = starts[start_tok]
            char_end = starts[end_tok] + len(tokens[end_tok])
            text = sentence[char_start:char_end]

            ent: Dict[str, Any] = {
                "text": text,
                "type": cur_type,
                "char_start": char_start,
                "char_end": char_end,
                "token_start": start_tok,
                "token_end": end_tok,
            }
            if bio == "I":
                ent["note"] = "started_with_I"
            entities.append(ent)

            i = j
            continue

        i += 1

    return entities


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    OUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_ENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    sentences_tokens, sentences_tags = read_bio_file(IN_FILE)

    sentence_texts: List[str] = [" ".join(toks).replace("\n", " ").strip() for toks in sentences_tokens]
    sentence_texts = [s for s in sentence_texts if s]

    # One continuous text block
    global_text = " ".join(sentence_texts)
    OUT_TEXT.write_text(global_text, encoding="utf-8")

    # Mapping with global offsets
    global_cursor = 0
    with OUT_ENTS_JSONL.open("w", encoding="utf-8") as f:
        for sid, (toks, tags, sent_text) in enumerate(zip(sentences_tokens, sentences_tags, sentence_texts)):
            ents = bio_to_entities(toks, tags)

            sent_start_global = global_cursor
            sent_end_global = sent_start_global + len(sent_text)

            ents_global = []
            for e in ents:
                ents_global.append(
                    {
                        "text": e["text"],
                        "type": e["type"],
                        "char_start": sent_start_global + int(e["char_start"]),
                        "char_end": sent_start_global + int(e["char_end"]),
                        "token_start": e.get("token_start"),
                        "token_end": e.get("token_end"),
                        "note": e.get("note"),
                    }
                )

            rec = {
                "sentence_id": sid,
                "sentence_text": sent_text,
                "sentence_char_start_global": sent_start_global,
                "sentence_char_end_global": sent_end_global,
                "entities": ents,
                "entities_global": ents_global,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # move cursor by sentence length + 1 space (except last; harmless if +1)
            global_cursor = sent_end_global + 1

    print("Done.")
    print(f"- Wrote continuous text: {OUT_TEXT}")
    print(f"- Wrote entities mapping: {OUT_ENTS_JSONL}")
    print(f"- Sentences: {len(sentence_texts)}")


if __name__ == "__main__":
    main()
