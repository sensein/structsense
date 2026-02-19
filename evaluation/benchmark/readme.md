# Benchmarking Dataset

This repository provides a benchmarking dataset for evaluating **StructSense**.

The dataset includes named entity recognition (NER) annotations from two widely used biomedical corpora:
- **NCBI Disease** dataset (disease entities)
- **JNLPBA** dataset (gene and protein entities)

## Repository Structure

- `*.txt` — Plain text files used as input to StructSense.
- `*.jsonl` — Ground-truth annotations in JSON Lines format, containing sentence-level entity information.
- `script/bio_txt.py` — Utility script for converting BIO-tagged data into the corresponding `*.txt` and `*.jsonl` formats.
