# Benchmarking Dataset

This repository provides a benchmarking dataset for evaluating **StructSense**.

The dataset includes named entity recognition (NER) annotations (test set) from two widely used biomedical corpora:
- **NCBI Disease** dataset (disease entities)
- **JNLPBA** dataset (gene and protein entities)
  
Original dataset: [https://drive.google.com/drive/u/3/folders/1R3_z3pv7ELtlJhwnophKEaOO0h7omjud](https://drive.google.com/drive/u/3/folders/1R3_z3pv7ELtlJhwnophKEaOO0h7omjud)

## Repository Structure

- `*.txt` — Plain text files used as input to StructSense.
- `*.jsonl` — Ground-truth annotations in JSON Lines format, containing sentence-level entity information.
- `script/bio_txt.py` — Utility script for converting BIO-tagged data into the corresponding `*.txt` and `*.jsonl` formats.

## Source
This original data is taken from BioNLP-Corpus at https://github.com/bionlp-hzau/BioNLP-Corpus/tree/master.
