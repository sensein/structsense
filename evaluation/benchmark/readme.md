# Benchmarking Dataset

This directory provides benchmarking datasets and StructSense results for evaluating NER performance.

The dataset includes named entity recognition (NER) annotations (test set) from widely used biomedical corpora:
- **NCBI Disease** (`ncbi/`) — disease entities 
- **S800** (`s800/`) — species entities 

Original dataset: [https://drive.google.com/drive/u/1/folders/11xgW5F_z6F3ePARTbNDA9InpBScIyNte](https://drive.google.com/drive/u/1/folders/11xgW5F_z6F3ePARTbNDA9InpBScIyNte)

## Repository Structure

Each dataset is organized in its own subdirectory with the following layout:

```
benchmark/
├── <dataset>/                         # e.g. ncbi, jnlpba, s800, bc5cdr
│   ├── *.txt                          # Plain text input to StructSense
│   ├── *.jsonl                        # Ground-truth annotations (sentence-level)
│   └── results/
│       ├── staged_hil/                # Stage-by-stage outputs — with human-in-the-loop
│       │   ├── 00_extractor_agent_extraction_task.json #note name might vary
│       │   ├── 01_alignment_agent_alignment_task.json
│       │   └── 02_judge_agent_judge_task.json
│       ├── staged_nhil/               # Stage-by-stage outputs — without human-in-the-loop
│       │   ├── 00_extractor_agent_extraction_task.json
│       │   ├── 01_alignment_agent_alignment_task.json
│       │   └── 02_judge_agent_judge_task.json
│       └── <dataset>_result_<hil or non_hil>.json      # Final merged StructSense output for human and non-human loop.
├── script/
│   └── bio_txt.py                     # Convert BIO-tagged data → *.txt + *.jsonl
└── ner-config-gpt.yaml                # StructSense config used for benchmark runs
```

> `staged_hil` and `staged_nhil` will be present for **all** dataset subdirectories.

### File types

| File | Description |
|---|---|
| `*.txt` | Plain text input fed to StructSense |
| `*.jsonl` | Ground-truth annotations in JSON Lines format, one sentence per line |
| `results/staged_hil/` | Intermediate stage outputs from a run **with** human-in-the-loop feedback |
| `results/staged_nhil/` | Intermediate stage outputs from a run **without** human-in-the-loop feedback |
| `script/bio_txt.py` | Utility to convert BIO-tagged corpora into `*.txt` + `*.jsonl` format |

## Source

Original data taken from BioNLP-Corpus: https://github.com/bionlp-hzau/BioNLP-Corpus/tree/master
