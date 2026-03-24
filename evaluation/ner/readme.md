# 🧠 Named Entity Recognition (NER) Pipeline

This project uses `structsense` to automate the extraction of named entities from neuroscience papers, focusing on identifying and aligning anatomical regions, experimental conditions, cell types, and other domain-specific entities.

---

## 🔬 What This System Does

Given a neuroscience paper (PDF), this pipeline:

1. **Extracts named entities** such as anatomical regions, cell types, and experimental conditions
2. **Aligns entities** with neuroscience ontologies and structured vocabularies
3. **Evaluates alignment quality** with scoring from 0-1
4. **Incorporates human feedback** to improve entity recognition and alignment

---

## 🧑‍💼 Agents & Their Roles

| Agent Name           | Description |
|----------------------|-------------|
| `extractor_agent`    | Performs NER on neuroscience literature, processing by paragraph and extracting structured entities |
| `alignment_agent`    | Aligns extracted entities with ontological terms and structured vocabularies |
| `judge_agent`        | Evaluates alignment quality and assigns accuracy scores (0-1) |
| `humanfeedback_agent`| Processes human feedback to refine entity recognition and alignment |

---

## 📁 Directory Structure

The system creates one folder per paper, with the folder name derived from the paper title. For example, the paper `Multiscale Spatial Transcriptomic Atlas of Human Basal Ganglia Cell-Type and Cellular Community Organization` may be stored in a folder such as `paper_discovery_of_optimal_cell`.

Each paper-specific folder includes:
- the source PDF(s), such as the publication itself
- configuration files
- result subdirectories following the naming convention `results-<model>`

Each `results-<model>` subdirectory contains:
- output JSON files for both human-in-the-loop and non-human-in-the-loop executions
- staged outputs for intermediate pipeline steps

The staged outputs are organized into:
- `staged_hil` for human-in-the-loop execution
- `staged_nhil` for non-human-in-the-loop execution

## 📁 Expected Output

The pipeline generates both intermediate and final outputs in **JSON** format:
- **staged output files** for intermediate pipeline steps
- a **final output file** in the user-specified format

The final output includes:
- extracted entities
- the corresponding source sentence for each entity
- the ontology concept(s) each entity is aligned to
- judge remarks, when available

## Papers Used

The following papers were used to evaluate the pipeline:

- [Discovery of Optimal Cell Type Classification Marker Genes from Single-Cell RNA Sequencing Data](https://www.biorxiv.org/content/10.1101/2024.04.22.590194v2)  
  Result directory: `paper_discovery_of_optimal_cell`

- [Latent Circuit Inference from Heterogeneous Neural Responses During Cognitive Tasks](https://www.nature.com/articles/s41593-025-01869-7)  
  Result directory: `latent-circuit-inference`

## Command Used to Run `structsense`

Make sure to replace placeholders such as `outputfile`, `config_name.yaml`, `input_pdf.pdf`, and the API key with appropriate values.

```bash
structsense-cli extract \
  --env_file .env \
  --save_file outputfile.json \
  --chunk_size 600 \
  --max_workers 8 \
  --enable_chunking \
  --config config_name.yaml \
  --source input_pdf.pdf \
  --api_key sk-<your_api_key>
```