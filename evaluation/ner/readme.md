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

## 📁 Expected Output Structure

The pipeline produces a JSON file containing the final evaluation from the judge agent.

Key points:
1. **Format**: Always a JSON file
2. **Content**: Depends on your configuration in `task_config.judge_task.expected_output`

The output structure includes:
- Extracted entities with their types
- Ontological alignments (IDs and labels)
- Confidence scores per occurrence
- Quality assessment remarks

---

## 📚 Example Papers

The NER pipeline has been tested with the following Nature Neuroscience papers:

1. Langdon, C., Engel, T.A. Latent circuit inference from heterogeneous neural responses during cognitive tasks. Nat Neurosci 28, 665–675 (2025).
2. Hansen, J.Y., Cauzzo, S., Singh, K. et al. Integrating brainstem and cortical functional architectures. Nat Neurosci 27, 2500–2511 (2024).
3. Oby, E.R., Degenhart, A.D., Grigsby, E.M. et al. Dynamical constraints on neural population activity. Nat Neurosci 28, 383–393 (2025).
---

## 🧪 Usage

### Using OpenRouter
```bash
structsense-cli extract \
  --source your_neuroscience_paper.pdf \
  --api_key <YOUR_API_KEY> \
  --config config.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Using Ollama (Local)
```bash
structsense-cli extract \
  --source your_neuroscience_paper.pdf \
  --config config.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Tips

- The configuration file (`config.yaml`) in this directory is pre-configured for neuroscience NER
- Users should provide their own neuroscience papers as PDFs when running the commands
- The system expects at least 100 entities from large texts for comprehensive extraction
- Entity extraction includes position tracking (start/end indices) and paper location metadata

---

## 🔍 Entity Types

Based on the configuration, the system extracts:

- **Animal species**: mouse, drosophila, zebrafish, etc.
- **Brain/anatomical regions**: neocortex, mushroom body, cerebellum, hippocampus, etc.
- **Experimental conditions**: control, tetrodotoxin treatment, Scn1a knockout, etc.
- **Cell types**: pyramidal neuron, direction-sensitive mechanoreceptor, oligodendrocyte, etc.

Each entity is mapped to relevant ontologies such as:
- NCBITaxon (for species)
- UBERON (for anatomical structures)
- CL (Cell Ontology)
- Custom neuroscience vocabularies

---

## 🤖 Language Models

The current `config.yaml` specifies `openrouter/openai/gpt-4o-mini` as the default LLM. However, for evaluation purposes, this pipeline has been tested with:

- **GPT-4o-mini** by OpenAI
- **Claude 3.5 Sonnet** by Anthropic
- **DeepSeek V3** (open-source model) by DeepSeek

You can modify the LLM settings in the `agent_config` section of `config.yaml` to use different models based on your requirements.