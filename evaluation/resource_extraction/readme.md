# 🧠 BBQS Resource Extraction Pipeline

This project uses `structsense` to extract and curate structured metadata about scientific resources relevant to the Brain Behavior Quantification and Synchronization (BBQS) consortium. The pipeline focuses on tools, datasets, models, and benchmarks that advance understanding of brain-behavior relationships.

---

## 🔬 What This System Does

Given a scientific paper, webpage, or resource description, this pipeline:

1. **Extracts structured metadata** about a single primary resource (name, type, category, target species, URL)
2. **Captures secondary mentions** of related datasets, models, benchmarks, or tools referenced within the resource
3. **Aligns target species** with standardized ontologies (e.g., NCBITaxon)
4. **Evaluates extraction quality** with confidence scores from 0-1
5. **Incorporates human feedback** to refine and validate the extracted information

---

## 🧑‍💼 Agents & Their Roles

| Agent Name           | Description |
|----------------------|-------------|
| `extractor_agent`    | Extracts one structured resource entry per input, capturing related resources as secondary mentions |
| `alignment_agent`    | Aligns metadata fields with controlled vocabularies and maps species to ontology terms |
| `judge_agent`        | Evaluates extraction quality and BBQS standard compliance with confidence scores |
| `humanfeedback_agent`| Reviews and refines results based on domain expert feedback |

---

## 📁 Expected Output Structure

The pipeline produces a JSON file containing the final evaluation from the judge agent.

Key points:
1. **Format**: Always a JSON file
2. **Content**: One resource per input with the following structure:
   - Primary resource metadata (name, description, type, category, targets)
   - Ontological mappings for target species
   - Optional mentions of related resources
   - Confidence scores and evaluation rationale

Resource types include:
- **Models**: Pose estimation models, embedding models, etc.
- **Datasets**: Annotated video data, behavioral recordings
- **Papers**: Methods or applications for behavioral quantification
- **Tools**: Analysis software, labeling interfaces
- **Benchmarks**: Standardized evaluation datasets or protocols
- **Leaderboards**: Model performance ranking systems

---

## 📚 Example Papers

1. Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022). Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35, 38571-38584.
2. Lauer, J., Zhou, M., Ye, S. et al. Multi-animal pose estimation, identification and tracking with DeepLabCut. Nat Methods 19, 496–504 (2022). https://doi.org/10.1038/s41592-022-01443-0 
---

## 🧪 Usage

### Using OpenRouter
```bash
structsense-cli extract \
  --source your_resource_paper.pdf \
  --api_key <YOUR_API_KEY> \
  --config <config-file>.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Using Ollama (Local)
```bash
structsense-cli extract \
  --source your_resource_paper.pdf \
  --config <config-file>.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Tips

- The configuration file (`config.yaml`) is pre-configured for BBQS resource extraction
- Each input should describe a single primary resource
- Secondary resources mentioned within the primary resource are captured in the `mentions` field
- Best suited for papers, webpages, or descriptions about behavioral quantification tools and datasets

---

## 🔍 Resource Categories

The system extracts resources in these categories:

### Primary Categories
- **Pose Estimation**: Models and tools for tracking animal/human body positions
- **Gaze Detection**: Eye tracking and visual attention systems
- **Behavioral Quantification**: General behavior analysis frameworks
- **Motion Analysis**: Movement pattern detection and classification

### Target Species
Automatically mapped to ontologies:
- **General**: Animal, Human, Mammals
- **Specific**: Mice (NCBITaxon:10090), Macaque (NCBITaxon:9541), Dogs (NCBITaxon:9615), etc.

### Secondary Mentions
Resources referenced within the primary resource:
- Related models and architectures
- Training/evaluation datasets
- Benchmark datasets
- Associated tools or software

---

## 🤖 Language Models

The current `config.yaml` specifies `openrouter/openai/gpt-4o-mini` as the default LLM. However, for evaluation purposes, this pipeline has been tested with:

- **GPT-4o-mini** by OpenAI
- **Claude 3.5 Sonnet** by Anthropic
- **DeepSeek V3** (open-source model) by DeepSeek

You can modify the LLM settings in the `agent_config` section of `config.yaml` to use different models based on your requirements.