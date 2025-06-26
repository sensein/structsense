# 🧠 ReproSchema Conversion Pipeline

This project uses `structsense` to automate the extraction, transformation, and validation of questionnaire content from PDF files into ReproSchema-compatible schema files. The goal is to standardize survey-based data collection workflows by producing machine-readable, interoperable, and reusable JSON-LD representations of assessments.

---

## 🔧 What This System Does

Given a questionnaire PDF, this pipeline:

1. **Extracts structured information** (e.g., items, response options, languages, scoring) from the raw document.
2. **Generates a ReproSchema-compatible folder structure**, including:
   - `items/`: individual question files (`item.jsonld`)
   - `activities/`: composite assessment (`activity.jsonld`)
   - `activity_schema.jsonld`: metadata and UI behavior
3. **Evaluates schema fidelity** against the original questionnaire.
4. **Applies human-in-the-loop feedback** to revise and improve schema quality.

---

## 🧑‍💼 Agents & Their Roles

| Agent Name           | Description |
|----------------------|-------------|
| `extractor_agent`    | Parses the questionnaire PDF and extracts questions, options, logic, and metadata. |
| `alignment_agent`    | Converts the extracted structure into a ReproSchema JSON-LD folder layout. |
| `judge_agent`        | Evaluates whether the generated schema matches the source content and highlights issues. |
| `humanfeedback_agent`| Applies human-provided feedback to revise and align the schema output more accurately. |

---

## 📁 Expected Output Structure

The pipeline produces a JSON file containing the final output from the judge agent. 

Key points about the output:
1. **Format**: The output is always a JSON file
2. **Content**: The specific structure and content of the JSON depends on how you define the `expected_output` in the judge task configuration

For example, if your config specifies the judge agent to evaluate ReproSchema compliance, the output might include:
- Schema validation results
- Fidelity scores
- Identified issues or discrepancies
- Recommendations for improvements

The exact JSON structure is fully customizable through your configuration file's `task_config.judge_task.expected_output` field.

---

## 🧠 Human-in-the-Loop

After the automated processing and evaluation steps, human feedback can be used to:
- Correct extraction mistakes
- Improve ontology mappings (e.g., add SSSOM-aligned terms)
- Clarify ambiguous response options or language labels
- Trigger regeneration of files with updated logic

---

## 🤖 Language Models

The configuration specifies `openrouter/openai/gpt-4o-mini` as the default LLM. However, for evaluation purposes, this pipeline has been tested with:

- **GPT-4o-mini** by OpenAI
- **Claude 3.5 Sonnet** by Anthropic
- **DeepSeek V3** (open-source model) by DeepSeek

You can modify the LLM settings in the `agent_config` section of your configuration file to use different models based on your requirements.

---

## 🧪 Usage

### Using OpenRouter
```bash
structsense-cli extract \
  --source sample_pdf.pdf \
  --api_key <YOUR_API_KEY> \
  --config config.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Using Ollama (Local)
```bash
structsense-cli extract \
  --source sample_pdf.pdf \
  --config config.yaml \
  --env_file .env \
  --save_file result.json  # optional
```

### Tips

- Best used with source questionnaires similar to PHQ-9, GAD-7, or eCOBIDAS formats
- The configuration file should define the expected ReproSchema structure
- Human-in-the-loop feedback can significantly improve schema accuracy

---

## 📚 References

- [ReproSchema Documentation](https://www.repronim.org/reproschema/)
- [CrewAI Docs](https://docs.crewai.com/)
- [LinkML](https://linkml.io)
- [FAIR Principles](https://www.go-fair.org/fair-principles/)
