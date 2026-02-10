## Overview

This folder contains configuration files used to run different tasks. It includes a base template that can be adapted as needed, along with task-specific configuration files.

### Contents
- `config.yaml`: a configuration template
- `ner-config.yaml`: a configuration file that can be used for the NER task
- `resource-extraction-config.yaml`: a configuration file that can be used for the resource extraction task
- `pdf2_reproschema.yaml`: Converts a survey questionnaire PDF into JSON-LD. See [Pdf2ReproSchema](../example/pdf2_reproschema) for details. Generated files are written to the [outputs](outputs) directory. The input PDF used for this workflow is located in the `Pdf2ReproSchema` directory.


#### ⚠️ Important Notes

- **Do not rename** predefined YAML keys such as `task_config` and `agent_config`.
  You can update: agent descriptions, task descriptions, and `embedder_config`.   

- **Do not replace variables** enclosed in curly braces (`{}`); they are dynamically populated at runtime. Names must match the pipeline input map (see `config_template` for examples):
  - **Extraction input:** `{input_text}` — input text (e.g. PDF content or raw text)
  - **Alignment input:** `{extracted_structured_information}` — output from the extractor agent
  - **Judge input:** `{aligned_structured_information}` — output from the alignment agent
  - **Human feedback input:** `{judged_structured_information_with_human_feedback}` — output from the judge agent; `{modification_context}` and `{user_feedback_text}` — user feedback for the feedback agent

### Agent Configuration

The following agents should not be renamed or removed:
- `extractor_agent`
- `alignment_agent`
- `judge_agent`
- `humanfeedback_agent`



Each agent should be configured with the following fields: `role`, `goal`, `backstory`, and `llm`.

For best practices, refer to the [Crew AI Core Principles of Effective Agent Design](https://docs.crewai.com/guides/agents/crafting-effective-agents#core-principles-of-effective-agent-design).

```yaml
agent_config:
  extractor_agent:
    role: >
      agent role
    goal: >
      goal
    backstory: >
      agent backstory
    llm:
      model: openrouter/openai/gpt-4o-mini
      base_url: https://openrouter.ai/api/v1

  alignment_agent:
    ...
```
### Using Ollama
In the snippet above, we use the openai/gpt-4o-mini model via OpenRouter. If you prefer to use open-source models with Ollama, you'll need to update the model and base URL accordingly. This approach is especially useful as it doesn't require an API key from paid providers like OpenRouter or OpenAI. However, you must ensure that Ollama is running and that the desired model is installed and available locally.
```yaml
agent_config:
  extractor_agent:
    role: >
      agent role
    goal: >
      goal
    backstory: >
      agent backstory
    llm:
      model: ollama/deepseek-r1:14b #notice the difference
      base_url: http://localhost:11434 #notice the difference

  alignment_agent:
    ...
```

### 🧾 Task Configuration

Each task corresponds to a specific agent and must not be renamed:

- `extraction_task`
- `alignment_task`
- `judge_task`
- `humanfeedback_task`

Each task should include:

- **`description`**:
  A detailed explanation of the task, including the required input (e.g., `{literature}` for extraction, `{extracted_structured_information}` for alignment, etc.).

- **`expected_output`**:
  The expected output format. The format must be JSON. You may specify the structure or give an example.

- **`agent_id`**:
  This key assigns the task to its corresponding agent. The value must match the agent ID defined under `agent_config`.

Example:
```yaml
task_config:
  extraction_task:
    description: >
      Extract structured information from the given literature.
      Input: {literature}
    expected_output: >
      Format: JSON
      Example: {"entities": [...], "relations": [...]}
    agent_id: extractor_agent
```

To learn more about the tasks, see [Crafting Effective Tasks for Your Agents](https://docs.crewai.com/guides/agents/crafting-effective-agents#crafting-effective-tasks-for-your-agents).
### 👤 Human-in-the-Loop (Human Feedback)
Human feedback is **off by default**. Enable it with:

```bash
ENABLE_HUMAN_FEEDBACK=true
```

### Embedding Configuration
Defines the embedding model used for memory (e.g. RAG) when **Crew memory** is enabled. Use this to avoid defaulting to OpenAI (which requires `OPENAI_API_KEY`). Example with Ollama:
```yaml
embedder_config:
  provider: ollama
  config:
    api_base: http://localhost:11434
    model: nomic-embed-text  # or nomic-embed-text:v1.5
```