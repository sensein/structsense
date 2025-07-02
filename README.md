# 🧩 StructSense

Welcome to `structsense`!

`structsense` is a powerful multi-agent system designed to extract structured information from unstructured data. By orchestrating intelligent agents, it helps you make sense of complex information — hence the name *structsense*.

Whether you're working with scientific texts, documents, or messy data, `structsense` enables you to transform it into meaningful, structured insights.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.


---
## 🏗️ Architecture
The below is the architecture of the `StructSense`.

![](structsense_arch.png)
---
## 🚀 Features

- 🔍 Multi-agent architecture for modular processing
  - 📑 Extraction of (structured) information from text--based on configuration
  - 🤝 Collaboration between agents
  - ⚙️ Easy use
  - 🧠 Designed as general purpose domain agnostic framework

---

## 🧠 Example Use Cases
- Entity and relation extraction from text
  - Knowledge graph construction
## 📁 Examples

- Explore the [`example`](example) directory for usage examples and reference configurations.

## 📄 Blank Configuration Template

- Refer to the [`config_template`](config_template) directory for a blank configuration template.
  Before using it, make sure to read the detailed explanation of the configuration template provided in the sections below (`Configuration Template`).

---
### Installation
After cloning the repository.
```shell
pip install -e .
```
## 📄 Requirements
### 📄 PDF Extraction Configuration

By default, the system uses the **local Grobid service** for PDF content extraction. If you have Grobid installed locally, **no additional setup is required** — everything is preconfigured for local usage.

**Grobid Installation via Docker**
```shell
docker pull lfoppiano/grobid:0.8.0
docker run --init -p 8070:8070 -e JAVA_OPTS="-XX:+UseZGC" lfoppiano/grobid:0.8.0
```
JAVA_OPTS="-XX:+UseZGC" helps to resolve the following error in MAC OS.

---

#### 🔧 Using a Remote Grobid Server

If you're running Grobid on a **remote server**, set the following environment variable:

```bash
GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=http://your-remote-grobid-server:PORT

```

### 🌐 Using an External PDF Extraction API
If you prefer to use an external PDF extraction API service, you must:

- Set the API endpoint:

  ```shell
    GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=https://api.SOMEAPIENDPOINT.com/api/extract
  ```
- Enable the external API mode:
  ```shell
    EXTERNAL_PDF_EXTRACTION_SERVICE=True
  ```
> Note:
At the moment, the external API is assumed to be publicly accessible and does not require authentication (e.g., no JWT token or API key). Support for authenticated requests may be added in future versions.

---

## 📄 Configuration
`structsense` supports flexible customization through both environment variables and a YAML configuration file.

The YAML config can be passed as a parameter (e.g., `--agentconfig config/ner_agent.yaml`), allowing you to define models, agents, and behaviors specific to your use case.

### 🔧 Environment Variables

You need to set the following environment variables (e.g., in a `.env` file). WEAVIATE is a vector database that we use to store the knolwledge, which in our case is the ontology/schemas.

- WEAVIATE related environment variables are only necessary if you want to use vector database as a knowledge source.

#### 🧠 Core Keys

| Variable              | Description                                  | Default          |
|-----------------------|----------------------------------------------|------------------|
 | `ENABLE_KG_SOURCE`    | Enable access to knowledge source, i.e., vector database.| `false`|
| `WEAVIATE_API_KEY`    | **Required.** API key for Weaviate access    | —                |

#### 🌐 [Weaviate](https://weaviate.io/) Configuration
This configuration is optional and only necessary if you plan to integrate a knowledge source (e.g., a vector store) into the pipeline.

| Variable                   | Description                                  | Default   |
|---------------------------|----------------------------------------------|-----------|
| `WEAVIATE_HTTP_HOST`      | HTTP host for Weaviate                       | `localhost` |
| `WEAVIATE_HTTP_PORT`      | HTTP port for Weaviate                       | `8080`    |
| `WEAVIATE_HTTP_SECURE`    | Use HTTPS for HTTP connection (`true/false`) | `false`   |
| `WEAVIATE_GRPC_HOST`      | gRPC host for Weaviate                       | `localhost` |
| `WEAVIATE_GRPC_PORT`      | gRPC port for Weaviate                       | `50051`   |
| `WEAVIATE_GRPC_SECURE`    | Use secure gRPC (`true/false`)              | `false`   |

#### 🧪 Weaviate Timeouts

| Variable                   | Description                                  | Default   |
|---------------------------|----------------------------------------------|-----------|
| `WEAVIATE_TIMEOUT_INIT`   | Timeout for initialization (in seconds)     | `30`      |
| `WEAVIATE_TIMEOUT_QUERY`  | Timeout for query operations (in seconds)   | `60`      |
| `WEAVIATE_TIMEOUT_INSERT` | Timeout for data insertions (in seconds)    | `120`     |

#### 🤖 Ollama Configuration for WEAVIATE

| Variable              | Description                                   | Default                                 |
|-----------------------|-----------------------------------------------|-----------------------------------------|
| `OLLAMA_API_ENDPOINT` | API endpoint for Ollama model                 | `http://host.docker.internal:11434`     |
| `OLLAMA_MODEL`        | Name of the Ollama embedding model            | `nomic-embed-text`                      |

> ⚠️ **Note**:  If ollama is running in host machine and vector database, i.e., WEAVIATE, in docker, then we use `http://host.docker.internal:11434`, which is also the default value. However, if both are running in docker in the same host, use `http://localhost:11434 `.
#### 🧵 Optional: Experiment Tracking

| Variable               | Description                                                                | Default           |
|------------------------|----------------------------------------------------------------------------|-------------------|
| `ENABLE_WEIGHTSANDBIAS` | Enable [Weights & Biases](https://wandb.ai/site) monitoring (`true/false`) | `false`           |
| `ENABLE_MLFLOW`        | Enable [MLflow](https://mlflow.org/) logging (`true/false`)                | `false`           |
| `MLFLOW_TRACKING_URL`  | MLflow tracking server URL                                                 | `http://localhost:5000` |
> ⚠️ **Note**: `WEAVIATE_API_KEY` is **required** for `structsense` to run. If it's not set, the system will raise an error.
>   For Weights & Biases you need to create a project and provide it's key.



```shell
# Example .env file

WEAVIATE_API_KEY=your_api_key
WEAVIATE_HTTP_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_HTTP_SECURE=false

WEAVIATE_GRPC_HOST=localhost
WEAVIATE_GRPC_PORT=50051
WEAVIATE_GRPC_SECURE=false

WEAVIATE_TIMEOUT_INIT=30
WEAVIATE_TIMEOUT_QUERY=60
WEAVIATE_TIMEOUT_INSERT=120

OLLAMA_API_ENDPOINT=http://host.docker.internal:11434
OLLAMA_MODEL=nomic-embed-text

ENABLE_WEAVE=true
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URL=http://localhost:5000
```
#### 🛠️ Minimum Required Environment Variables

Below are the **minimum required environment variables** to run `structsense`.
This configuration assumes all other optional variables will use their default values.

In this minimal setup:

- 🚫 **Weights & Biases** is disabled
  - 🚫 **MLflow tracking** is disabled
  - 🚫 **Knowledge source integration** is disabled
  - 📦 As a result, **no vector database** (e.g., Weaviate) is used

```shell
ENABLE_WEIGHTSANDBIAS=false
ENABLE_MLFLOW=false
ENABLE_KG_SOURCE=false
```
---
### 📄 Configuration Template

A blank configuration template is available in the [config_template](config_template) directory.

Please follow the guidelines below when updating the configuration:

---

#### ⚠️ Important Notes

- **Do not rename** predefined YAML keys such as `task_config` and `agent_config`.
  Only update the following:
  - Agent descriptions
  - Task descriptions
  - Embedding configurations
  - Knowledge configurations

- **Do not replace variables** enclosed in curly braces (`{}`); they are dynamically populated at runtime:
  - `{literature}`: Input text, e.g., extracted PDF content
  - `{extracted_structured_information}`: Output from the extractor agent
  - `{aligned_structured_information}`: Output from the alignment agent
  - `{judged_structured_information_with_human_feedback}`: Output from the judge agent
  - `{modification_context}` and `{user_feedback_text}`: User feedback used by the feedback agent

---

### 🧠 Agent Configuration

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
Running via CLI without an API key (not required for local models):
```shell
structsense-cli extract \
--source SOME.pdf \
--config config.yaml \
--env_file .env
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
### 👤 Human-in-the-Loop Configuration
Controls whether the human feedback loop is enabled.
```yaml
# Human-in-the-loop configuration
human_in_loop_config:
  humanfeedback_agent: true
```
### 🧬 Embedding Configuration
Defines the configuration for the embedding model used in the system.
```yaml
embedder_config:
  provider: ollama
  config:
    api_base: http://localhost:11434
    model: nomic-embed-text:latest
```

## Running 
### Using OpenRouter

```bash
structsense-cli extract \
  --source somefile.pdf \
  --api_key <YOUR_API_KEY> \
  --config someconfig.yaml \
  --env_file .env_file \ 
  --save_file result.json  # optional
```

### Using Ollama (Local)

```bash
structsense-cli extract \
  --source somefile.pdf \
  --config someconfig.yaml \
  --env_file .env_file \
  --save_file result.json  # optional
```

### Chunking
By default chunking is false you can enable it by passing `--chunking True`.

## Docker
The docker directory contains the individual and also merged docker compose file which installs Grobid, Ollama and Vector database.

### License
[Apache License Version 2.0](LICENSE.txt)
