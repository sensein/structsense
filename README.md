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

## 📁Examples

-  [`Using openrouter/`](./example/ner_example) 
  - You need the openrouter API key
- [`Using Ollama`](./example/ner_example_ollama) 
  - Install ollama and pull the models which you intend to use. This example uses `deepseek-r1:14b` model. You can get it from ollama by running `ollama pull deepseek-r1:14b` command. If you want to use different models, e.g., `llama3.2:latest`, you need to pull it similar to `deepseek-r1:14b`. Make sure that ollama is running. You can run ollama using `ollama serve`.
---
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

### 📄 YAML Configuration
In order to run `structsense` you need 4 YAML configuration files.
- The first is the `agent configuration`.
  - The agent configuration. You can define as many agents as you want, we process it dynamically.
    - Example agent configuration.
      ```yaml 
        extractor_agent:
          role: >
            Neuroscience Named Entity Recognition (NER) Extractor Agent
          goal: >
            Perform Named Entity Recognition (NER) on neuroscience {literature} and return structured JSON output.
          backstory: >
            You are an AI assistant specialized in processing neuroscience and who do not hallucinate. 
            Your expertise includes recognizing and categorizing named entities such as anatomical regions, experimental conditions, and cell types. 
            Your responses strictly adhere to JSON format, ensuring accurate and structured data extraction for downstream applications.
          llm:
            model: openrouter/openai/gpt-4o-2024-11-20
            base_url: https://openrouter.ai/api/v1
            frequency_penalty: 0.1
            temperature: 0.7
            seed: 53
            api_key: sk-or-v1-
        
        alignment_agent:
          role: >
            Neuroscience Named Entity Recognition (NER) Concept Alignment Agent
          goal: >
            Perform concept alignment to the extracted Named Entity Recognition (NER) by extractor_agent {extracted_structured_information} and return structured JSON output.
          backstory: >
            You are an AI assistant specialized in processing neuroscience concept alignment with structured models, i.e., ontologies or schemas and who do not hallucinate. 
            Your expertise includes recognizing and categorizing extracted named entities such as anatomical regions, experimental conditions, and cell types and aligning the recognized named entities such as cell types with corresponding ontological terms. 
            Your responses strictly adhere to JSON format, ensuring accurate and structured data extraction for downstream applications.
          llm:
            model: openrouter/openai/gpt-4o-2024-11-20
            base_url: https://openrouter.ai/api/v1
            frequency_penalty: 0.1
            temperature: 0.7
            seed: 53
            api_key: sk-or-v1-
        
        judge_agent:
          role: >
            Neuroscience Named Entity Recognition (NER) Judge Agent
          goal: >
            Evaluate the {aligned_structured_information} based on predefined criteria and generate a structured JSON output reflecting the assessment results.
          backstory: >
            You are an AI assistant with expert knowledge in neuroscience and structured models, i.e., ontologies or schemas, and someone who does not hallucinate.  
            Your task is to evaluate the {aligned_structured_information} based on the accuracy and quality of the alignment. 
            Assign the score between 0-1 with 1 being the highest score of your evaluation.
            Your responses strictly adhere to JSON format, ensuring accurate and structured data extraction for downstream applications.
          llm:
            model: openrouter/openai/gpt-4o-2024-11-20
            base_url: https://openrouter.ai/api/v1
            frequency_penalty: 0.1
            temperature: 0.7
            seed: 53
            api_key: sk-or-v1-
        ```
    - In the YAML file: 
        - **ID**: Unique identifier
        - **Goal**: Task to be performed
        - **LLM config**: Model, base URL, temperature, etc.
        - **Backstory**: Background knowledge the agent leverages
        - **Output variable**: Result name for the next agent/task
      
    For further details, refer to [Role-Goal-Backstory](https://docs.crewai.com/guides/agents/crafting-effective-agents#core-principles-of-effective-agent-design)
- The second is the `task configuration`.
  - Task configuration allows you to describes the tasks for the agent. 
     - Example task configuration.
       ```yaml
        extraction_task:
          description: >
            From the given literature extract named entities from neuroscience statements.
            A named entity is anything that can be referred to with a proper name.
            Some common named entities in neuroscience articles are animal species (e.g., mouse, drosophila, zebrafish),
            anatomical regions (e.g., neocortex, mushroom body, cerebellum), experimental conditions (e.g., control, tetrodotoxin treatment, Scn1a knockout),
            and cell types (e.g., pyramidal neuron, direction-sensitive mechanoreceptor, oligodendrocyte)
        
            Literature:
            {literature}
          expected_output: >
            output format: json
            Example output:
            {
              "extracted_terms": {
                "1": [
                  {
                    "entity": "mouse",
                    "label": "ANIMAL_SPECIES",
                    "sentence": "These particles were visualized by fluorescent immunohistochemistry using mouse monoclonal anti-human myelin basic protein (MBPh) antibody (clone SMI-99).",
                    "start": 79,
                    "end": 84,
                    "paper_location": "methods",
                    "paper_title": "Concentration of myelin debris-like myelin basic protein-immunoreactive particles in the distal (anterior)-most part of the myelinated region in the normal rat optic nerve",
                    "doi": "10.1101/2025.03.19.643597"
                  }
                ]
              }
            }
          agent_id: extractor_agent
        
        alignment_task:
          description: >
            Take the output of extractor_agent {extracted_structured_information} as input and perform the concept alignment using the ontological concepts.
            A concept alignment is where you align the given entity to the matching concept or class from an ontology or schema.
          expected_output: >
            output format: json
            Example output:
            {
              "aligned_ner_terms": {
                "1": [
                  {
                    "entity": "oligodendrocyte",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000128",
                    "ontology_label": "Oligodendrocyte",
                    "sentence": "Individual oligodendrocytes provide...",
                    "start": 14,
                    "end": 29,
                    "paper_location": "discussion",
                    "paper_title": "Concentration of myelin debris-like...",
                    "doi": "10.1101/2025.03.19.643597"
                  }
                ] 
              }
            }
          agent_id: alignment_agent
        
        judge_task:
          description: >
            Take the output of alignment agent {aligned_structured_information} as input and perform the following evaluation:
            1. Assess the quality and accuracy of the alignment with the ontology or schema.
            2. Assign a score between 0 and 1 as a judge_score.
            3. Update the {aligned_structured_information} by adding the judge_score.
          expected_output: >
            output format: json
            Example output:
            {
              "judge_ner_terms": {
                "1": [
                  {
                    "entity": "oligodendrocyte",
                    "label": "CELL_TYPE",
                    "ontology_id": "CL:0000128",
                    "ontology_label": "Oligodendrocyte",
                    "judge_score": "0.8",
                    "sentence": "Individual oligodendrocytes provide...",
                    "start": 14,
                    "end": 29,
                    "paper_location": "discussion",
                    "paper_title": "Concentration of myelin debris-like...",
                    "doi": "10.1101/2025.03.19.643597"
                  }
                ]
              }
            }
          agent_id: judge_agent
       ```
     - Each task links to a specific agent via `agent_id` and defines:
       - **Description**: What the task does
       - **Input/Output schema**: Example output structure in JSON
       - **Agent Link**: Tied to an `id` from `agents configuration`
       - 
       > ⚠️ **Note**:  The variables {variable_name} are replaced at the run-time. Also, pay attention in the tasks where we are using the output variable defined in `agent configuration`.


- The third is the `embedding configuration` For more about the different embedding configurations using different provider see [https://docs.crewai.com/concepts/memory#additional-embedding-providerscl](https://docs.crewai.com/concepts/memory#additional-embedding-providerscl).
  ```yaml
  embedder_config:
    provider: ollama
    config:
      api_base: http://localhost:11434
      model: nomic-embed-text:latest
  ``` 
- The fourth and the final one is the `search configuration`, where we define the search keys. Since the ontology/schemas are our current knowledge source, which is why you see the label and entity as search key. _This is optional if you do not use knowledge source._
  ```yaml
  search_key: #local vector database
    - entity
    - label
  ```

## 📦 Installation
Install this package via :

```sh
pip install structsense
```

Or get the newest development version via:

```sh
pip install git+https://github.com/sensein/ner_framework.git
```

### 🧪 CLI Usage

You can run `StructSense` using the CLI tool `structsense-cli`. Below are a few examples showing different ways to provide input.

---

#### 📄 1. Extract from a PDF file (with knowledge source)

```bash
structsense-cli extract \
  --agentconfig config/your_agent.yaml \
  --taskconfig config/your_agent_task.yaml \
  --embedderconfig config/embedding.yaml \
  --flowconfig config/agent_processing_flow.yaml \
  --knowledgeconfig config/search_ontology_knowledge.yaml \
  --source someliterature.pdf
```
#### 💬 2. Extract from raw text (with knowledge source)

```shell
structsense-cli extract \
  --agentconfig config/your_agent.yaml \
  --taskconfig config/your_agent_task.yaml \
  --embedderconfig config/embedding.yaml \
  --flowconfig config/agent_processing_flow.yaml \
  --knowledgeconfig config/search_ontology_knowledge.yaml \
  --source "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function."
```

#### ⚡ 3. Extract from raw text (minimal setup without knowledge source)

```shell
structsense-cli extract \
  --agentconfig config/your_agent.yaml \
  --taskconfig config/your_agent_task.yaml \
  --embedderconfig config/embedding.yaml \
  --flowconfig config/flow_ner.yaml \
  --source "Additionally, mutations in the APOE gene have been linked to neurodegenerative disorders, impacting astrocytes and microglia function."
```

## In progress
- [X] [`More examples (e.g., using ollama)`](example/ner_example_ollama)
- [ ] Validations (e.g., benchmarking)
- [ ] Human feedback component.