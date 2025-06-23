## 🧠 StructSense OHBM Demo

This guide provides instructions for running extraction tasks (NER and Resource Extraction) using either **OpenRouter** or **Ollama**, with support for Docker-based setup.

---
## Install StructSense
```shell
pip install git+https://github.com/sensein/structsense.git
```


## 🔧 Setup Instructions

1. **Navigate to the Docker setup directory:**

   ```bash
   cd Docker/merged
   ```

2. **Start the services (GROBID and Ollama) using Docker Compose:**

   ```bash
   docker compose up
   ```
   Test (open in browser):

   - Grobid: [http://localhost:8070/](http://localhost:8070/)
   - Ollama: [http://localhost:11434/](http://localhost:11434/)

   Once the containers are running, you’re ready to execute the examples.

   If there's an issue with Grobid, run it individually. The command below has been tested on MaC.
   
   ```shell
   docker pull lfoppiano/grobid:0.8.0
   docker run --init -p 8070:8070 -e JAVA_OPTS="-XX:+UseZGC" lfoppiano/grobid:0.8.0
   ```
   JAVA_OPTS="-XX:+UseZGC" helps to resolve the following error in MAC OS.

   Force platform emulation with --platform linux/amd64.

   ```shell
   docker run --platform linux/amd64 --init -p 8070:8070 -e JAVA_OPTS="-XX:+UseZGC" lfoppiano/grobid:0.8.0
   
   ```
   For more see [https://grobid.readthedocs.io/en/latest/Run-Grobid/](https://grobid.readthedocs.io/en/latest/Run-Grobid/).

   Official image: 

   ```shell
   docker pull grobid/grobid:0.8.2
   docker run grobid/grobid:0.8.2
   ```
---



---

## 🧪 Example Usage

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

---

## 📄 Environment File Example (`.env_file`)

Make sure your `.env_file` contains the following variables (for Ollama):

```env
ENABLE_KG_SOURCE=false
OLLAMA_API_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text:v1.5
```

> **Note:** If you change the embedding model in your Docker Compose setup, be sure to update both your `.env_file` and `config.yaml` accordingly.

---

## 🧪 Tasks

This repo supports two tasks:
- **Named Entity Recognition (NER)**
- **Resource Extraction**

### NER Example (OpenRouter)

```bash
structsense-cli extract \
  --source test_small.pdf \
  --api_key sk-or-v1-<your-key> \
  --config config_gpt.yaml \
  --env_file .env_ohbm_hackathon \
  --save_file result.json
```

### NER Example (Ollama)

```bash
structsense-cli extract \
  --source test_small.pdf \ 
  --config config_ollama.yaml \
  --env_file .env_ohbm_hackathon \
  --save_file result.json
```

### Resource Extraction Example (OpenRouter)

```bash
structsense-cli extract \
  --source paper_1909.11229v2.pdf \
  --api_key sk-or-v1-<your-key> \
  --config config_gpt.yaml \
  --env_file .env_ohbm_hackathon \
  --save_file result.json
```
### Resource Extraction Example (Ollama)

```bash
structsense-cli extract \
  --source paper_1909.11229v2.pdf \ 
  --config config_ollama.yaml \
  --env_file .env_ohbm_hackathon \
  --save_file result.json
```

> ⚠️ A version of the config file for **Ollama** is also included, but note that small local models may fail or produce suboptimal results. If you use **Ollama** version make sure to adjust the docker compose file so that it pulls the correct model that you've specified in config.yaml.

> ⚠️ For this demonstration we've disabled knowledge source as we do not have any ontologies in our vector databse.

> ⚠️ To enable chunking pass `--chunking True`. It still needs improvement and would be nice to have some help.