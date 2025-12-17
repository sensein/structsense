# 🐳 Docker Setup

This repository provides the Docker Compose configurations needed to run the **StructSense** system and its associated services seamlessly. Use these files to quickly spin up the full environment for development or deployment.

## 🚀 Getting Started with Docker Compose

To start the services using Docker Compose (V2):

```bash
docker compose up
```

> ℹ️ If you're using Docker Compose V1, the command is:
> ```bash
> docker-compose up
> ```

You can also specify a particular Compose file with the `-f` flag:

```bash
docker compose -f custom-compose.yml up
```

## 📁 Directory Structure

- **Individual**: Contains individual Docker Compose files for each service
  - `grobid-service/`: GROBID PDF extraction service (optional)
  - `ollama/`: Ollama LLM service
  - `weaviate-vector-database/`: Weaviate vector database
- **Merged**: Contains a single Docker Compose file that consolidates all configurations from the individual files into one unified setup

## 🔧 Service Components

### Core Services (Root `docker-compose.yaml`)
The root `docker-compose.yaml` includes only the essential services:
- **Weaviate**: Vector database for ontology storage

### Optional Services

#### GROBID Service (Optional)
GROBID is used for PDF extraction but is **optional**. You have several alternatives:

1. **Run GROBID via Docker** (Recommended for local development):
   ```bash
   cd docker/individual/grobid-service
   docker compose up -d
   ```

2. **Use a hosted GROBID service**: Configure the URL in your `.env` file
3. **Use an external PDF extraction service**: Set `EXTERNAL_PDF_EXTRACTION_SERVICE=True`

See the [GROBID Setup Guide](../docs/GROBID_SETUP.md) for detailed instructions on all options.

#### Other Services
- **Ollama**: For running local LLM models
- **Complete Stack**: Use `docker/merged/docker-compose.yaml` to run all services together

## 🎯 Usage Examples

### Start Only Core Services
```bash
# From repository root
docker compose up -d
```

### Start GROBID Service (Optional)
```bash
cd docker/individual/grobid-service
docker compose up -d
```

### Start All Services (Including GROBID)
```bash
cd docker/merged
docker compose up -d
```

### Stop Services
```bash
docker compose down
```

## ⚠️ Requirements

Please ensure you have the **latest version of Docker and Docker Compose** installed. Older versions may result in compatibility errors related to the Compose file format.

- Docker Engine 20.10+
- Docker Compose V2 (recommended)

## 💡 Tips

- GROBID is **not required** if you're using hosted services or external PDF APIs
- Start only the services you need to save resources
- Use the merged configuration for a complete development environment
- Individual service configurations allow for more flexible deployment
