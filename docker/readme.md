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

## Directory
- Individual
  - It consists individual docker compose file.
- Merged
  - It contains a single Docker Compose file that consolidates all configurations from the individual files into one unified setup.
  
## ⚠️ Requirements

Please ensure you have the **latest version of Docker and Docker Compose** installed. Older versions may result in compatibility errors related to the Compose file format.
