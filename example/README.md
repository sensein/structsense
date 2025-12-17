# StructSense Examples

This directory contains example configurations and notebooks for using StructSense.

## Prerequisites

Before running these examples, you need to set up GROBID for PDF processing. You have multiple options:

### Option 1: Docker (Recommended for local development)
```bash
cd ../docker/individual/grobid-service
docker compose up -d
```

### Option 2: Use hosted GROBID service
Set the URL in your `.env` file:
```bash
GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=https://your-grobid-service.com
```

### Option 3: Docker run command (Quick start)
```bash
docker run --init -p 8070:8070 -e JAVA_OPTS="-XX:+UseZGC" lfoppiano/grobid:0.8.0
```

**Note:** Docker is now optional! See [docs/GROBID_SETUP.md](../docs/GROBID_SETUP.md) for all setup options including hosted services.

## Verify Setup

Test your GROBID connection:
```bash
python scripts/test_grobid_connection.py
```

Or check manually:
```bash
curl http://localhost:8070/api/version
```

## Available Examples

### NER_EXAMPLE_OPENROUTER
Named Entity Recognition example using OpenRouter API.

**Setup:**
1. Ensure GROBID is running (see prerequisites above)
2. Set your OpenRouter API key in `.env`
3. Run the notebook

### resource_extraction
Example for extracting structured metadata about scientific resources.

**Setup:**
1. Ensure GROBID is running (see prerequisites above)
2. Configure your LLM API keys in `.env`
3. Follow the example README for detailed usage

### pdf2_reproschema
Example for converting PDF documents to ReproSchema format.

**Setup:**
1. Ensure GROBID is running (see prerequisites above)
2. Configure your LLM API keys in `.env`
3. Follow the example README for detailed usage

## Configuration

All examples can be configured using environment variables. Copy `.env.example` to `.env` and configure:

```bash
# From repository root
cp .env.example .env
# Edit .env with your settings
```

Key configuration options:
- `GROBID_SERVER_URL_OR_EXTERNAL_SERVICE`: URL of GROBID service
- `EXTERNAL_PDF_EXTRACTION_SERVICE`: Set to True to use non-GROBID PDF service
- LLM API keys (OpenAI, Anthropic, etc.)

## Troubleshooting

### GROBID Connection Issues

If you get connection errors:
1. Check if GROBID is running: `docker ps | grep grobid`
2. Test the connection: `python scripts/test_grobid_connection.py`
3. See [docs/GROBID_SETUP.md](../docs/GROBID_SETUP.md) for detailed troubleshooting

### Memory Issues

If GROBID crashes or runs slowly:
1. Increase Docker memory limits (Docker Desktop settings)
2. Ensure at least 2-4GB RAM is available

## More Information

- [GROBID Setup Guide](../docs/GROBID_SETUP.md) - Comprehensive guide for all GROBID setup options
- [Docker Setup](../docker/readme.md) - Information about Docker services
- [Main Documentation](http://docs.brainkb.org/structsense_overview.html) - Full StructSense documentation
