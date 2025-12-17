# 🧩 StructSense

Welcome to `structsense`!

`structsense` is a powerful multi-agent system designed to extract structured information from unstructured data. By orchestrating intelligent agents, it helps you make sense of complex information — hence the name *structsense*.

Whether you're working with scientific texts, documents, or messy data, `structsense` enables you to transform it into meaningful, structured insights.

## 📋 Quick Start

### Prerequisites

For PDF processing, StructSense requires a GROBID service. You have multiple options:

1. **Docker (Recommended)**: Run GROBID locally using Docker Compose
2. **Hosted Service**: Use a managed GROBID instance
3. **Manual Installation**: Install GROBID directly

See the [GROBID Setup Guide](docs/GROBID_SETUP.md) for detailed instructions on all setup options.

### Installation

```bash
pip install structsense
```

### Basic Usage

```bash
# Set up your environment variables (see GROBID Setup Guide)
export GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=http://localhost:8070
export EXTERNAL_PDF_EXTRACTION_SERVICE=False

# Run StructSense
structsense-cli extract --source document.pdf --config config.yaml
```

## 📚 Documentation

- **Complete Documentation**: [docs.brainkb.org](http://docs.brainkb.org/structsense_overview.html)
- **GROBID Setup Guide**: [docs/GROBID_SETUP.md](docs/GROBID_SETUP.md)
- **Docker Setup**: [docker/readme.md](docker/readme.md)

## 🔑 Key Features

- **Multi-Agent System**: Orchestrates intelligent agents for structured extraction
- **Flexible PDF Processing**: Supports multiple GROBID deployment options
- **Scientific Text Support**: Optimized for scientific papers and technical documents
- **Ontology Integration**: Aligns extracted terms with standardized ontologies
- **Human-in-the-Loop**: Optional feedback integration for improved accuracy

## ⚙️ Configuration

StructSense uses environment variables for configuration. Key variables:

- `GROBID_SERVER_URL_OR_EXTERNAL_SERVICE`: URL of GROBID server (default: `http://localhost:8070`)
- `EXTERNAL_PDF_EXTRACTION_SERVICE`: Use external PDF service instead of GROBID (default: `False`)

See the [GROBID Setup Guide](docs/GROBID_SETUP.md) for complete configuration options.

## 📄 License
[Apache License Version 2.0](LICENSE.txt)
