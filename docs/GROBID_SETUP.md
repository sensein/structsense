# GROBID Setup Guide

This guide provides multiple options for setting up GROBID with StructSense. Choose the option that best fits your needs.

## Overview

StructSense uses GROBID for extracting structured content from PDF files. GROBID is a Java-based service that requires a server to run. The `grobidarticleextractor` Python package acts as a client to communicate with the GROBID server.

## Setup Options

### Option 1: Docker Compose (Recommended for Development)

This is the easiest way to get started with GROBID locally.

#### Steps:

1. Navigate to the GROBID docker directory:
   ```bash
   cd docker/individual/grobid-service
   ```

2. Start GROBID using Docker Compose:
   ```bash
   docker compose up -d
   ```

3. Verify GROBID is running:
   ```bash
   curl http://localhost:8070/api/version
   ```

4. Configure your environment (`.env` file):
   ```bash
   GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=http://localhost:8070
   EXTERNAL_PDF_EXTRACTION_SERVICE=False
   ```

5. Stop GROBID when done:
   ```bash
   docker compose down
   ```

**Pros:**
- Easy to set up and manage
- Consistent environment
- Easy to start/stop

**Cons:**
- Requires Docker installed
- Uses system resources when running

---

### Option 2: Using a Managed/Hosted GROBID Service

If you have access to a hosted GROBID instance (e.g., institutional server, cloud service), you can configure StructSense to use it directly.

#### Steps:

1. Configure your environment (`.env` file) with the hosted GROBID URL:
   ```bash
   GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=https://your-grobid-instance.example.com
   EXTERNAL_PDF_EXTRACTION_SERVICE=False
   ```

2. Verify the service is accessible:
   ```bash
   curl https://your-grobid-instance.example.com/api/version
   ```

**Pros:**
- No local Docker required
- No local resource usage
- Maintained by service provider
- Can be shared across team

**Cons:**
- Requires network connectivity
- May have usage limits or costs
- Dependent on external service availability

---

### Option 3: Manual GROBID Installation

You can run GROBID directly without Docker if needed.

#### Prerequisites:
- Java 11 or higher
- At least 2GB RAM

#### Steps:

1. Download GROBID:
   ```bash
   wget https://github.com/kermitt2/grobid/archive/0.8.0.zip
   unzip 0.8.0.zip
   cd grobid-0.8.0
   ```

2. Build GROBID:
   ```bash
   ./gradlew clean install
   ```

3. Start the GROBID service:
   ```bash
   ./gradlew run
   ```

4. Configure your environment (`.env` file):
   ```bash
   GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=http://localhost:8070
   EXTERNAL_PDF_EXTRACTION_SERVICE=False
   ```

**Pros:**
- No Docker required
- Full control over the installation

**Cons:**
- More complex setup
- Manual dependency management
- Requires Java installation

---

### Option 4: Using External PDF Extraction Services

If you have access to alternative PDF extraction APIs, you can configure StructSense to use them.

#### Steps:

1. Configure your environment (`.env` file):
   ```bash
   GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=https://your-pdf-api.example.com/extract
   EXTERNAL_PDF_EXTRACTION_SERVICE=True
   ```

**Note:** The external service must accept PDF files via POST request and return JSON with metadata and sections in the format expected by StructSense.

**Pros:**
- Flexibility to use different services
- No GROBID maintenance required

**Cons:**
- Requires compatible API
- May need custom integration

---

## Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GROBID_SERVER_URL_OR_EXTERNAL_SERVICE` | URL of GROBID server or external PDF extraction service | `http://localhost:8070` | `https://grobid.example.com` |
| `EXTERNAL_PDF_EXTRACTION_SERVICE` | Whether to use external service instead of GROBID | `False` | `True` or `False` |

---

## Troubleshooting

### GROBID Service Not Responding

**Problem:** Connection refused when trying to access GROBID.

**Solutions:**
1. Verify GROBID is running:
   ```bash
   docker ps | grep grobid
   ```
   
2. Check GROBID logs:
   ```bash
   docker logs <grobid-container-id>
   ```

3. Verify the port is not in use:
   ```bash
   lsof -i :8070
   ```

4. Try accessing GROBID directly:
   ```bash
   curl http://localhost:8070/api/version
   ```

### Memory Issues with GROBID

**Problem:** GROBID crashes or runs slowly.

**Solutions:**
1. Increase Docker memory limits (Docker Desktop settings)
2. Use the ZGC garbage collector (already configured in docker-compose.yaml):
   ```yaml
   environment:
     JAVA_OPTS: -XX:+UseZGC
   ```

### PDF Processing Fails

**Problem:** PDF extraction returns errors or empty results.

**Solutions:**
1. Verify PDF file is not corrupted
2. Check GROBID logs for specific errors
3. Try processing a simple test PDF
4. Ensure GROBID service has been warmed up (first requests may be slow)

### Network Connectivity Issues

**Problem:** Cannot connect to hosted GROBID service.

**Solutions:**
1. Check network connectivity
2. Verify URL is correct and accessible
3. Check firewall rules
4. Verify authentication if required

---

## Testing Your Setup

Use this Python script to test your GROBID configuration:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from GrobidArticleExtractor import GrobidArticleExtractor

# Load environment variables
load_dotenv()

# Get GROBID configuration
grobid_url = os.getenv("GROBID_SERVER_URL_OR_EXTERNAL_SERVICE", "http://localhost:8070")

# Test GROBID connection
try:
    extractor = GrobidArticleExtractor(grobid_url=grobid_url)
    print(f"✓ Successfully connected to GROBID at {grobid_url}")
    
    # Test with a sample PDF (provide your own test PDF)
    # pdf_path = Path("test.pdf")
    # if pdf_path.exists():
    #     xml_content = extractor.process_pdf(pdf_path)
    #     result = extractor.extract_content(xml_content)
    #     print(f"✓ Successfully processed PDF: {len(result.get('sections', []))} sections extracted")
    
except Exception as e:
    print(f"✗ Error connecting to GROBID: {e}")
```

---

## Performance Tips

1. **Warm up GROBID**: The first request is slower as models load. Consider making a test request on startup.
2. **Batch processing**: Process multiple PDFs in batches for better efficiency.
3. **Resource allocation**: Ensure adequate memory (2-4GB) for GROBID.
4. **Network**: Use local GROBID for best performance; hosted services add network latency.

---

## Security Considerations

1. **API Keys**: If using a hosted service, secure your API keys properly (use `.env` file, not hardcoded).
2. **Network**: Consider running GROBID behind a reverse proxy with authentication.
3. **Data Privacy**: Be aware that uploaded PDFs are processed by the GROBID service.
4. **Rate Limiting**: Hosted services may have rate limits; implement retry logic.

---

## Additional Resources

- [GROBID Documentation](https://grobid.readthedocs.io/)
- [GROBID GitHub Repository](https://github.com/kermitt2/grobid)
- [GrobidArticleExtractor Package](https://github.com/sensein/GrobidArticleExtractor)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
