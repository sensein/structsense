# StructSense Scripts

This directory contains utility scripts to help with setup, testing, and maintenance of StructSense.

## Available Scripts

### test_grobid_connection.py

Tests the connection to your GROBID service and verifies it's configured correctly.

**Usage:**
```bash
# Test with environment variable configuration
python scripts/test_grobid_connection.py

# Test with custom URL
python scripts/test_grobid_connection.py --url http://grobid.example.com:8070
```

**What it tests:**
1. GROBID service is reachable
2. GROBID API endpoints are accessible
3. GrobidArticleExtractor can initialize properly

**Prerequisites:**
- `grobidarticleextractor` package installed
- `python-dotenv` package installed
- GROBID service running (or accessible URL)

## More Information

- [GROBID Setup Guide](../docs/GROBID_SETUP.md)
- [Docker Setup](../docker/readme.md)
- [Main Documentation](http://docs.brainkb.org/structsense_overview.html)
