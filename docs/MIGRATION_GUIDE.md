# Migration Guide: Docker-based GROBID to Flexible Setup

This guide helps existing users migrate from the Docker-only GROBID setup to the new flexible configuration system.

## What Changed?

Previously, StructSense required users to run GROBID via Docker. Now, you have multiple options:

1. **Docker (Local)** - Run GROBID in a Docker container (backward compatible)
2. **Hosted Service** - Use a managed GROBID instance  
3. **Manual Installation** - Install GROBID directly without Docker
4. **External Service** - Use alternative PDF extraction APIs

## For Existing Users

### If You're Already Using Docker

**Good news:** Your setup continues to work without any changes!

The existing Docker setup remains fully supported. You can continue using:

```bash
cd docker/individual/grobid-service
docker compose up -d
```

### If You Want to Switch to Hosted GROBID

1. Get access to a hosted GROBID service (institutional or cloud-hosted)

2. Create or update your `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Configure the GROBID URL:
   ```bash
   GROBID_SERVER_URL_OR_EXTERNAL_SERVICE=https://your-grobid-service.com
   EXTERNAL_PDF_EXTRACTION_SERVICE=False
   ```

4. Stop your local Docker GROBID (optional):
   ```bash
   cd docker/individual/grobid-service
   docker compose down
   ```

5. Test the connection:
   ```bash
   python scripts/test_grobid_connection.py
   ```

### If You Want to Remove Docker Dependency

1. Choose an alternative setup from the [GROBID Setup Guide](GROBID_SETUP.md)

2. Configure your `.env` file accordingly

3. Verify the setup works:
   ```bash
   python scripts/test_grobid_connection.py
   ```

## New Features

### Environment Configuration

The new `.env.example` file provides a template for all configuration options:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Connection Test Script

Verify your GROBID setup is working:

```bash
python scripts/test_grobid_connection.py
```

### Improved Error Messages

The code now provides helpful error messages when GROBID is not available, with suggestions on how to fix common issues.

### Comprehensive Documentation

- [GROBID Setup Guide](GROBID_SETUP.md) - All setup options
- [Docker Setup](../docker/readme.md) - Docker-specific instructions
- [Example README](../example/README.md) - Example-specific setup

## Backward Compatibility

All changes are fully backward compatible:

- ✅ Existing Docker setups continue to work
- ✅ No changes required to existing code
- ✅ Environment variables use the same names
- ✅ Default values remain unchanged

## Benefits of the New Approach

1. **Flexibility** - Choose the setup that works best for your environment
2. **No Docker Required** - Use hosted services without local Docker
3. **Better Documentation** - Comprehensive guides for all scenarios
4. **Improved Errors** - Helpful messages when things go wrong
5. **Easy Testing** - Built-in connection test script

## Troubleshooting

### "Cannot connect to GROBID service"

1. Check if GROBID is running:
   ```bash
   docker ps | grep grobid
   ```

2. Test the connection:
   ```bash
   python scripts/test_grobid_connection.py
   ```

3. Verify your `.env` configuration

4. See [GROBID Setup Guide](GROBID_SETUP.md) for detailed troubleshooting

### "Module 'dotenv' not found"

Install required dependencies:
```bash
pip install python-dotenv requests grobidarticleextractor
```

Or install the full package:
```bash
pip install structsense
```

## Need Help?

- 📖 [GROBID Setup Guide](GROBID_SETUP.md)
- 📖 [Main Documentation](http://docs.brainkb.org/structsense_overview.html)
- 🐛 [Report Issues](https://github.com/sensein/structsense/issues)
- 💬 [Discussions](https://github.com/sensein/structsense/discussions)
