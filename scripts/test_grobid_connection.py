#!/usr/bin/env python3
"""
GROBID Connection Test Script

This script helps you verify that your GROBID setup is working correctly.
It tests the connection to your GROBID service and provides helpful diagnostics.

Usage:
    python scripts/test_grobid_connection.py
    
    Or with custom URL:
    python scripts/test_grobid_connection.py --url http://your-grobid-server:8070
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_grobid_connection(grobid_url: str) -> bool:
    """Test connection to GROBID service.
    
    Args:
        grobid_url: URL of the GROBID service
        
    Returns:
        True if connection successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing GROBID Connection")
    print(f"{'='*70}")
    print(f"GROBID URL: {grobid_url}")
    
    # Test 1: Check if service is reachable
    print(f"\n[1/3] Checking if GROBID service is reachable...")
    try:
        version_url = f"{grobid_url.rstrip('/')}/api/version"
        response = requests.get(version_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✓ GROBID service is reachable")
            print(f"  Version endpoint: {version_url}")
            if response.content:
                try:
                    version_info = response.json()
                    print(f"  Response: {version_info}")
                except json.JSONDecodeError:
                    print(f"  Response: {response.text[:100]}")
            else:
                print(f"  Response: {response.text}")
        else:
            print(f"✗ GROBID service returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to GROBID service at {grobid_url}")
        print(f"\nPossible solutions:")
        print(f"  1. Start GROBID with Docker:")
        print(f"     cd docker/individual/grobid-service && docker compose up -d")
        print(f"  2. Check if GROBID is running:")
        print(f"     docker ps | grep grobid")
        print(f"  3. Verify the URL is correct")
        print(f"  4. See docs/GROBID_SETUP.md for setup instructions")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Connection to GROBID service timed out")
        print(f"  The service might be starting up. Wait a moment and try again.")
        return False
    except Exception as e:
        print(f"✗ Error connecting to GROBID: {str(e)}")
        return False
    
    # Test 2: Check processHeaderDocument endpoint
    print(f"\n[2/3] Testing GROBID processHeaderDocument endpoint...")
    try:
        header_url = f"{grobid_url.rstrip('/')}/api/processHeaderDocument"
        # Send a minimal test request
        response = requests.post(header_url, timeout=5)
        
        # We expect 200 (with content) or 400 (bad request without file)
        # Both indicate the endpoint is accessible
        if response.status_code in [200, 400]:
            print(f"✓ processHeaderDocument endpoint is accessible")
        elif response.status_code == 500:
            print(f"⚠ processHeaderDocument endpoint returned 500 (Internal Server Error)")
            print(f"  The service is reachable but may have configuration issues")
            print(f"  Check GROBID logs for details")
            # Continue - service is reachable even if not fully functional
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing endpoint: {str(e)}")
        return False
    
    # Test 3: Check if GrobidArticleExtractor can initialize
    print(f"\n[3/3] Testing GrobidArticleExtractor initialization...")
    try:
        from GrobidArticleExtractor import GrobidArticleExtractor
        
        extractor = GrobidArticleExtractor(grobid_url=grobid_url)
        print(f"✓ GrobidArticleExtractor initialized successfully")
        print(f"  Using GROBID at: {extractor.grobid_url}")
        
    except ImportError:
        print(f"✗ GrobidArticleExtractor package not found")
        print(f"  Install with: pip install grobidarticleextractor")
        return False
    except Exception as e:
        print(f"✗ Error initializing GrobidArticleExtractor: {str(e)}")
        return False
    
    # All tests passed
    print(f"\n{'='*70}")
    print(f"✓ All tests passed! GROBID is configured correctly.")
    print(f"{'='*70}")
    return True


def check_environment():
    """Check and display environment configuration."""
    print(f"\n{'='*70}")
    print(f"Environment Configuration")
    print(f"{'='*70}")
    
    grobid_url = os.getenv("GROBID_SERVER_URL_OR_EXTERNAL_SERVICE", "http://localhost:8070")
    external_service = os.getenv("EXTERNAL_PDF_EXTRACTION_SERVICE", "False")
    
    print(f"GROBID_SERVER_URL_OR_EXTERNAL_SERVICE: {grobid_url}")
    print(f"EXTERNAL_PDF_EXTRACTION_SERVICE: {external_service}")
    
    env_file = Path(".env")
    if env_file.exists():
        print(f"\n✓ .env file found at: {env_file.absolute()}")
    else:
        print(f"\n⚠ .env file not found")
        print(f"  Consider copying .env.example to .env and configuring it")
    
    return grobid_url


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test GROBID connection and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with environment variable configuration
  python scripts/test_grobid_connection.py
  
  # Test with custom URL
  python scripts/test_grobid_connection.py --url http://grobid.example.com:8070
  
For more information, see docs/GROBID_SETUP.md
        """
    )
    parser.add_argument(
        "--url",
        help="GROBID service URL (overrides environment variable)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Get GROBID URL
    if args.url:
        grobid_url = args.url
    else:
        grobid_url = check_environment()
    
    # Run tests
    success = test_grobid_connection(grobid_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
