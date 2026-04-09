import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


ENV_PATH = Path(__file__).parent / "configs/.env_example"

skip_if_no_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytest.fixture(scope="module")
def load_env():
    logging.info("Loading environment variables from: %s", ENV_PATH)
    load_dotenv(ENV_PATH, override=True)
