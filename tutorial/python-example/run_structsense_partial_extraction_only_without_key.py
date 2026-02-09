# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------
 
# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : run_structsense_partial_extraction_only_without_key.py
# @Software: PyCharm

import yaml
from structsense.app import StructSenseFlow
import asyncio
import json
#load the configuration file for NER
with open("ner-config.yaml") as f:
    all_config = yaml.safe_load(f)

flow = StructSenseFlow(
    agent_config=all_config["agent_config"],
    task_config=all_config["task_config"],
    embedder_config=all_config.get("embedder_config", {}),
    input_source="test_small.pdf",
    enable_chunking=True,
    chunk_size=2000,
    max_workers=8,
    env_file=".env_example_partial",
)

result = asyncio.run(flow.extraction())

with open("result_extraction_without_key.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

