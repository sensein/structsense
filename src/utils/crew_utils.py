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
# @File    : crew_utils.py
# @Software: PyCharm

import json
from typing import List, Dict, Any, Optional
from crewai import  Crew




def _run_crew_on_retry(
    crew: Crew,
    text: str,
    input_key: str = "input_text",
    default_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    SAFE + RETRY-ONCE Crew runner (generic).

    Args:
        crew: The Crew instance to run
        text: The input text to process
        input_key: The key to use in the inputs dict (default: "input_text")
        default_result: Default result dict if parsing fails (default: empty dict)

    Returns:
        Dict with parsed results. If both attempts fail, returns:
          {**default_result, "error": "..."}
    """
    if default_result is None:
        default_result = {}

    def attempt() -> Dict[str, Any]:
        try:
            res = crew.kickoff(inputs={input_key: text})

            # Different possible shapes:
            if isinstance(res, str):
                return json.loads(res)

            raw = getattr(res, "raw", res)
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    print("[WARN] Crew returned non-JSON string; returning default.")
                    return default_result.copy()

            if isinstance(raw, dict):
                return raw

            return default_result.copy()
        except Exception as e:
            result = default_result.copy()
            result["error"] = str(e)
            return result

    # First try
    r1 = attempt()
    if "error" not in r1:
        return r1

    print(f"[WARN] First attempt failed: {r1['error']}. Retrying once...")

    # Second try
    r2 = attempt()
    if "error" not in r2:
        print("[OK] Retry succeeded.")
        return r2

    print(f"[WARN] Retry failed: {r2['error']}. Returning default result with error.")
    result = default_result.copy()
    result["error"] = r2["error"]
    return result