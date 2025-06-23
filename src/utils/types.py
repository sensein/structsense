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
# @File    : types.py
# @Software: PyCharm

from typing import Any, Dict, List

from pydantic import BaseModel


class ExtractedTermsDynamic(BaseModel):
    extracted_structured_information: Any


class AlignedTermsDynamic(BaseModel):
    aligned_structured_information: Any


class JudgedTermsDynamic(BaseModel):
    judged_structured_information: Any
