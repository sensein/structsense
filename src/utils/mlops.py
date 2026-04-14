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
# @File    : mlops.py
# @Software: PyCharm

"""
MLOps and monitoring utilities.

This module provides functions for setting up monitoring tools like
Weights & Biases and MLflow.
"""

import os
import logging

logger = logging.getLogger(__name__)


def setup_monitoring() -> None:
    """Set up monitoring tools if enabled.
    
    Checks environment variables and initializes:
    - Weights & Biases (if ENABLE_WEIGHTSANDBIAS=true)
    - MLflow (if ENABLE_MLFLOW=true)
    """
    if os.getenv("ENABLE_WEIGHTSANDBIAS", "false").lower() == "true":
        try:
            import weave
            weave.init(project_name="StructSense")
            logger.info("Weights & Biases monitoring enabled")
        except ImportError:
            logger.warning("Weights & Biases package not found, monitoring disabled")

    if os.getenv("ENABLE_MLFLOW", "false").lower() == "true":
        try:
            import mlflow
            mlflow.crewai.autolog()
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL", "http://localhost:5000"))
            mlflow.set_experiment("StructSense")
            logger.info("MLflow monitoring enabled")
        except ImportError:
            logger.warning("MLflow package not found, monitoring disabled")