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
# @File    : dynamic_agent_task.py
# @Software: PyCharm

"""Build CrewAI tasks from configuration.

This module provides :class:`DynamicAgentTask`, which creates a CrewAI
:class:`Task` from a task config dict and assigns it to an agent. Supports
optional Pydantic output schema.

See Also
--------
- :mod:`crew.dynamic_agent` – Build agents from config.
- :mod:`utils.crew_utils` – :func:`initialize_agent_and_task` used by the main pipeline.
"""

from crewai import Agent, Task
from typing import Any, Optional


class DynamicAgentTask:
    """Builds CrewAI tasks from a task configuration dictionary.

    Wraps :class:`crewai.Task` with config (description, etc.) and optional
    :class:`pydantic.BaseModel` for structured output.
    """

    def __init__(self, tasks_config):
        """Initialize the builder with task configuration.

        Parameters
        ----------
        tasks_config : dict
            Task configuration passed to :class:`crewai.Task` (e.g. description,
            expected_output, context). Keys depend on CrewAI's Task API.
        """
        self.tasks_config = tasks_config

    def build_task(self, agent: Agent, pydantic_output: Optional[Any] = None) -> Task:
        """Create and return a CrewAI task assigned to the given agent.

        Parameters
        ----------
        agent : Agent
            The CrewAI agent that will execute the task.
        pydantic_output : type or None, optional
            Optional Pydantic model class for structured task output.
            If provided, passed as ``output_pydantic`` to :class:`crewai.Task`.

        Returns
        -------
        Task
            A configured :class:`crewai.Task` instance.
        """
        task_kwargs = {
            "config": self.tasks_config,
            "agent": agent,
        }
        if pydantic_output:
            task_kwargs["output_pydantic"] = pydantic_output

        return Task(**task_kwargs, async_execution=True)
