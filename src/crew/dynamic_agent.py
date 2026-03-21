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
# @File    : dynamic_agent.py
# @Software: PyCharm

"""Build CrewAI agents from configuration.

This module provides :class:`DynamicAgent`, which creates a CrewAI :class:`Agent`
from role, goal, backstory, LLM, and embedder config. Used as an alternative
to initializing agents directly when config is loaded from YAML/JSON.

See Also
--------
- :mod:`crew.dynamic_agent_task` – Build tasks from config.
- :mod:`utils.crew_utils` – :func:`initialize_agent_and_task` used by the main pipeline.
"""

from crewai import LLM, Agent


class DynamicAgent:
    """Builds CrewAI agents from configuration dictionaries.

    Uses :class:`crewai.Agent` with :class:`crewai.LLM` and optional embedder
    and tools. Configuration is expected to provide role, goal, backstory, and llm.
    """

    def __init__(
        self,
        agents_config: list[dict],
        embedder_config: dict,
        tools: list = [],
        max_iter: int = 20,
    ):
        """Initialize the builder with agent and embedder config.

        Parameters
        ----------
        agents_config : list of dict
            List of agent configuration dictionaries. Each dict may contain
            ``role``, ``goal``, ``backstory``, ``llm`` (passed to :class:`crewai.LLM`).
        embedder_config : dict
            Embedding configuration; typically includes ``embedder_config``
            for the CrewAI embedder.
        tools : list, optional
            Optional list of CrewAI tools to attach to the agent. Default ``[]``.
        max_iter : int, optional
            Maximum number of reasoning iterations the agent may perform before
            it is forced to return its best answer.  CrewAI default is 20.
            Lower values (e.g. 3–5) reduce cost and latency for straightforward
            extraction tasks; higher values give the agent more attempts to
            correct itself on complex tasks.
        """
        self.agents_config = agents_config
        self.embedder_config = embedder_config
        self.tools = tools
        self.max_iter = max_iter

    def build_agent(self) -> Agent:
        """Build and return a single CrewAI agent from the stored config.

        Uses the first / current agent configuration (role, goal, backstory, llm)
        and the embedder/tools set at construction.

        Returns
        -------
        Agent
            A configured :class:`crewai.Agent` instance.
        """
        agent_config = self.agents_config
        agent_role = agent_config.get("role", "")
        agent_goal = agent_config.get("goal", "")
        agent_backstory = agent_config.get("backstory", "")
        llm_config = agent_config.get("llm", "")
        embedder_config = self.embedder_config.get("embedder_config")

        agent = Agent(
            role=agent_role,
            goal=agent_goal,
            backstory=agent_backstory,
            llm=LLM(**llm_config),
            embedder=embedder_config,
            tools=self.tools,
            allow_delegation=False,
            verbose=True,
            max_iter=self.max_iter,
        )

        return agent
