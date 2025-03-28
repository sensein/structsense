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

from crewai import LLM, Agent


class DynamicAgent:
    def __init__(
        self, agents_config: list[dict], embedder_config: dict, tools: list = []
    ):
        """Initializes the DynamicAgent class for multiple agents.

        Args:
            agents_config (list): List of agent configuration dictionaries.
            embedder_config (dict): Embedding configuration.
            tools (list): Optional list of tools shared across agents.
        """
        self.agents_config = agents_config
        self.embedder_config = embedder_config
        self.tools = tools

    def build_agents(self) -> dict:
        """Builds and returns agents mapped by ID.

        Returns:
            dict: Mapping of agent_id to Agent instance.
        """
        agents_by_id = {}
        for config in self.agents_config:
            agent_id = config["id"]
            llm_config = config.get("llm", {})
            agent = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config.get("backstory", ""),
                llm=LLM(**llm_config),
                embedder=self.embedder_config.get("embedder_config"),
                tools=self.tools,
                allow_delegation=False,
                verbose=True,
            )
            agents_by_id[agent_id] = agent
        return agents_by_id
