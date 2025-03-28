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

from crewai import Agent, Task


class DynamicAgentTask:
    def __init__(self, tasks_config):
        self.tasks_config = tasks_config

    def build_tasks(self, agents_by_id: dict) -> list:
        tasks = []
        for task_cfg in self.tasks_config:
            agent_id = task_cfg["agent_id"]
            agent = agents_by_id.get(agent_id)

            if agent is None:
                raise ValueError(
                    f"Agent with id '{agent_id}' not found for task '{task_cfg.get('id')}'"
                )

            # Remove fields not accepted by Task
            task_cfg_cleaned = {
                k: v for k, v in task_cfg.items() if k not in ["id", "agent_id"]
            }

            task = Task(config=task_cfg_cleaned, agent=agent)
            tasks.append(task)

        return tasks
