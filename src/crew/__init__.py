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

"""CrewAI dynamic agent and task builders.

This package provides helpers to build CrewAI :class:`crewai.Agent` and
:class:`crewai.Task` instances from configuration (e.g. YAML/JSON).

Modules
-------
- :mod:`crew.dynamic_agent` – :class:`DynamicAgent` builds agents from config.
- :mod:`crew.dynamic_agent_task` – :class:`DynamicAgentTask` builds tasks from config.

See Also
--------
- :mod:`structsense.app` – Uses :mod:`utils.crew_utils` for agent/task initialization;
  this package is an alternative builder layer.
"""

# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : __init__.py.py
# @Software: PyCharm
