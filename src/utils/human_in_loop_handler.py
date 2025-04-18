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
# @File    : human_in_loop_handler.py
# @Software: PyCharm

from typing import Dict, Optional, Tuple, Union, Any, Callable, List
import logging
import json
import tempfile
import subprocess
import os
import platform

class HumanInterventionRequired(Exception):
    """Exception raised when human intervention is required."""
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
class HumanInTheLoop:
    """
    Manages human-in-the-loop interactions within the flow.

    This class provides methods for requesting human feedback, approvals,
    and interventions at critical points in the processing flow. It supports
    CLI interactions.
    """

    def __init__(self,
                 enable_human_feedback: bool = False,
                 agent_feedback_config: Dict[str, bool] = None,
                 input_handler: Callable = None,
                 output_handler: Callable = None,
                 timeout_seconds: int = 300):
        """
        Initialize the human-in-the-loop component.

        Args:
            enable_human_feedback: Whether human feedback is enabled globally
            agent_feedback_config: Dictionary mapping agent names to feedback enabled status
            input_handler: Custom function to handle user input (defaults to input())
            output_handler: Custom function to handle output to the user (defaults to print)
            timeout_seconds: Timeout for human feedback in seconds
        """
        self.enable_human_feedback = enable_human_feedback
        self.agent_feedback_config = agent_feedback_config or {}
        self.input_handler = input_handler or input
        self.output_handler = output_handler or print
        self.timeout_seconds = timeout_seconds

        logger.info(f"Human-in-the-loop component initialized : {enable_human_feedback}")
        if agent_feedback_config:
            logger.info(f"Agent-specific feedback configuration: {agent_feedback_config}")

    def is_feedback_enabled_for_agent(self, agent_name: str) -> bool:
        """
        Check if the feedback is enabled for the agent. First the global feedback should be enabled
        and then also for the agent feedback

        Args:
            agent_name: Name of the agent to check

        Returns:
            Boolean indicating if feedback is enabled for this agent
        """
        # If global feedback is disabled, then all agents are disabled
        if not self.enable_human_feedback:
            return False

        # agent specific feedback configuration
        if agent_name in self.agent_feedback_config:
            return self.agent_feedback_config[agent_name]

        # Default to global setting if agent is not specifically configured
        return self.enable_human_feedback

    def request_feedback(self, data: Dict, step_name: str, agent_name: Optional[str] = None) -> Dict:
        """
        Request human feedback on data at a particular step.

        Args:
            data: The data to review, i.e., output of the agent
            step_name: The name of the current step
            agent_name: Optional name of the agent requesting feedback

        Returns:
            The original or modified data based on human feedback
        """
        # Check if feedback is enabled for this agent/step
        if agent_name and not self.is_feedback_enabled_for_agent(agent_name):
            logger.debug(f"Human feedback disabled for agent '{agent_name}', skipping feedback request for {step_name}")
            return data
        elif not self.enable_human_feedback:
            logger.debug(f"Human feedback globally disabled, skipping feedback request for {step_name}")
            return data

        # CLI mode - show options to request feedback from the user.
        try:
            agent_info = f" (Agent: {agent_name})" if agent_name else ""
            self.output_handler(f"\n{'=' * 80}\nHUMAN FEEDBACK REQUIRED FOR: {step_name}{agent_info}\n{'=' * 80}")

            # Format data for better readability
            if isinstance(data, dict) and len(str(data)) > 500:
                # For large dictionaries, show a summary
                keys_str = ", ".join(data.keys())
                self.output_handler(f"Data contains keys: {keys_str}")

                # Show sample of first items if it's a collection
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        sample = value[0] if len(value) > 0 else "empty"
                        self.output_handler(f"Sample of '{key}' (total: {len(value)} items): {sample}")
                    break
            else:
                self.output_handler(f"Current data: {data}")

            # options for user, i.e., feedback
            self.output_handler("\nOptions:")
            self.output_handler("1. Approve and continue")
            self.output_handler("2. View agent output")
            self.output_handler("3. Modify")

            choice = self.input_handler("Enter choice (1/2/3/4): ")

            if choice == "1":
                # user has approved the agent output without modification, so proceed
                logger.info(f"Human approved {step_name} data" + (f" for agent {agent_name}" if agent_name else ""))
                return data
            elif choice == "2":
                # View the agent output
                self.output_handler("\nView Agent Output:")
                # If data contains collections, show them more clearly
                if isinstance(data, dict):
                    for key, value in data.items():
                        self.output_handler(f"\n{key}:")
                        if isinstance(value, list):
                            for i, item in enumerate(value[:5]):
                                self.output_handler(f"  [{i}] {item}")
                            if len(value) > 5:
                                self.output_handler(f"  ... and {len(value) - 5} more items")
                        else:
                            self.output_handler(f"  {value}")
                else:
                    self.output_handler(data)

                # ask for feedback after reviewing
                return self.request_feedback(data, step_name, agent_name)
            elif choice == "3":
                self.output_handler("Please enter the JSON representation of the modified data:")
                self.output_handler("(For large data structures, consider editing specific parts only)")

                import json
                try:
                    # First offer to see the data as JSON for easier editing
                    json_str = json.dumps(data, indent=2)
                    self.output_handler(f"Current data as JSON:\n{json_str}")
                except:
                    self.output_handler("Could not convert current data to JSON for display")

                self.output_handler("\nEnter modified JSON data (or press Enter to open in editor):")
                modified_input = self.input_handler()

                if not modified_input:
                    # Try to use a temporary file and external editor if available
                    try:
                        editor = os.environ.get('EDITOR',
                                                'nano' if platform.system().lower() != 'windows' else 'notepad')

                        # Create temporary file
                        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tf:
                            json.dump(data, tf, indent=2)
                            temp_filename = tf.name

                        self.output_handler(f"Opening data in {editor}. Save and close when done.")

                        # Try to open the editor
                        try:
                            subprocess.run([editor, temp_filename], check=True)

                            # Read the modified content
                            with open(temp_filename, 'r') as tf:
                                modified_input = tf.read()

                            # Clean up
                            os.unlink(temp_filename)

                        except subprocess.CalledProcessError as e:
                            self.output_handler(f"Editor process failed: {e}")
                            self.output_handler("Please enter modified data directly:")
                            modified_input = self.input_handler()
                        except Exception as e:
                            self.output_handler(f"Failed to open editor: {e}")
                            self.output_handler("Please enter modified data directly:")
                            modified_input = self.input_handler()

                    except Exception as e:
                        self.output_handler(f"Editor setup failed: {e}")
                        self.output_handler("Please enter modified data directly:")
                        modified_input = self.input_handler()

                if modified_input:
                    try:
                        # Try to parse the input as JSON
                        modified_data = json.loads(modified_input)
                        logger.info(
                            f"Human modified {step_name} data" + (f" for agent {agent_name}" if agent_name else ""))
                        return modified_data
                    except json.JSONDecodeError as e:
                        self.output_handler(f"Error parsing JSON: {e}. Would you like to try again? (y/n)")
                        retry = self.input_handler().lower().startswith('y')
                        if retry:
                            return self.request_feedback(data, step_name, agent_name)
                        else:
                            self.output_handler("Using original data.")
                            return data
                else:
                    self.output_handler("No modifications made. Using original data.")
                    return data

        except Exception as e:
            if not isinstance(e, HumanInterventionRequired):
                logger.error(f"Error during human feedback: {e}")
                self.output_handler(f"Error during feedback: {e}. Using original data.")
            return data

    def request_approval(self, message: str, details: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """
        Request human approval with a yes/no question.

        Args:
            message: Message explaining what needs approval
            details: Optional details to show
            agent_name: Optional name of the agent requesting approval

        Returns:
            Boolean indicating if approved
        """
        # Check if feedback is enabled for this agent
        if agent_name and not self.is_feedback_enabled_for_agent(agent_name):
            logger.debug(f"Human feedback disabled for agent '{agent_name}', auto-approving: {message}")
            return True
        elif not self.enable_human_feedback:
            logger.debug(f"Human feedback globally disabled, auto-approving: {message}")
            return True

        # CLI mode
        try:
            agent_info = f" (Agent: {agent_name})" if agent_name else ""
            self.output_handler(f"\n{'=' * 80}\nAPPROVAL REQUIRED{agent_info}\n{'=' * 80}")
            self.output_handler(message)

            if details:
                self.output_handler(f"\nDetails:\n{details}")

            response = self.input_handler("\nApprove? (y/n): ").lower()
            approved = response.startswith('y')

            if approved:
                logger.info(f"Human approved: {message}" + (f" for agent {agent_name}" if agent_name else ""))
            else:
                logger.warning(f"Human rejected: {message}" + (f" for agent {agent_name}" if agent_name else ""))

            return approved

        except Exception as e:
            logger.error(f"Error during approval request: {e}")
            self.output_handler(f"Error during approval request: {e}. Auto-approving.")
            return True

    def provide_observation(self, message: str, data: Optional[Any] = None, agent_name: Optional[str] = None) -> None:
        """
        Provide an observation to the human without requiring feedback.

        Args:
            message: The observation message
            data: Optional data to display with the observation
            agent_name: Optional name of the agent providing the observation
        """
        # Check if feedback is enabled for this agent
        if agent_name and not self.is_feedback_enabled_for_agent(agent_name):
            logger.debug(f"Human feedback disabled for agent '{agent_name}', skipping observation: {message}")
            return
        elif not self.enable_human_feedback:
            logger.debug(f"Human feedback globally disabled, skipping observation: {message}")
            return

        # CLI mode
        try:
            agent_info = f" (Agent: {agent_name})" if agent_name else ""
            self.output_handler(f"\n{'=' * 50}\nOBSERVATION{agent_info}\n{'=' * 50}")
            self.output_handler(message)

            if data:
                self.output_handler("\nAdditional data:")
                self.output_handler(data)

        except Exception as e:
            logger.error(f"Error providing observation: {e}")

    def request_confirmation(self, message: str, agent_name: Optional[str] = None) -> bool:
        """
        Request a simple yes/no confirmation from the human.

        Args:
            message: The message to confirm
            agent_name: Optional name of the agent requesting confirmation

        Returns:
            Boolean indicating if confirmed
        """
        if agent_name and not self.is_feedback_enabled_for_agent(agent_name):
            logger.debug(f"Human feedback disabled for agent '{agent_name}', auto-confirming: {message}")
            return True
        elif not self.enable_human_feedback:
            logger.debug(f"Human feedback globally disabled, auto-confirming: {message}")
            return True

        # CLI mode
        try:
            agent_info = f" (Agent: {agent_name})" if agent_name else ""
            self.output_handler(f"\n{'=' * 80}\nCONFIRMATION REQUIRED{agent_info}\n{'=' * 80}")
            self.output_handler(message)

            while True:
                response = self.input_handler("\nConfirm? (y/n): ").lower()
                if response in ['y', 'n']:
                    return response == 'y'
                self.output_handler("Please enter 'y' or 'n'.")

        except Exception as e:
            logger.error(f"Error during confirmation request: {e}")
            return True


class ProgrammaticFeedbackHandler:
    pass
