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
# @File    : humanloop.py
# @Software: PyCharm

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, Optional, Union


class HumanInterventionRequired(Exception):
    """Exception raised when human intervention is required."""

    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_feedback_input(feedback_input):
    """Parse feedback input to extract both natural language text and JSON if present.
    Returns a dict with 'user_feedback_text' and/or 'user_feedback_json'.
    """
    feedback_input = feedback_input.strip()
    # Try to find JSON block in the input
    json_match = re.search(r"({[\s\S]+})", feedback_input)
    if json_match:
        json_str = json_match.group(1)
        try:
            feedback_json = json.loads(json_str)
            text_part = feedback_input.replace(json_str, "").strip()
            result = {}
            if text_part:
                result["user_feedback_text"] = text_part
            result["user_feedback_json"] = feedback_json
            return result
        except Exception:
            pass  # If JSON parsing fails, treat as plain text
    # If not JSON, treat as text
    return {"user_feedback_text": feedback_input} if feedback_input else {}


class HumanInTheLoop:
    """Manages human-in-the-loop interactions within the flow.

    This class provides methods for requesting human feedback, approvals,
    and interventions at critical points in the processing flow. It supports
    CLI interactions.
    """

    def __init__(
        self,
        enable_human_feedback: bool = False,
        agent_feedback_config: Dict[str, bool] = None,
        input_handler: Callable = None,
        output_handler: Callable = None,
        timeout_seconds: int = 300,
    ):
        """Initialize the human-in-the-loop component.

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
        """Check if the feedback is enabled for the agent. First the global feedback should be enabled
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
        """Request human feedback on data at a particular step.

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
            self.output_handler("4. Abort")

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
                self.output_handler("Opening your default editor for feedback...")
                # Prepare template with instructions and current output JSON
                template = (
                    "# Enter your feedback below. You may write natural language above, and/or edit the JSON below.\n"
                    "# Lines starting with # will be ignored.\n"
                    "# Example:\n"
                    "# fix entity extraction as some are incorrect.\n"
                    "# {\n"
                    '#   "judged_structured_information": { ... }\n'
                    "# }\n\n"
                    "\n# --- Current Output JSON ---\n"
                    f"{json.dumps(data, indent=2)}\n"
                )
                feedback_input = self.open_editor_with_template(template)
                # Remove comment lines
                feedback_input = "\n".join(line for line in feedback_input.splitlines() if not line.strip().startswith("#"))
                parsed = parse_feedback_input(feedback_input)
                if not parsed:
                    self.output_handler("No feedback provided. Using original data.")
                    return data
                # If only JSON, use as modified data; if both, merge
                if "user_feedback_json" in parsed and not parsed.get("user_feedback_text"):
                    return "feedback"
                else:
                    # Attach text feedback to the data for the agent to interpret
                    result = data.copy() if isinstance(data, dict) else {}
                    if "user_feedback_json" in parsed:
                        result["user_feedback_json"] = parsed["user_feedback_json"]
                    if "user_feedback_text" in parsed:
                        result["user_feedback_text"] = parsed["user_feedback_text"]
                    return result

        except Exception as e:
            if not isinstance(e, HumanInterventionRequired):
                logger.error(f"Error during human feedback: {e}")
                self.output_handler(f"Error during feedback: {e}. Using original data.")
            return data

    def request_approval(self, message: str, details: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """Request human approval with a yes/no question.

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
            approved = response.startswith("y")

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
        """Provide an observation to the human without requiring feedback.

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
        """Request a simple yes/no confirmation from the human.

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
                if response in ["y", "n"]:
                    return response == "y"
                self.output_handler("Please enter 'y' or 'n'.")

        except Exception as e:
            logger.error(f"Error during confirmation request: {e}")
            return True

    def open_editor_with_template(self, template: str) -> str:
        editor = "nano"
        with tempfile.NamedTemporaryFile(suffix=".tmp", mode="w+", delete=False) as tf:
            tf.write(template)
            tf.flush()
            subprocess.call([editor, tf.name])
            tf.seek(0)
            content = tf.read()
        os.unlink(tf.name)
        return content


class ProgrammaticFeedbackHandler:
    """A custom handler for programmatic human feedback.
    This allows you to control the feedback process programmatically.
    """

    def __init__(self, feedback_responses=None):
        """Initialize the custom feedback handler.

        Args:
            feedback_responses: Dictionary mapping step names to feedback responses
                              Example: {
                                  "structured information extraction": {
                                      "choice": "1",  # Approve
                                      "modified_data": None
                                  },
                                  "information alignment": {
                                      "choice": "3",  # Modify
                                      "modified_data": {"terms": [...]}
                                  }
                              }
        """
        self.feedback_responses = feedback_responses or {}
        self.current_step = None
        self.pending_feedback = None
        self.feedback_result = None
        self.human_feedback_processed = False
        self.agent_feedback_config = {"extractor_agent": False, "alignment_agent": False, "judge_agent": False, "humanfeedback_agent": True}
        self.valid_choices = {"1", "2", "3", "4"}

    def is_feedback_enabled_for_agent(self, agent_name: str) -> bool:
        """Check if feedback is enabled for a specific agent.

        Args:
            agent_name: Name of the agent to check

        Returns:
            Boolean indicating if feedback is enabled for this agent
        """
        return self.agent_feedback_config.get(agent_name, False)

    def input(self, prompt: str) -> str:
        """Handle input requests."""
        # For programmatic feedback, we don't need user input
        return "3"  # Always return "3" for modify

    def output(self, message: str) -> None:
        """Handle output messages."""
        print(message)

    def set_current_step(self, step_name: str) -> None:
        """Set the current step for feedback handling."""
        self.current_step = step_name
        # Reset human feedback flag when moving to a new step
        if step_name != "human_feedback_processing":
            self.human_feedback_processed = False

    def has_pending_feedback(self) -> bool:
        """Check if there is any pending feedback to process.

        Returns:
            Boolean indicating if there is pending feedback
        """
        return self.pending_feedback is not None

    def validate_feedback_choice(self, choice: str) -> None:
        """Validate the feedback choice.

        Args:
            choice: The feedback choice to validate

        Raises:
            ValueError: If the choice is invalid
        """
        if choice not in self.valid_choices:
            raise ValueError(f"Invalid choice: {choice}. Must be one of: {', '.join(self.valid_choices)}")

    def request_feedback(self, data: Dict[str, Any], step_name: str, agent_name: Optional[str] = None) -> Union[Dict[str, Any], str]:
        """Request human feedback on data at a particular step.

        Args:
            data: The data to review
            step_name: The name of the current step
            agent_name: Optional name of the agent requesting feedback

        Returns:
            The original or modified data based on human feedback, or "feedback" to trigger the feedback loop
        """
        print(f"\n{'='*50}")
        print(f"Requesting feedback for step: {step_name}")
        print(f"Agent: {agent_name}")
        print(f"Data type: {type(data)}")
        print(f"Data: {data}")
        print(f"{'='*50}\n")

        self.current_step = step_name
        self.pending_feedback = {"data": data, "step_name": step_name, "agent_name": agent_name}

        # Return "feedback" to trigger the feedback loop
        return "feedback"

    def provide_feedback(self, choice: str, modified_data: Optional[str] = None) -> Dict[str, Any]:
        """Provide feedback for the pending request.

        Args:
            choice: The feedback choice ("1" for approve, "3" for modify, etc.)
            modified_data: Optional modified data if choice is "3"

        Returns:
            The processed data based on the feedback

        Raises:
            ValueError: If there is no pending feedback or if the choice is invalid
        """
        if not self.pending_feedback:
            raise ValueError("No pending feedback request")

        self.validate_feedback_choice(choice)

        data = self.pending_feedback["data"]
        step_name = self.pending_feedback["step_name"]
        agent_name = self.pending_feedback["agent_name"]

        if step_name == "human_feedback_processing" or agent_name == "humanfeedback_agent":
            self.human_feedback_processed = True

        if choice == "1":  # Approve
            print(f"Human approved {step_name} data" + (f" for agent {agent_name}" if agent_name else ""))
            self.clear_pending_feedback()
            return data
        elif choice == "3":  # Modify
            if modified_data is None:
                raise ValueError("Modified data required for modification choice")
            parsed = parse_feedback_input(modified_data)
            if not parsed:
                print("No feedback provided. Using original data.")
                self.clear_pending_feedback()
                return data
            if "user_feedback_json" in parsed and not parsed.get("user_feedback_text"):
                self.clear_pending_feedback()
                return "feedback"
            else:
                result = data.copy() if isinstance(data, dict) else {}
                if "user_feedback_json" in parsed:
                    result["user_feedback_json"] = parsed["user_feedback_json"]
                if "user_feedback_text" in parsed:
                    result["user_feedback_text"] = parsed["user_feedback_text"]
                self.clear_pending_feedback()
                return "feedback"
        else:
            print(f"Invalid choice: {choice}")
            self.clear_pending_feedback()
            return data

    def get_pending_feedback(self) -> Optional[Dict[str, Any]]:
        """Get the current pending feedback request.

        Returns:
            Dictionary containing the pending feedback request or None if no pending feedback
        """
        return self.pending_feedback

    def clear_pending_feedback(self) -> None:
        """Clear the pending feedback request."""
        self.pending_feedback = None

    def provide_observation(self, message: str, data: Optional[Any] = None, agent_name: Optional[str] = None) -> None:
        """Provide an observation to the human without requiring feedback.

        Args:
            message: The observation message
            data: Optional data to display with the observation
            agent_name: Optional name of the agent providing the observation
        """
        # For programmatic feedback, we just print the observation
        agent_info = f" (Agent: {agent_name})" if agent_name else ""
        print(f"\n{'=' * 50}\nOBSERVATION{agent_info}\n{'=' * 50}")
        print(message)

        if data:
            print("\nAdditional data:")
            print(data)

    def request_approval(self, message: str, details: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """Request human approval with a yes/no question.

        Args:
            message: Message explaining what needs approval
            details: Optional details to show
            agent_name: Optional name of the agent requesting approval

        Returns:
            Boolean indicating if approved
        """
        # For programmatic feedback, we just print the approval request
        agent_info = f" (Agent: {agent_name})" if agent_name else ""
        print(f"\n{'=' * 80}\nAPPROVAL REQUIRED{agent_info}\n{'=' * 80}")
        print(message)

        if details:
            print(f"\nDetails:\n{details}")

        # For programmatic feedback, we always approve
        print("Auto-approving request")
        return True
