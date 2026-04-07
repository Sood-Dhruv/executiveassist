"""
ExecutiveAssist-Env: An OpenEnv environment simulating an AI Executive Assistant.
The agent handles scheduling, inbox triage, meeting coordination, and task management.
"""

import json
import random
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from tasks import TASKS
from graders import grade_action


class ExecutiveAssistEnv:
    """
    OpenEnv-compliant environment for AI Executive Assistant tasks.

    The agent acts as a personal executive assistant (like a smart secretary),
    handling real-world tasks such as:
      - Scheduling meetings and finding free slots
      - Triaging and prioritizing emails
      - Extracting action items from meeting notes
      - Multi-step calendar coordination across multiple people

    Step API:
        reset(task_id)  → initial state dict
        step(action)    → (state, reward, done, info)
        state()         → current state dict
    """

    # Task difficulty tiers
    EASY_TASKS   = ["schedule_meeting", "confirm_slot", "cancel_meeting"]
    MEDIUM_TASKS = ["inbox_triage", "reschedule_conflict", "draft_reply"]
    HARD_TASKS   = ["multi_party_schedule", "meeting_notes_extraction", "full_day_plan"]

    ALL_TASKS = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS

    def __init__(self, task_id: Optional[str] = None, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self._task_id = task_id
        self._current_task: Optional[Dict] = None
        self._state: Dict[str, Any] = {}
        self._step_count = 0
        self._max_steps = 10
        self._done = False
        self._history: List[Dict] = []
        self._total_reward = 0.0

    # ──────────────────────────────────────────────
    # OpenEnv Standard Interface
    # ──────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment to a fresh task. Returns initial state."""
        if seed is not None:
            self.seed = seed
            random.seed(seed)

        # Pick task
        chosen_id = task_id or self._task_id or random.choice(self.ALL_TASKS)
        if chosen_id not in TASKS:
            raise ValueError(f"Unknown task_id: '{chosen_id}'. Valid: {list(TASKS.keys())}")

        self._task_id = chosen_id
        self._current_task = deepcopy(TASKS[chosen_id])
        self._step_count = 0
        self._done = False
        self._history = []
        self._total_reward = 0.0

        # Build initial state
        self._state = {
            "task_id": self._task_id,
            "task_description": self._current_task["description"],
            "difficulty": self._current_task["difficulty"],
            "context": deepcopy(self._current_task["context"]),
            "instructions": self._current_task["instructions"],
            "expected_actions": self._current_task.get("expected_actions", []),
            "step": 0,
            "done": False,
            "reward": 0.0,
            "total_reward": 0.0,
            "feedback": "Task started. Awaiting your first action.",
            "history": [],
        }
        return deepcopy(self._state)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one action in the environment.

        Args:
            action: dict with at minimum {"type": str, ...task-specific fields}

        Returns:
            (state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._current_task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        self._step_count += 1

        # Grade the action
        reward, feedback, done_signal = grade_action(
            task_id=self._task_id,
            task=self._current_task,
            action=action,
            step=self._step_count,
            history=self._history,
        )

        self._total_reward += reward
        self._history.append({
            "step": self._step_count,
            "action": action,
            "reward": reward,
            "feedback": feedback,
        })

        # Check termination
        if done_signal or self._step_count >= self._max_steps:
            self._done = True

        # Update state
        self._state.update({
            "step": self._step_count,
            "done": self._done,
            "reward": reward,
            "total_reward": self._total_reward,
            "feedback": feedback,
            "history": deepcopy(self._history),
            "context": deepcopy(self._current_task["context"]),
        })

        info = {
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "task_id": self._task_id,
            "difficulty": self._current_task["difficulty"],
        }

        return deepcopy(self._state), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the current state without advancing the environment."""
        return deepcopy(self._state)

    # ──────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────

    @property
    def task_ids(self) -> List[str]:
        return self.ALL_TASKS

    @property
    def action_space(self) -> Dict[str, Any]:
        """Describe valid action types."""
        return {
            "type": "dict",
            "required_key": "type",
            "action_types": [
                "schedule",        # Book a meeting slot
                "cancel",          # Cancel a meeting
                "reschedule",      # Move a meeting
                "reply",           # Draft/send a reply
                "triage",          # Prioritize inbox items
                "extract",         # Pull action items from notes
                "plan",            # Full daily plan
                "delegate",        # Assign task to someone
            ],
        }

    def render(self) -> str:
        """Return a human-readable string of current state."""
        if not self._current_task:
            return "No task loaded."
        lines = [
            f"=== ExecutiveAssist-Env ===",
            f"Task:       {self._task_id} ({self._current_task['difficulty']})",
            f"Step:       {self._step_count}/{self._max_steps}",
            f"Reward:     {self._total_reward:.2f}",
            f"Done:       {self._done}",
            f"",
            f"Description: {self._current_task['description']}",
            f"",
            f"Feedback: {self._state.get('feedback', '')}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"ExecutiveAssistEnv(task_id={self._task_id!r}, step={self._step_count})"
