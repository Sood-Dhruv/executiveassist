"""
inference.py — Baseline AI agent for ExecutiveAssist-Env.

Runs a GPT-4o agent through all 9 tasks and prints results.

Usage:
    export OPENAI_API_KEY=sk-...
    export IMAGE_NAME=executiveassist-env   # if using Docker
    python inference.py

    # Or target a live HF Space:
    python inference.py --base-url https://your-space.hf.space
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:7860"
MODEL = "gpt-4o"
MAX_STEPS = 8

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AI Executive Assistant.
    You will be given a task and the current state of an executive's environment
    (calendar, inbox, meeting notes, etc.).

    Your job is to respond with ONE JSON action that advances the task.
    
    ALWAYS respond with a single valid JSON object.
    DO NOT include markdown code fences, just raw JSON.
    
    Available action types:
      - schedule:   {"type": "schedule", "date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM", "attendees": [...], "title": "..."}
      - cancel:     {"type": "cancel", "event_id": "...", "reason": "..."}
      - reschedule: {"type": "reschedule", "event_id": "...", "new_date": "YYYY-MM-DD", "new_start_time": "HH:MM", "new_end_time": "HH:MM"}
      - reply:      {"type": "reply", "to": ["email@..."], "subject": "...", "body": "..."}
      - triage:     {"type": "triage", "assignments": {"email-id": "URGENT|IMPORTANT|DELEGATE|ARCHIVE", ...}}
      - extract:    {"type": "extract", "action_items": [{"task": "...", "owner": "...", "due_date": "YYYY-MM-DD"}], "decisions": [...], "open_questions": [...]}
      - plan:       {"type": "plan", "schedule": [{"start_time": "HH:MM", "end_time": "HH:MM", "type": "meeting|task|email|travel|break", "title": "..."}]}
      - delegate:   {"type": "delegate", "task": "...", "to": "...", "deadline": "YYYY-MM-DD"}
    
    Think step by step. Analyze the context carefully before acting.
""")


# ──────────────────────────────────────────────────────────
# Environment Client
# ──────────────────────────────────────────────────────────

class EnvClient:
    """HTTP client for the ExecutiveAssist-Env server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def reset(self, task_id: Optional[str] = None, seed: int = 42) -> Dict:
        payload = {"seed": seed}
        if task_id:
            payload["task_id"] = task_id
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict) -> Dict:
        r = self.client.post(f"{self.base_url}/step", json={"action": action})
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict:
        r = self.client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> Dict:
        r = self.client.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()


# ──────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────

class ExecutiveAssistAgent:
    """Baseline agent that uses GPT-4o to complete tasks."""

    def __init__(self, openai_client: OpenAI):
        self.llm = openai_client
        self.conversation: list = []

    def reset(self):
        self.conversation = []

    def act(self, state: Dict) -> Dict:
        """Given current state, return next action as a dict."""
        # Build user message
        user_msg = self._build_user_message(state)
        self.conversation.append({"role": "user", "content": user_msg})

        response = self.llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation,
            temperature=0.0,
            max_tokens=1500,
        )

        raw = response.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": raw})

        try:
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            print(f"  [WARN] Could not parse action JSON: {e}")
            print(f"  Raw response: {raw[:200]}")
            action = {"type": "reply", "to": [], "subject": "Error", "body": raw}

        return action

    def _build_user_message(self, state: Dict) -> str:
        ctx = state.get("context", {})
        feedback = state.get("feedback", "")
        history = state.get("history", [])

        msg_parts = [
            f"=== TASK: {state.get('task_id')} (Difficulty: {state.get('difficulty')}) ===",
            f"",
            f"DESCRIPTION: {state.get('task_description')}",
            f"",
            f"INSTRUCTIONS: {state.get('instructions')}",
            f"",
            f"CONTEXT:",
            json.dumps(ctx, indent=2),
        ]

        if history:
            msg_parts.append(f"\nPREVIOUS STEPS:")
            for h in history[-3:]:  # Last 3 for context
                msg_parts.append(f"  Step {h['step']}: {h['action'].get('type')} → reward={h['reward']:.2f}")
                msg_parts.append(f"  Feedback: {h['feedback'][:200]}")

        if feedback and "Task started" not in feedback:
            msg_parts.append(f"\nLAST FEEDBACK:\n{feedback}")

        msg_parts.append(f"\nNow provide your next action as a JSON object.")
        return "\n".join(msg_parts)


# ──────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────

def run_task(env: EnvClient, agent: ExecutiveAssistAgent, task_id: str) -> Dict:
    """Run a single task to completion. Returns result summary."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    agent.reset()
    state = env.reset(task_id=task_id)

    print(f"  Description: {state['task_description']}")
    print(f"  Difficulty:  {state['difficulty']}")
    print()

    total_reward = 0.0
    steps_taken = 0
    rewards = []

    for step_num in range(1, MAX_STEPS + 1):
        if state.get("done"):
            break

        print(f"  [Step {step_num}] Thinking...", end=" ", flush=True)
        action = agent.act(state)
        print(f"action_type={action.get('type')!r}")

        result = env.step(action)
        state    = result["state"]
        reward   = result["reward"]
        done     = result["done"]

        total_reward += reward
        rewards.append(reward)
        steps_taken = step_num

        print(f"           reward={reward:.2f}  done={done}")
        # Print first line of feedback
        fb_line = state.get("feedback", "").split("\n")[0]
        print(f"           feedback: {fb_line}")

        if done:
            break

    final_reward = max(rewards) if rewards else 0.0
    success = final_reward >= 0.70

    print()
    print(f"  RESULT: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"  Steps:  {steps_taken}   Best reward: {final_reward:.2f}")
    print(f"  [END] success={success} steps={steps_taken} score={final_reward:.2f} "
          f"rewards={','.join(f'{r:.2f}' for r in rewards)}")

    return {
        "task_id": task_id,
        "difficulty": state.get("difficulty"),
        "success": success,
        "steps": steps_taken,
        "best_reward": final_reward,
        "rewards": rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="ExecutiveAssist-Env baseline agent")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment server URL")
    parser.add_argument("--task", default=None, help="Run a specific task only")
    parser.add_argument("--all", action="store_true", help="Run all 9 tasks")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)
    env_client    = EnvClient(base_url=args.base_url)
    agent         = ExecutiveAssistAgent(openai_client=openai_client)

    # Check server health
    try:
        tasks_info = env_client.tasks()
        print(f"Connected to ExecutiveAssist-Env")
        print(f"Available tasks: {tasks_info['all']}")
    except Exception as e:
        print(f"ERROR: Could not connect to environment at {args.base_url}: {e}")
        sys.exit(1)

    # Decide which tasks to run
    if args.task:
        task_ids = [args.task]
    elif args.all:
        task_ids = tasks_info["all"]
    else:
        # Default: run one of each difficulty
        task_ids = ["schedule_meeting", "inbox_triage", "meeting_notes_extraction"]

    # Run
    results = []
    for task_id in task_ids:
        result = run_task(env_client, agent, task_id)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    by_diff: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for r in results:
        diff = r.get("difficulty", "easy")
        by_diff.setdefault(diff, []).append(r)

    total_success = sum(1 for r in results if r["success"])
    avg_score = sum(r["best_reward"] for r in results) / len(results) if results else 0

    for diff, diff_results in by_diff.items():
        if not diff_results:
            continue
        ok = sum(1 for r in diff_results if r["success"])
        print(f"  {diff.upper():8s}: {ok}/{len(diff_results)} passed  "
              f"avg={sum(r['best_reward'] for r in diff_results)/len(diff_results):.2f}")

    print(f"\n  TOTAL: {total_success}/{len(results)} passed  avg_score={avg_score:.2f}")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"    {status} {r['task_id']:35s}  score={r['best_reward']:.2f}  steps={r['steps']}")


if __name__ == "__main__":
    main()
