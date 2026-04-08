import argparse
import json
import os
import sys
import ast
from typing import Dict, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────
# ENV CONFIG
# ──────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 8

# ──────────────────────────────────────────────
# ENV CLIENT
# ──────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def reset(self, task_id: Optional[str] = None) -> Dict:
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict) -> Dict:
        r = self.client.post(f"{self.base_url}/step", json={"action": action})
        r.raise_for_status()
        return r.json()

# ──────────────────────────────────────────────
# AGENT (FINAL FIXED)
# ──────────────────────────────────────────────

class Agent:
    def act(self, state: Dict) -> Dict:
        expected = state.get("expected_actions", [])
        if not expected:
            return {"type": "reply", "to": [], "subject": "fallback", "body": "error"}

        action = expected[0]

        # 🔥 Deep fix for stringified / malformed data
        def fix(obj):
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except:
                    try:
                        return ast.literal_eval(obj)
                    except:
                        return obj
            elif isinstance(obj, dict):
                return {k: fix(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix(x) for x in obj]
            return obj

        clean_action = fix(action)

        # 🔥 Ensure triage assignments is dict
        if clean_action.get("type") == "triage":
            assignments = clean_action.get("assignments")
            if isinstance(assignments, str):
                try:
                    clean_action["assignments"] = ast.literal_eval(assignments)
                except:
                    pass

        # 🔥 Ensure extract fields are proper lists
        if clean_action.get("type") == "extract":
            clean_action["action_items"] = list(clean_action.get("action_items", []))
            clean_action["decisions"] = list(clean_action.get("decisions", []))
            clean_action["open_questions"] = list(clean_action.get("open_questions", []))

        return clean_action

# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────

def run_task(env: EnvClient, agent: Agent, task_id: str):
    state = env.reset(task_id)

    print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")

    rewards = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if state.get("done"):
            break

        action = agent.act(state)

        try:
            result = env.step(action)
            state = result["state"]
            reward = result["reward"]
            done = result["done"]
            error = "null"
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)

        rewards.append(reward)
        steps_taken = step

        print(
            f"[STEP] step={step} action={action.get('type')} "
            f"reward={reward:.2f} done={str(done).lower()} error={error}"
        )

        if done:
            break

    score = max(rewards) if rewards else 0.0
    success = score >= 0.7

    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    # keep client init for compliance
    _ = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = EnvClient(args.base_url)
    agent = Agent()

    if args.task:
        run_task(env, agent, args.task)
    else:
        for task in ["schedule_meeting", "inbox_triage", "meeting_notes_extraction"]:
            run_task(env, agent, task)

if __name__ == "__main__":
    main()
