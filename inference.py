

import argparse
import json
import os
import sys
import textwrap
from typing import Dict, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────
# ENV VARIABLES
# ──────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 8

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert AI Executive Assistant.

You must output EXACTLY one valid JSON object representing an action.

Allowed actions:

schedule:
{"type": "schedule", "date": "YYYY-MM-DD", "start_time": "HH:MM", "end_time": "HH:MM", "attendees": [...], "title": "..."}

cancel:
{"type": "cancel", "event_id": "...", "reason": "..."}

reschedule:
{"type": "reschedule", "event_id": "...", "new_date": "YYYY-MM-DD", "new_start_time": "HH:MM", "new_end_time": "HH:MM"}

reply:
{"type": "reply", "to": [...], "subject": "...", "body": "..."}

triage:
{"type": "triage", "assignments": {"email-id": "URGENT|IMPORTANT|DELEGATE|ARCHIVE"}}

extract:
{"type": "extract", "action_items": [...], "decisions": [...], "open_questions": [...]}

plan:
{"type": "plan", "schedule": [...]}

Rules:
- Output ONLY JSON
- No explanation
- No markdown
- Use correct field names exactly
"""

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
# AGENT
# ──────────────────────────────────────────────

class Agent:
    def __init__(self, client: OpenAI):
        self.client = client

    def act(self, state: Dict) -> Dict:
        try:
            user_prompt = f"""
Task: {state.get('task_description')}

Instructions: {state.get('instructions')}

Expected Action Format:
{json.dumps(state.get('expected_actions', []), indent=2)}

Context:
{json.dumps(state.get('context', {}), indent=2)}

Return ONLY valid JSON action.
"""

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=500,
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())

        except Exception:
            # fallback action (safe)
            return {"type": "reply", "to": [], "subject": "fallback", "body": "error"}

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

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(args.base_url)
    agent = Agent(client)

    if args.task:
        run_task(env, agent, args.task)
    else:
        for task in ["schedule_meeting", "inbox_triage", "meeting_notes_extraction"]:
            run_task(env, agent, task)

if __name__ == "__main__":
    main()
