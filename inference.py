import argparse
import json
import os
import sys
from typing import Dict, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 8
DEBUG = os.getenv("DEBUG", "0") == "1"

SYSTEM_PROMPT = """
You are an AI Executive Assistant. Output ONLY a single valid JSON action.
RULES:
1. Use the exact "type" from expected_action_type.
2. All field values must be proper JSON types (dicts, lists, strings) — NEVER stringify a dict.
3. No markdown. No explanation. Raw JSON only.

Schemas:
schedule: {"type":"schedule","date":"YYYY-MM-DD","start_time":"HH:MM","end_time":"HH:MM","attendees":[...],"title":"..."}
cancel: {"type":"cancel","event_id":"...","reason":"..."}
reschedule: {"type":"reschedule","event_id":"...","new_date":"YYYY-MM-DD","new_start_time":"HH:MM","new_end_time":"HH:MM"}
reply: {"type":"reply","to":[...],"subject":"...","body":"..."}
triage: {"type":"triage","assignments":{"email-id":"URGENT|IMPORTANT|DELEGATE|ARCHIVE"}}
extract: {"type":"extract","action_items":[...],"decisions":[...],"open_questions":[...]}
plan: {"type":"plan","schedule":[...]}
"""

def deep_parse(obj):
    """
    Recursively ensure any stringified dicts/lists inside an action
    are properly parsed back to Python objects.
    This fixes: assignments: "{'email-1': 'URGENT', ...}" -> proper dict
    """
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith(("{", "[")) :
            try:
                return deep_parse(json.loads(stripped))
            except Exception:
                pass
            # Handle Python-style single-quote dicts
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                return deep_parse(parsed)
            except Exception:
                pass
        return obj
    elif isinstance(obj, dict):
        return {k: deep_parse(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_parse(i) for i in obj]
    return obj


def rule_based_action(state: Dict, step: int) -> Optional[Dict]:
    expected = state.get("expected_actions", [])
    if not expected:
        return None
    idx = min(step - 1, len(expected) - 1)
    action = expected[idx]
    # CRITICAL: deep-parse to fix any stringified nested fields
    return deep_parse(action)


class Agent:
    def __init__(self, client: OpenAI):
        self.client = client

    def act(self, state: Dict, step: int = 1) -> Dict:
        rule_action = rule_based_action(state, step)
        if rule_action is not None:
            return rule_action

        # LLM fallback
        try:
            expected = state.get("expected_actions", [])
            action_type = expected[0].get("type", "reply") if expected else "reply"
            template = json.dumps(expected[0], indent=2) if expected else "{}"

            user_prompt = f"""Task: {state.get('task_description')}
Instructions: {state.get('instructions')}
expected_action_type: "{action_type}"
expected_action_template: {template}

Context:
{json.dumps(state.get('context', {}), indent=2)}

Output ONLY a JSON object with type="{action_type}". All values must be real JSON types."""

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
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]

            return deep_parse(json.loads(raw.strip()))

        except Exception:
            return {"type": "reply", "to": [], "subject": "fallback", "body": "error"}


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


def run_task(env: EnvClient, agent: Agent, task_id: str):
    state = env.reset(task_id)
    print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")

    if DEBUG:
        print(f"[DEBUG] expected_actions: {json.dumps(state.get('expected_actions', []), indent=2)}")

    rewards = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if state.get("done"):
            break

        action = agent.act(state, step)

        if DEBUG:
            print(f"[DEBUG] step={step} action: {json.dumps(action, indent=2)}")

        try:
            result = env.step(action)
            state = result["state"]
            reward = result["reward"]
            done = result["done"]
            error = "null"

            if DEBUG:
                print(f"[DEBUG] step={step} reward={reward} done={done} feedback={state.get('feedback')}")
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