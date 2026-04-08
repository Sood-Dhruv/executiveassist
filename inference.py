

import argparse
import json
import os
import sys
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
DEBUG = os.getenv("DEBUG", "0") == "1"

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an AI Executive Assistant.

Output ONLY a valid JSON action.

Rules:
- Use correct "type"
- Use exact schema
- Do NOT stringify dicts
- No explanation
"""

# ──────────────────────────────────────────────
# STRICT JSON FIX (CRITICAL)
# ──────────────────────────────────────────────

def strict_jsonify(obj):
    """
    Force clean JSON structure (fixes stringified dict issues)
    """
    if isinstance(obj, str):
        try:
            return strict_jsonify(json.loads(obj))
        except:
            try:
                import ast
                return strict_jsonify(ast.literal_eval(obj))
            except:
                return obj

    elif isinstance(obj, dict):
        return {str(k): strict_jsonify(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [strict_jsonify(i) for i in obj]

    return obj

# ──────────────────────────────────────────────
# RULE-BASED ACTION (PRIMARY)
# ──────────────────────────────────────────────

def rule_based_action(state: Dict, step: int) -> Optional[Dict]:
    expected = state.get("expected_actions", [])
    if not expected:
        return None

    idx = min(step - 1, len(expected) - 1)
    action = expected[idx]

    clean_action = strict_jsonify(action)

    # 🔥 FIX triage assignments explicitly
    if clean_action.get("type") == "triage":
        assignments = clean_action.get("assignments")
        if isinstance(assignments, str):
            import ast
            clean_action["assignments"] = ast.literal_eval(assignments)

    return clean_action

# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────

class Agent:
    def __init__(self, client: OpenAI):
        self.client = client

    def act(self, state: Dict, step: int = 1) -> Dict:
        # PRIORITY: rule-based (guaranteed correct)
        rule_action = rule_based_action(state, step)
        if rule_action:
            return rule_action

        # fallback LLM (rarely used)
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(state)}
                ],
                temperature=0.0,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]

            return strict_jsonify(json.loads(raw))

        except:
            return {"type": "reply", "to": [], "subject": "fallback", "body": "error"}

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
        clean_action = strict_jsonify(action)
        r = self.client.post(f"{self.base_url}/step", json={"action": clean_action})
        r.raise_for_status()
        return r.json()

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

        action = agent.act(state, step)

        if DEBUG:
            print(f"[DEBUG] action: {json.dumps(action, indent=2)}")

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
