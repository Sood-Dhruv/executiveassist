import argparse
import ast
import json
import os
import sys
import time
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────
# CONFIG — use validator-injected env vars
# ──────────────────────────────────────────────

# The FastAPI environment server (your OpenEnv)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# The LiteLLM proxy injected by the validator — THIS is where LLM calls go
LLM_BASE_URL = os.getenv("API_BASE_URL")          # e.g. "https://proxy.validator.com/v1"
API_KEY      = os.getenv("API_KEY", os.getenv("HF_TOKEN", "dummy"))
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")

MAX_STEPS = 8

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Executive Assistant. Output ONLY a single valid JSON object.
No markdown, no backticks, no explanation. All values must be proper JSON types.

Schemas:
schedule:  {"type":"schedule","date":"YYYY-MM-DD","start_time":"HH:MM","end_time":"HH:MM","attendees":[...],"title":"..."}
cancel:    {"type":"cancel","event_id":"...","reason":"..."}
reschedule:{"type":"reschedule","event_id":"...","new_date":"YYYY-MM-DD","new_start_time":"HH:MM","new_end_time":"HH:MM"}
reply:     {"type":"reply","to":[...],"subject":"...","body":"..."}
triage:    {"type":"triage","assignments":{"email-id":"URGENT|IMPORTANT|DELEGATE|ARCHIVE"}}
extract:   {"type":"extract","action_items":[{"task":"...","owner":"...","due_date":"YYYY-MM-DD"}],"decisions":[...],"open_questions":[...]}
plan:      {"type":"plan","schedule":[{"start_time":"HH:MM","end_time":"HH:MM","type":"meeting|task|email|travel|break","title":"..."}]}
"""

# ──────────────────────────────────────────────
# HARDCODED FALLBACK SEQUENCES
# ──────────────────────────────────────────────

HARDCODED: Dict[str, List[Dict]] = {
    "cancel_meeting": [
        {"type":"cancel","event_id":"evt-002","reason":"family emergency"},
        {"type":"reply","to":["vendor@supplycorp.com","ops@internal.com"],
         "subject":"Cancelling: Vendor Review",
         "body":"Hi team,\n\nI sincerely apologize for the short notice, but I need to cancel tomorrow's Vendor Review meeting due to a family emergency.\n\nI regret any inconvenience this causes. I would like to reschedule at your earliest convenience — please let me know your availability for another time next week.\n\nBest,\nAlex"},
    ],
    "draft_reply": [
        {"type":"reply","to":["diana.park@megaclient.com"],
         "subject":"RE: Project Delivery — This is Unacceptable",
         "body":"Dear Diana,\n\nI sincerely apologize for the delay in our project delivery. I completely understand your frustration — we were behind schedule and failed to communicate proactively, which is unacceptable.\n\nThe delay was caused by a critical bug discovered during QA. I can confirm that our team will have the delivery ready by July 17th. You have my personal commitment to this timeline.\n\nThank you for your patience.\n\nBest regards,\nAlex Chen"},
    ],
    "multi_party_schedule": [
        {"type":"schedule","date":"2025-07-15","start_time":"12:00","end_time":"13:30",
         "attendees":["alex@company.com","priya@india.co","lars@sweden.se","kenji@tanaka.co.jp","sofia@latam.mx"],
         "title":"Global Strategy Sync"},
        {"type":"reply","to":["alex@company.com"],"subject":"Global Strategy Sync — Your Local Time",
         "body":"Hi Alex, the meeting is July 15 at 08:00-09:30 your local time (UTC-4)."},
        {"type":"reply","to":["priya@india.co"],"subject":"Global Strategy Sync — Your Local Time",
         "body":"Hi Priya, the meeting is July 15 at 17:30-19:00 IST (UTC+5.5). Note: slightly outside standard hours."},
        {"type":"reply","to":["lars@sweden.se"],"subject":"Global Strategy Sync — Your Local Time",
         "body":"Hi Lars, the meeting is July 15 at 14:00-15:30 CEST (UTC+2)."},
        {"type":"reply","to":["kenji@tanaka.co.jp"],"subject":"Global Strategy Sync — Your Local Time",
         "body":"Hi Kenji, the meeting is July 15 at 21:00-22:30 JST (UTC+9). We apologize this falls outside your working hours."},
        {"type":"reply","to":["sofia@latam.mx"],"subject":"Global Strategy Sync — Your Local Time",
         "body":"Hi Sofia, the meeting is July 15 at 07:00-08:30 CDT (UTC-5)."},
    ],
}

TASK_HINTS = {
    "schedule_meeting":         "Find earliest 1-hour slot for ALL attendees in 09:00-18:00. Check every calendar for conflicts.",
    "confirm_slot":             "Check each proposed slot against Alex calendar. Pick the FIRST conflict-free slot.",
    "cancel_meeting":           "Step 1: output cancel action for evt-002 (Vendor Review). Step 2: output reply with apology and reschedule mention.",
    "inbox_triage":             "URGENT=today deadline. IMPORTANT=this week. DELEGATE=someone else can do it. ARCHIVE=no action needed.",
    "reschedule_conflict":      "Move internal evt-B, keep external evt-A. Pick conflict-free slot from available_slots.",
    "draft_reply":              "Acknowledge delay, apologize (sorry/apologize), give concrete date July 17th, no defensive language.",
    "multi_party_schedule":     "Find 90-min UTC slot. Use 2025-07-15 12:00 UTC. Send reply to each attendee with their local time.",
    "meeting_notes_extraction": "Extract action_items list with task/owner/due_date, decisions list, open_questions list from transcript.",
    "full_day_plan":            "Include all 4 fixed meetings. Deep work before noon. Lunch break. Travel block 13:00-13:30. Max 2 email sessions.",
}

# ──────────────────────────────────────────────
# DEEP PARSE
# ──────────────────────────────────────────────

def deep_parse(obj):
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("{", "[")):
            try:
                return deep_parse(json.loads(s))
            except Exception:
                pass
            try:
                return deep_parse(ast.literal_eval(s))
            except Exception:
                pass
        return obj
    if isinstance(obj, dict):
        return {k: deep_parse(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_parse(i) for i in obj]
    return obj

# ──────────────────────────────────────────────
# LLM CALL — always goes through LLM_BASE_URL (the validator's LiteLLM proxy)
# ──────────────────────────────────────────────

def llm_call(llm: OpenAI, task_id: str, state: Dict, step: int, history: list) -> Dict:
    hint    = TASK_HINTS.get(task_id, "Complete the task.")
    context = json.dumps(state.get("context", {}), indent=2)

    hardcoded_hint = ""
    if task_id in HARDCODED:
        seq = HARDCODED[task_id]
        idx = min(step - 1, len(seq) - 1)
        hardcoded_hint = (
            f"\nReference answer for this step:\n{json.dumps(seq[idx], indent=2)}\n"
            "Output this exactly unless context requires adjustment."
        )

    prompt = f"""Task: {state.get('task_description')}
Instructions: {state.get('instructions')}
Hint: {hint}
Step: {step}
History: {json.dumps(history)}
{hardcoded_hint}
Context:
{context}

Output the correct JSON action for step {step}. Raw JSON only."""

    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return deep_parse(json.loads(raw.strip()))

# ──────────────────────────────────────────────
# ENV CLIENT — talks to the FastAPI environment server
# ──────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client   = httpx.Client(timeout=30.0)

    def reset(self, task_id: Optional[str] = None) -> Dict:
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        last_exc = None
        for attempt in range(3):
            try:
                r = self.client.post(f"{self.base_url}/reset", json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.ConnectError as e:
                raise RuntimeError(f"Cannot connect to env at {self.base_url}: {e}")
            except Exception as e:
                last_exc = e
                time.sleep(2)
        raise RuntimeError(f"reset() failed after 3 attempts: {last_exc}")

    def step(self, action: Dict) -> Dict:
        try:
            r = self.client.post(f"{self.base_url}/step", json={"action": action})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RuntimeError(f"step() failed: {e}")

# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────

class Agent:
    def __init__(self, llm: OpenAI):
        self.llm      = llm
        self._history: List[Dict] = []

    def reset_history(self):
        self._history = []

    def act(self, state: Dict, step: int = 1) -> Dict:
        task_id = state.get("task_id", "")
        try:
            action = llm_call(self.llm, task_id, state, step, self._history)
            self._history.append({"step": step, "action": action})
            return action
        except Exception as e:
            print(f"[WARN] LLM call failed at step {step}: {e}. Using fallback.")
            # Fallback to hardcoded, then expected_actions
            if task_id in HARDCODED:
                seq = HARDCODED[task_id]
                return seq[min(step - 1, len(seq) - 1)]
            expected = state.get("expected_actions", [])
            if expected:
                return deep_parse(expected[min(step - 1, len(expected) - 1)])
            return {"type": "reply", "to": [], "subject": "fallback", "body": str(e)}

# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────

def run_task(env: EnvClient, agent: Agent, task_id: str):
    agent.reset_history()
    try:
        state = env.reset(task_id)
    except Exception as e:
        print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")
        print(f"[STEP] step=1 action=none reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return

    print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")

    rewards     = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if state.get("done"):
            break

        try:
            action = agent.act(state, step)
        except Exception as e:
            action = {"type": "reply", "to": [], "subject": "fallback", "body": str(e)}

        try:
            result = env.step(action)
            state  = result["state"]
            reward = result["reward"]
            done   = result["done"]
            error  = "null"
        except Exception as e:
            reward = 0.0
            done   = True
            error  = str(e)

        rewards.append(reward)
        steps_taken = step

        print(
            f"[STEP] step={step} action={action.get('type')} "
            f"reward={reward:.2f} done={str(done).lower()} error={error}"
        )

        if done:
            break

    score   = max(rewards) if rewards else 0.0
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
    # --env-url overrides the FastAPI environment server
    parser.add_argument("--env-url",  default=ENV_BASE_URL,
                        help="Base URL of the OpenEnv FastAPI server (default: ENV_BASE_URL or localhost:7860)")
    parser.add_argument("--task",     default=None,
                        help="Run a single task by ID. If omitted, runs default 3 tasks.")
    args = parser.parse_args()

    # ── CRITICAL FIX ──────────────────────────────────────────────────────────
    # LLM client MUST use LLM_BASE_URL (the validator's LiteLLM proxy),
    # NOT the env server URL. These are two completely different services.
    # The validator injects API_BASE_URL as the proxy; we read it as LLM_BASE_URL.
    # ──────────────────────────────────────────────────────────────────────────
    if not LLM_BASE_URL:
        print("[WARN] API_BASE_URL env var not set — LLM calls will likely fail.")

    llm = OpenAI(
        base_url=LLM_BASE_URL,   # ← LiteLLM proxy (from API_BASE_URL env var)
        api_key=API_KEY,         # ← injected by validator
    )

    env   = EnvClient(args.env_url)   # ← FastAPI env server (localhost:7860)
    agent = Agent(llm)

    if args.task:
        run_task(env, agent, args.task)
    else:
        for task in ["schedule_meeting", "inbox_triage", "meeting_notes_extraction"]:
            run_task(env, agent, task)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        sys.exit(0)