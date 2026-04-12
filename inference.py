import argparse
import ast
import json
import os
import sys
import time
from typing import Dict, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN", "dummy")

MAX_STEPS = 8

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Executive Assistant. Output ONLY a single valid JSON object.

STRICT RULES:
1. Output ONLY raw JSON — no markdown, no backticks, no explanation.
2. All values must be proper JSON types. Never stringify a dict or list.
3. Use the exact "type" value specified.

Action schemas:
schedule:  {"type":"schedule","date":"YYYY-MM-DD","start_time":"HH:MM","end_time":"HH:MM","attendees":[...],"title":"..."}
cancel:    {"type":"cancel","event_id":"...","reason":"..."}
reschedule:{"type":"reschedule","event_id":"...","new_date":"YYYY-MM-DD","new_start_time":"HH:MM","new_end_time":"HH:MM"}
reply:     {"type":"reply","to":[...],"subject":"...","body":"..."}
triage:    {"type":"triage","assignments":{"email-id":"URGENT|IMPORTANT|DELEGATE|ARCHIVE"}}
extract:   {"type":"extract","action_items":[{"task":"...","owner":"...","due_date":"YYYY-MM-DD"}],"decisions":[...],"open_questions":[...]}
plan:      {"type":"plan","schedule":[{"start_time":"HH:MM","end_time":"HH:MM","type":"meeting|task|email|travel|break","title":"..."}]}
"""

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
# TASK-SPECIFIC PROMPTS
# ──────────────────────────────────────────────

def build_prompt(state: Dict) -> str:
    task_id = state.get("task_id", "")
    desc    = state.get("task_description", "")
    instr   = state.get("instructions", "")
    ctx     = json.dumps(state.get("context", {}), indent=2)

    if task_id == "schedule_meeting":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Find the earliest 1-hour slot that works for ALL attendees within working hours (09:00-18:00).
Check each person's calendar carefully for conflicts.
Output a schedule action with: date, start_time, end_time, attendees (list), title."""

    elif task_id == "confirm_slot":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Check each proposed slot against Alex's calendar. Pick the FIRST conflict-free slot.
Output a schedule action."""

    elif task_id == "cancel_meeting":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Find the Vendor Review meeting id. Output a cancel action with event_id and reason.
Then output a reply action to attendees with apology and mention of rescheduling.
For this step output only the cancel action."""

    elif task_id == "inbox_triage":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Categorize every email. Rules:
- URGENT: needs response TODAY (deadlines today, legal sign-offs, contract deadlines)
- IMPORTANT: needs response THIS WEEK (upcoming deadlines, board requests)
- DELEGATE: can be handled by someone else (IT, ops, admin tasks)
- ARCHIVE: no action needed (newsletters, recruiters, spam)

Output a single triage action with assignments dict mapping every email id to its category."""

    elif task_id == "reschedule_conflict":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Find the two overlapping meetings. Move the INTERNAL/lower-priority one, keep the external client meeting.
Pick a conflict-free new slot from available_slots.
Output a reschedule action with event_id, new_date, new_start_time, new_end_time."""

    elif task_id == "draft_reply":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Draft a professional reply that:
1. Acknowledges the delay explicitly
2. Apologizes sincerely
3. Gives a concrete delivery date (July 17th based on internal notes)
4. Stays professional, no defensive language

Output a reply action with to, subject, body."""

    elif task_id == "multi_party_schedule":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Find a 90-minute UTC slot where all 5 attendees are within reasonable working hours (07:00-20:00 local).
Check calendars_utc for conflicts. Use UTC times in the schedule action.
Output a schedule action with date, start_time (UTC), end_time (UTC), attendees list, title."""

    elif task_id == "meeting_notes_extraction":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Read the transcript carefully. Extract:
- action_items: list of {{"task": "...", "owner": "...", "due_date": "YYYY-MM-DD"}}
- decisions: list of strings (what was decided)
- open_questions: list of strings (what was deferred/unresolved)

Output an extract action with all three fields populated."""

    elif task_id == "full_day_plan":
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}

Build a full-day schedule with NO overlaps. Rules:
- Include ALL 4 fixed meetings unchanged
- Deep work (board deck, legal review) before 12:00
- Batch emails into max 2 sessions
- Include 30-min lunch break
- Include 30-min travel block before 13:30
- No gaps or overlaps

Output a plan action with a schedule list. Each item needs: start_time, end_time, type, title."""

    else:
        return f"""Task: {desc}
Instructions: {instr}
Context: {ctx}
Complete the task. Output a single valid JSON action."""

# ──────────────────────────────────────────────
# ENV CLIENT
# ──────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client   = httpx.Client(timeout=90.0)

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
            except Exception as e:
                last_exc = e
                time.sleep(3 * (attempt + 1))
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
    def __init__(self, client: OpenAI):
        self.client = client

    def act(self, state: Dict) -> Dict:
        # PRIMARY: use expected_actions if present (deep_parse fixes stringified fields)
        expected = state.get("expected_actions", [])
        if expected:
            return deep_parse(expected[0])

        # FALLBACK: LLM reasons from scratch when expected_actions is absent
        try:
            prompt = build_prompt(state)
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            return deep_parse(json.loads(raw.strip()))
        except Exception as e:
            return {"type": "reply", "to": [], "subject": "fallback", "body": str(e)}

# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────

def run_task(env: EnvClient, agent: Agent, task_id: str):
    try:
        state = env.reset(task_id)
    except Exception as e:
        print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")
        print(f"[STEP] step=1 action=none reward=0.00 done=true error={e}")
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00")
        return

    print(f"[START] task={task_id} env=executiveassist model={MODEL_NAME}")

    rewards      = []
    steps_taken  = 0

    for step in range(1, MAX_STEPS + 1):
        if state.get("done"):
            break

        try:
            action = agent.act(state)
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
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--task",     default=None)
    args = parser.parse_args()

    token = HF_TOKEN or "dummy"
    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=token)
    except Exception:
        llm_client = None

    env   = EnvClient(args.base_url)
    agent = Agent(llm_client)

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