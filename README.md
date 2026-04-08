---
title: Executive Assist
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# 🗂️ ExecutiveAssist-Env

**An OpenEnv RL environment where an AI agent acts as a personal executive assistant.**

Unlike game-based environments, every task here maps directly to a real workflow that an executive assistant handles daily — scheduling across time zones, triaging a cluttered inbox, extracting decisions from meeting transcripts, and building conflict-free daily plans under hard constraints.

---

## ⚡ Quick Start

### Option A — Docker (recommended)
```bash
docker build -t executiveassist-env .
docker run -p 7860:7860 executiveassist-env
```

### Option B — Direct
```bash
pip install -r requirements.txt
python server.py
```

### Smoke test
```bash
# 1. Reset to a task
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "schedule_meeting", "seed": 42}' | python -m json.tool

# 2. Submit the correct action
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "type": "schedule",
      "date": "2025-07-15",
      "start_time": "15:00",
      "end_time": "16:00",
      "attendees": ["Alex Chen", "Jordan Lee", "Sam Rivera"],
      "title": "Q3 Planning Sync"
    }
  }' | python -m json.tool
# → reward: 1.0, done: true
```

### Run baseline agent
```bash
export OPENAI_API_KEY=sk-...
python inference.py --all                         # all 9 tasks
python inference.py --task meeting_notes_extraction  # single task
```

---

## 🎯 Task Design

Nine tasks across three difficulty tiers, each graded with a **partial-credit rubric**.

### Easy
| Task | What the agent must do | Key skill tested |
|------|----------------------|-----------------|
| `schedule_meeting` | Find the first free 1-hour slot across 3 calendars | Overlap detection |
| `confirm_slot` | Choose the conflict-free option from 3 proposed times | Constraint checking |
| `cancel_meeting` | Cancel an event + send a polite apology note (2 actions) | Multi-step sequencing |

### Medium
| Task | What the agent must do | Key skill tested |
|------|----------------------|-----------------|
| `inbox_triage` | Sort 8 emails into URGENT / IMPORTANT / DELEGATE / ARCHIVE | Priority reasoning |
| `reschedule_conflict` | Identify a calendar clash, move the lower-priority meeting | Priority + rescheduling |
| `draft_reply` | Write a professional, empathetic response to an angry client | Tone + NLP |

### Hard
| Task | What the agent must do | Key skill tested |
|------|----------------------|-----------------|
| `multi_party_schedule` | Best-compromise 90-min slot for 5 people across 4 time zones | Timezone arithmetic |
| `meeting_notes_extraction` | Extract all action items (owner + date), decisions, open questions from a transcript | Comprehension + structured output |
| `full_day_plan` | Build a no-overlap full-day plan from 4 meetings, 4 tasks, 3 urgent emails, and a travel requirement | Constraint satisfaction |

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.
```json
{ "task_id": "inbox_triage", "seed": 42 }
```
Returns the initial state. `task_id` and `seed` are optional.

### `POST /step`
Submit one action.
```json
{ "action": { "type": "triage", "assignments": { "email-1": "URGENT", ... } } }
```
Returns:
```json
{
  "state":  { "task_id": "...", "context": {...}, "feedback": "...", "done": false, ... },
  "reward": 0.80,
  "done":   false,
  "info":   { "step_count": 1, "max_steps": 10 }
}
```

### `GET /state` — current state (no side effects)
### `GET /tasks` — list all task IDs by difficulty
### `GET /health` — liveness check

---

## 🎮 Action Schema

```jsonc
// Book or confirm a meeting
{ "type": "schedule", "date": "2025-07-15", "start_time": "15:00", "end_time": "16:00",
  "attendees": ["Alex Chen", "Jordan Lee"], "title": "Q3 Sync" }

// Cancel an event
{ "type": "cancel", "event_id": "evt-002", "reason": "family emergency" }

// Move an event
{ "type": "reschedule", "event_id": "evt-B",
  "new_date": "2025-07-16", "new_start_time": "15:00", "new_end_time": "16:00" }

// Send a reply / notification
{ "type": "reply", "to": ["email@example.com"], "subject": "Re: ...", "body": "..." }

// Triage inbox
{ "type": "triage",
  "assignments": { "email-1": "URGENT", "email-2": "ARCHIVE", "email-3": "DELEGATE" } }

// Extract from meeting notes
{ "type": "extract",
  "action_items": [{ "task": "...", "owner": "Ren", "due_date": "2025-07-18" }],
  "decisions": ["Ship mobile by July 28th"],
  "open_questions": ["Web version timeline?"] }

// Full day plan
{ "type": "plan",
  "schedule": [
    { "start_time": "09:00", "end_time": "09:30", "type": "meeting", "title": "Standup" },
    { "start_time": "09:30", "end_time": "11:30", "type": "task",    "title": "Write board deck" }
  ]}
```

---

## 📊 Grading

All tasks use **rubric-based partial credit**. An episode succeeds when reward ≥ 0.70.

Example rubric for `meeting_notes_extraction`:

| Criterion | Weight | What's checked |
|-----------|--------|---------------|
| Action items found | 30% | Keyword match against transcript |
| Action items have owners | 15% | Each item has an `owner` field |
| Action items have due dates | 15% | Each item has a `due_date` field |
| Decisions found | 20% | Key decisions extracted |
| Open questions found | 20% | Unresolved items flagged |

Example rubric for `draft_reply`:

| Criterion | Weight | What's checked |
|-----------|--------|---------------|
| Acknowledges delay | 20% | Delay/late mentioned |
| Contains apology | 25% | Sorry/apologize present |
| Concrete next step | 25% | Date, timeline, or commitment given |
| Professional tone | 20% | No defensive language |
| No defensive language | 10% | Double-weighted safety check |

---

## 🏗️ Architecture

```
my_env.py      ← Core environment (reset / step / state API)
tasks.py       ← All 9 task definitions with rich context data
graders.py     ← Per-task rubric graders with partial credit
server.py      ← FastAPI HTTP wrapper (OpenEnv-compliant)
inference.py   ← GPT-4o baseline agent
openenv.yaml   ← Environment metadata
Dockerfile     ← Production container (Python 3.11-slim)
```

The environment is **fully deterministic** given the same seed — no randomness in grading.

---

## 💡 Design Rationale

**Why an executive assistant?**
- Every task is a real, verifiable workflow — not a game abstraction
- Natural difficulty progression: slot-finding → priority judgement → constraint satisfaction
- Graders are deterministic and explainable (rubric output is human-readable)
- Hard tasks are genuinely hard: `multi_party_schedule` has timezone constraints that make a perfect solution impossible, rewarding agents that reason about trade-offs

**Partial credit philosophy:**
Agents that get the right *type* of action but wrong details score ~0.15–0.30. Agents that nail 60–70% of criteria score 0.60–0.75 and can iterate within the 10-step budget.

---

## License
MIT