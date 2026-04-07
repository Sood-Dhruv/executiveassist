# ExecutiveAssist-Env

**An OpenEnv environment where an AI agent acts as a personal executive assistant.**

The agent handles real-world tasks that a human secretary or EA would manage daily: scheduling meetings, triaging emails, extracting action items from meeting notes, resolving calendar conflicts, and building optimized daily plans.

---

## Quick Start

### Run locally with Docker
```bash
docker build -t executiveassist-env .
docker run -p 7860:7860 executiveassist-env
```

### Run locally without Docker
```bash
pip install -r requirements.txt
python server.py
```

### Test the environment
```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "schedule_meeting"}'

# Submit an action
curl -X POST http://localhost:7860/step \
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
  }'
```

### Run the baseline agent
```bash
export OPENAI_API_KEY=sk-...
python inference.py --all
```

---

## Environment Overview

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Body: `{"task_id": "...", "seed": 42}` |
| `/step`  | POST | Submit one action. Body: `{"action": {...}}` |
| `/state` | GET  | Get current state without advancing |
| `/tasks` | GET  | List all task IDs by difficulty |
| `/health`| GET  | Liveness check |

### State Object

```json
{
  "task_id": "schedule_meeting",
  "task_description": "...",
  "difficulty": "easy",
  "context": { ... },
  "instructions": "...",
  "step": 1,
  "done": false,
  "reward": 0.9,
  "total_reward": 0.9,
  "feedback": "✓ Correct date...",
  "history": [...]
}
```

### Step Response

```json
{
  "state":  { ... },
  "reward": 0.9,
  "done":   true,
  "info":   { "step_count": 1, "max_steps": 10, "task_id": "...", "difficulty": "easy" }
}
```

---

## Tasks

### Easy (3 tasks)

| Task ID | Description |
|---------|-------------|
| `schedule_meeting` | Find the first mutually free 1-hour slot for 3 people |
| `confirm_slot` | Pick the conflict-free option from 3 proposed slots |
| `cancel_meeting` | Cancel a meeting + send apology email to attendees |

### Medium (3 tasks)

| Task ID | Description |
|---------|-------------|
| `inbox_triage` | Categorize 8 emails into URGENT / IMPORTANT / DELEGATE / ARCHIVE |
| `reschedule_conflict` | Resolve a calendar conflict (move lower-priority meeting) |
| `draft_reply` | Write a professional response to an angry client email |

### Hard (3 tasks)

| Task ID | Description |
|---------|-------------|
| `multi_party_schedule` | Schedule 90-min meeting across 5 people in 4 time zones |
| `meeting_notes_extraction` | Extract action items, decisions, open questions from transcript |
| `full_day_plan` | Build an optimized full-day schedule from chaos |

---

## Action Types

```json
// Book a meeting
{"type": "schedule", "date": "2025-07-15", "start_time": "15:00", "end_time": "16:00",
 "attendees": ["..."], "title": "Meeting Name"}

// Cancel an event
{"type": "cancel", "event_id": "evt-002", "reason": "family emergency"}

// Reschedule
{"type": "reschedule", "event_id": "evt-B", "new_date": "2025-07-16",
 "new_start_time": "15:00", "new_end_time": "16:00"}

// Send a reply
{"type": "reply", "to": ["email@example.com"], "subject": "Re: ...", "body": "..."}

// Triage inbox
{"type": "triage", "assignments": {"email-1": "URGENT", "email-2": "ARCHIVE", ...}}

// Extract from meeting notes
{"type": "extract",
 "action_items": [{"task": "...", "owner": "...", "due_date": "2025-07-18"}],
 "decisions": ["..."],
 "open_questions": ["..."]}

// Full day plan
{"type": "plan", "schedule": [
  {"start_time": "09:00", "end_time": "09:30", "type": "meeting", "title": "Standup"},
  ...
]}
```

---

## Grading

All tasks use **partial-credit rubric grading** — agents earn partial rewards for partially correct answers.

- Reward range: `[0.0, 1.0]`
- Episode succeeds when reward ≥ `0.70`
- Max steps per episode: `10`

Example rubric for `schedule_meeting`:

| Criterion | Weight |
|-----------|--------|
| Correct action type | 0.10 |
| Correct date | 0.30 |
| Correct start time (±30 min) | 0.40 |
| All attendees included | 0.20 |

---

## Design Philosophy

This environment is designed to test:
1. **Calendar reasoning** — overlap detection, timezone arithmetic
2. **Priority judgment** — what's truly urgent vs. noise
3. **Natural language understanding** — reading transcripts, emails
4. **Multi-step planning** — coordinating across constraints
5. **Professional communication** — tone, empathy, conciseness

Unlike game environments, every task reflects situations that a real assistant faces daily.

---

## License
MIT
