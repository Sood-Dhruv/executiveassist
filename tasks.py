"""
Task definitions for ExecutiveAssist-Env.

Each task has:
  - description: plain-English summary
  - difficulty: easy / medium / hard
  - context: the data the agent can see (calendar, inbox, notes, etc.)
  - instructions: what the agent must do
  - expected_actions: what a perfect agent would output (used by graders)
  - grading_criteria: partial-credit rubric keys
"""

from typing import Any, Dict

TASKS: Dict[str, Any] = {

    # ──────────────────────────────────────────
    # EASY TASKS
    # ──────────────────────────────────────────

    "schedule_meeting": {
        "description": (
            "Schedule a 1-hour meeting between Alex (the executive) and three guests "
            "on the first mutually available weekday slot this week."
        ),
        "difficulty": "easy",
        "instructions": (
            "Look at the calendars provided. Find the earliest 1-hour slot "
            "that works for ALL attendees (Alex, Jordan, Sam). "
            "Respond with action type 'schedule' and fields: "
            "date, start_time, end_time, attendees, title."
        ),
        "context": {
            "today": "2025-07-14",  # Monday
            "executive": "Alex Chen",
            "meeting_title": "Q3 Planning Sync",
            "duration_minutes": 60,
            "attendees": ["Alex Chen", "Jordan Lee", "Sam Rivera"],
            "calendars": {
                "Alex Chen": [
                    {"date": "2025-07-14", "start": "09:00", "end": "10:00", "title": "Standup"},
                    {"date": "2025-07-14", "start": "14:00", "end": "15:30", "title": "Investor Call"},
                    {"date": "2025-07-15", "start": "10:00", "end": "11:00", "title": "1:1 with HR"},
                ],
                "Jordan Lee": [
                    {"date": "2025-07-14", "start": "09:00", "end": "12:00", "title": "Workshop"},
                    {"date": "2025-07-15", "start": "09:00", "end": "09:30", "title": "Standup"},
                    {"date": "2025-07-15", "start": "14:00", "end": "15:00", "title": "Design Review"},
                ],
                "Sam Rivera": [
                    {"date": "2025-07-14", "start": "13:00", "end": "16:00", "title": "Client Demo"},
                    {"date": "2025-07-15", "start": "11:00", "end": "12:00", "title": "Budget Review"},
                ],
            },
            "working_hours": {"start": "09:00", "end": "18:00"},
        },
        "expected_actions": [
            {
                "type": "schedule",
                "date": "2025-07-15",
                "start_time": "15:00",
                "end_time": "16:00",
                "attendees": ["Alex Chen", "Jordan Lee", "Sam Rivera"],
                "title": "Q3 Planning Sync",
            }
        ],
        "grading_criteria": {
            "correct_date": 0.3,
            "correct_time": 0.4,
            "all_attendees": 0.2,
            "correct_type": 0.1,
        },
    },

    "confirm_slot": {
        "description": (
            "A guest has proposed three meeting time options. "
            "Pick the one that doesn't conflict with Alex's calendar and confirm it."
        ),
        "difficulty": "easy",
        "instructions": (
            "Review the proposed slots and Alex's existing calendar. "
            "Select the first conflict-free slot. "
            "Respond with action type 'schedule' including the chosen slot details."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "proposed_slots": [
                {"date": "2025-07-14", "start": "14:00", "end": "15:00"},
                {"date": "2025-07-15", "start": "10:30", "end": "11:30"},
                {"date": "2025-07-16", "start": "11:00", "end": "12:00"},
            ],
            "calendars": {
                "Alex Chen": [
                    {"date": "2025-07-14", "start": "13:00", "end": "15:00", "title": "Board Prep"},
                    {"date": "2025-07-15", "start": "10:00", "end": "11:00", "title": "1:1"},
                    {"date": "2025-07-16", "start": "09:00", "end": "10:00", "title": "Standup"},
                ],
            },
            "guest": "Morgan Blake",
            "meeting_title": "Partnership Discussion",
        },
        "expected_actions": [
            {
                "type": "schedule",
                "date": "2025-07-16",
                "start_time": "11:00",
                "end_time": "12:00",
                "attendees": ["Alex Chen", "Morgan Blake"],
                "title": "Partnership Discussion",
            }
        ],
        "grading_criteria": {
            "correct_slot_chosen": 0.5,
            "no_conflict": 0.3,
            "correct_type": 0.2,
        },
    },

    "cancel_meeting": {
        "description": (
            "Alex needs to cancel tomorrow's 'Vendor Review' meeting "
            "and send a brief apology note to the attendees."
        ),
        "difficulty": "easy",
        "instructions": (
            "Find the 'Vendor Review' meeting on the calendar. "
            "Issue a cancel action and a reply action with a polite cancellation message. "
            "Actions: first 'cancel', then 'reply'."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "calendars": {
                "Alex Chen": [
                    {"id": "evt-001", "date": "2025-07-14", "start": "09:00", "end": "09:30", "title": "Standup", "attendees": ["Alex Chen"]},
                    {"id": "evt-002", "date": "2025-07-15", "start": "14:00", "end": "15:30", "title": "Vendor Review", "attendees": ["Alex Chen", "vendor@supplycorp.com", "ops@internal.com"]},
                    {"id": "evt-003", "date": "2025-07-16", "start": "10:00", "end": "11:00", "title": "Sprint Review", "attendees": ["Alex Chen", "team@internal.com"]},
                ],
            },
            "reason": "Alex has a family emergency.",
        },
        "expected_actions": [
            {
                "type": "cancel",
                "event_id": "evt-002",
                "reason": "family emergency",
            },
            {
                "type": "reply",
                "to": ["vendor@supplycorp.com", "ops@internal.com"],
                "subject": "Cancelling: Vendor Review",
                "body_contains": ["sorry", "reschedule", "apolog"],
            },
        ],
        "grading_criteria": {
            "correct_event_cancelled": 0.35,
            "reply_sent": 0.25,
            "reply_has_apology": 0.2,
            "reply_mentions_reschedule": 0.2,
        },
    },

    # ──────────────────────────────────────────
    # MEDIUM TASKS
    # ──────────────────────────────────────────

    "inbox_triage": {
        "description": (
            "Alex has 8 unread emails. Triage them into: URGENT (respond today), "
            "IMPORTANT (respond this week), DELEGATE (forward to someone else), "
            "and ARCHIVE (no action needed)."
        ),
        "difficulty": "medium",
        "instructions": (
            "Read the inbox. Categorize each email by its id into one of: "
            "URGENT, IMPORTANT, DELEGATE, ARCHIVE. "
            "Respond with a single 'triage' action with field 'assignments': "
            "a dict mapping email_id to category."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "inbox": [
                {"id": "email-1", "from": "ceo@bigclient.com", "subject": "Contract signing — TODAY by 5pm", "preview": "We need the signed NDA before EOD or the deal is off.", "received": "08:15"},
                {"id": "email-2", "from": "newsletter@techdigest.io", "subject": "Your weekly digest", "preview": "Top 10 AI tools this week...", "received": "07:00"},
                {"id": "email-3", "from": "board@company.com", "subject": "Q3 board deck — need by Friday", "preview": "Please prepare slides for Q3 review by Friday noon.", "received": "09:00"},
                {"id": "email-4", "from": "it@company.com", "subject": "Your laptop is ready", "preview": "Your new laptop has been set up. Please pick it up.", "received": "08:45"},
                {"id": "email-5", "from": "hr@company.com", "subject": "Salary review process — action needed", "preview": "Please submit your team's salary review by July 16.", "received": "09:30"},
                {"id": "email-6", "from": "recruiter@jobsite.com", "subject": "Exciting opportunity for you!", "preview": "We have an amazing role that matches your profile...", "received": "06:00"},
                {"id": "email-7", "from": "ops@company.com", "subject": "Office supplies order", "preview": "Can you approve the $200 office supplies order?", "received": "09:10"},
                {"id": "email-8", "from": "legal@company.com", "subject": "NDA review complete — your sign-off needed", "preview": "Legal has reviewed the NDA. Needs exec sign-off ASAP.", "received": "09:45"},
            ],
        },
        "expected_actions": [
            {
                "type": "triage",
                "assignments": {
                    "email-1": "URGENT",
                    "email-2": "ARCHIVE",
                    "email-3": "IMPORTANT",
                    "email-4": "DELEGATE",
                    "email-5": "IMPORTANT",
                    "email-6": "ARCHIVE",
                    "email-7": "DELEGATE",
                    "email-8": "URGENT",
                },
            }
        ],
        "grading_criteria": {
            "urgent_correct": 0.30,   # email-1, email-8
            "archive_correct": 0.20,  # email-2, email-6
            "important_correct": 0.25, # email-3, email-5
            "delegate_correct": 0.25,  # email-4, email-7
        },
    },

    "reschedule_conflict": {
        "description": (
            "Alex has a calendar conflict on Wednesday: two meetings overlap. "
            "Identify the conflict, decide which meeting to move, and reschedule it "
            "to the next available slot."
        ),
        "difficulty": "medium",
        "instructions": (
            "Find the two conflicting meetings. Decide which one to reschedule "
            "(prefer moving the lower-priority internal meeting over an external client call). "
            "Issue a 'reschedule' action with: event_id, new_date, new_start_time, new_end_time. "
            "Then send a 'reply' to the affected attendees."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "calendars": {
                "Alex Chen": [
                    {"id": "evt-A", "date": "2025-07-16", "start": "10:00", "end": "11:30",
                     "title": "External Client Pitch", "priority": "high",
                     "attendees": ["Alex Chen", "client@bigcorp.com"],
                     "is_external": True},
                    {"id": "evt-B", "date": "2025-07-16", "start": "10:30", "end": "11:30",
                     "title": "Internal Sprint Retro", "priority": "medium",
                     "attendees": ["Alex Chen", "team@internal.com"],
                     "is_external": False},
                    {"id": "evt-C", "date": "2025-07-16", "start": "12:00", "end": "13:00",
                     "title": "Lunch with Mentor", "priority": "low",
                     "attendees": ["Alex Chen", "mentor@alumni.org"],
                     "is_external": True},
                    {"id": "evt-D", "date": "2025-07-16", "start": "14:00", "end": "15:00",
                     "title": "Budget Review", "priority": "medium",
                     "attendees": ["Alex Chen", "finance@internal.com"],
                     "is_external": False},
                ],
            },
            "available_slots_wed": [
                {"start": "08:00", "end": "09:30"},
                {"start": "13:00", "end": "14:00"},  # conflicts with lunch
                {"start": "15:00", "end": "17:00"},
            ],
        },
        "expected_actions": [
            {
                "type": "reschedule",
                "event_id": "evt-B",
                "new_date": "2025-07-16",
                "new_start_time": "15:00",
                "new_end_time": "16:00",
            },
            {
                "type": "reply",
                "to": ["team@internal.com"],
                "body_contains": ["reschedule", "retro", "15:00"],
            },
        ],
        "grading_criteria": {
            "correct_event_rescheduled": 0.30,
            "external_not_moved": 0.20,
            "new_slot_conflict_free": 0.25,
            "notification_sent": 0.25,
        },
    },

    "draft_reply": {
        "description": (
            "Alex received an aggressive email from a client complaining about a delay. "
            "Draft a professional, empathetic reply that acknowledges the issue, "
            "apologizes, and provides a concrete next step."
        ),
        "difficulty": "medium",
        "instructions": (
            "Read the client email in context. Draft a 'reply' action with: "
            "to, subject, body. The body must: (1) acknowledge the delay, "
            "(2) apologize sincerely, (3) provide a specific next step or timeline, "
            "(4) remain professional and not defensive."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "email": {
                "from": "diana.park@megaclient.com",
                "subject": "RE: Project Delivery — This is Unacceptable",
                "body": (
                    "Alex,\n\n"
                    "I am extremely disappointed. We were promised delivery by July 10th "
                    "and it is now July 14th with no update. Our team has been waiting "
                    "and this delay is costing us money. I expect an explanation and "
                    "a firm commitment immediately or we will escalate to your CEO.\n\n"
                    "Diana Park\nVP Operations, MegaClient Corp"
                ),
                "received": "2025-07-14 08:55",
            },
            "internal_notes": (
                "Dev team says delivery will be ready July 17. The delay was caused by "
                "a critical bug found in QA. Fix is 90% complete."
            ),
        },
        "expected_actions": [
            {
                "type": "reply",
                "to": ["diana.park@megaclient.com"],
                "subject_contains": ["Project Delivery", "RE:"],
                "body_contains": ["apologize", "delay", "July 17", "delivery"],
                "body_not_contains": ["not our fault", "impossible", "blame"],
            }
        ],
        "grading_criteria": {
            "acknowledges_delay": 0.20,
            "contains_apology": 0.25,
            "gives_concrete_date": 0.25,
            "professional_tone": 0.20,
            "no_defensive_language": 0.10,
        },
    },

    # ──────────────────────────────────────────
    # HARD TASKS
    # ──────────────────────────────────────────

    "multi_party_schedule": {
        "description": (
            "Schedule a 90-minute strategy meeting between Alex and 4 stakeholders "
            "across 3 time zones this week. Some have hard constraints. "
            "Find a slot and send invites."
        ),
        "difficulty": "hard",
        "instructions": (
            "Review each attendee's calendar and timezone. Find a 90-minute window "
            "that falls within everyone's working hours (9am–6pm local). "
            "Issue a 'schedule' action. Then issue 'reply' actions to each attendee "
            "with a personalized invite showing their local time."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "timezone": "America/New_York (UTC-4)",
            "meeting_title": "Global Strategy Sync",
            "duration_minutes": 90,
            "attendees": [
                {"name": "Alex Chen",     "email": "alex@company.com",       "tz": "America/New_York",    "utc_offset": -4},
                {"name": "Priya Sharma",  "email": "priya@india.co",          "tz": "Asia/Kolkata",        "utc_offset": +5.5},
                {"name": "Lars Eriksson", "email": "lars@sweden.se",           "tz": "Europe/Stockholm",    "utc_offset": +2},
                {"name": "Kenji Tanaka",  "email": "kenji@japan.co.jp",       "tz": "Asia/Tokyo",          "utc_offset": +9},
                {"name": "Sofia Reyes",   "email": "sofia@latam.mx",           "tz": "America/Mexico_City", "utc_offset": -5},
            ],
            "calendars_utc": {
                "alex@company.com":  [
                    {"date": "2025-07-14", "start_utc": "13:00", "end_utc": "14:00"},
                    {"date": "2025-07-15", "start_utc": "14:00", "end_utc": "16:00"},
                ],
                "priya@india.co":    [
                    {"date": "2025-07-14", "start_utc": "04:30", "end_utc": "07:30"},
                    {"date": "2025-07-15", "start_utc": "05:30", "end_utc": "07:00"},
                ],
                "lars@sweden.se":    [
                    {"date": "2025-07-14", "start_utc": "07:00", "end_utc": "09:00"},
                    {"date": "2025-07-15", "start_utc": "06:00", "end_utc": "08:00"},
                ],
                "kenji@tanaka.co.jp": [],
                "sofia@latam.mx":    [
                    {"date": "2025-07-14", "start_utc": "16:00", "end_utc": "18:00"},
                    {"date": "2025-07-15", "start_utc": "15:00", "end_utc": "16:00"},
                ],
            },
            "working_hours_local": {"start": "09:00", "end": "18:00"},
        },
        "expected_actions": [
            {
                "type": "schedule",
                "date": "2025-07-15",
                "start_time_utc": "14:30",
                "end_time_utc": "16:00",
                "attendees": ["alex@company.com", "priya@india.co", "lars@sweden.se", "kenji@tanaka.co.jp", "sofia@latam.mx"],
                "title": "Global Strategy Sync",
            }
        ],
        "grading_criteria": {
            "slot_within_working_hours_all": 0.30,
            "no_conflicts": 0.25,
            "correct_duration": 0.15,
            "invites_sent": 0.15,
            "local_times_in_invites": 0.15,
        },
    },

    "meeting_notes_extraction": {
        "description": (
            "Alex attended a 45-minute product strategy meeting. "
            "Extract: all action items (with owners and due dates), "
            "key decisions made, and open questions. Then schedule any follow-up meetings."
        ),
        "difficulty": "hard",
        "instructions": (
            "Read the meeting transcript carefully. "
            "Issue an 'extract' action with fields: "
            "action_items (list of {task, owner, due_date}), "
            "decisions (list of strings), "
            "open_questions (list of strings). "
            "Then issue a 'reply' to all attendees with a formatted summary."
        ),
        "context": {
            "today": "2025-07-14",
            "executive": "Alex Chen",
            "meeting": {
                "title": "Product Strategy — Q3 Roadmap",
                "date": "2025-07-14",
                "attendees": ["Alex Chen", "Maya Patel (PM)", "Dev Team Lead (Chris)", "Design Lead (Ren)", "Investor Rep (Frank)"],
                "transcript": """
Alex: Let's get started. Maya, what's the status on the mobile feature?

Maya: We're behind. The new onboarding flow won't be ready until July 28th. 
      Chris, can your team prioritize it?

Chris: Yes, I'll have the backend done by July 21st. But we need the final 
       designs from Ren by July 18th or we slip further.

Ren: I can hit July 18th IF we cut the animation scope. Frank, is that okay 
     with investors?

Frank: Investors want the product to feel premium. Don't cut animations. 
       But July 28th is the hard deadline — we have a demo on July 30th.

Alex: Okay. Decision: we ship the mobile onboarding on July 28th, no scope cuts. 
      Ren, full designs by July 18th. Chris, backend by July 21st. 
      Maya, you own the overall delivery and communication to stakeholders.

Maya: What about the web version? We haven't decided.

Alex: Open question — defer web version decision to next week after we see 
      mobile progress. I'll schedule that meeting.

Chris: Also, we still don't have a decision on the third-party analytics SDK.

Alex: Another open question. I need a cost breakdown from Maya before we decide. 
      Maya, send that to me by July 17th.

Frank: One more thing — board wants a brief update on Q3 progress by July 16th.

Alex: I'll handle that. I'll send the board a written update by July 16th.
                """,
            },
        },
        "expected_actions": [
            {
                "type": "extract",
                "action_items": [
                    {"task": "Final mobile designs", "owner": "Ren", "due_date": "2025-07-18"},
                    {"task": "Backend development", "owner": "Chris", "due_date": "2025-07-21"},
                    {"task": "Mobile onboarding delivery + stakeholder comms", "owner": "Maya", "due_date": "2025-07-28"},
                    {"task": "Analytics SDK cost breakdown", "owner": "Maya", "due_date": "2025-07-17"},
                    {"task": "Board Q3 written update", "owner": "Alex Chen", "due_date": "2025-07-16"},
                ],
                "decisions": [
                    "Ship mobile onboarding on July 28th",
                    "No scope cuts to animations",
                ],
                "open_questions": [
                    "Web version launch timeline",
                    "Third-party analytics SDK selection",
                ],
            }
        ],
        "grading_criteria": {
            "action_items_found": 0.30,
            "action_items_have_owners": 0.15,
            "action_items_have_dates": 0.15,
            "decisions_found": 0.20,
            "open_questions_found": 0.20,
        },
    },

    "full_day_plan": {
        "description": (
            "Alex has a chaotic Tuesday: 6 tasks, 4 meetings, 3 urgent emails, "
            "a travel requirement, and limited focus time. "
            "Build an optimized full-day plan that maximizes productivity."
        ),
        "difficulty": "hard",
        "instructions": (
            "Review all inputs (tasks, meetings, emails, constraints). "
            "Issue a 'plan' action with a 'schedule' field: an ordered list of time blocks. "
            "Each block has: start_time, end_time, type (meeting/task/email/travel/break), "
            "title, priority. Ensure: no overlaps, all fixed meetings included, "
            "deep work tasks in morning, emails batched, lunch break present."
        ),
        "context": {
            "today": "2025-07-15",
            "executive": "Alex Chen",
            "fixed_meetings": [
                {"id": "m1", "start": "09:00", "end": "09:30", "title": "Daily Standup", "location": "office"},
                {"id": "m2", "start": "11:00", "end": "12:00", "title": "Investor Call", "location": "remote"},
                {"id": "m3", "start": "14:00", "end": "15:00", "title": "1:1 with Engineering Lead", "location": "office"},
                {"id": "m4", "start": "16:30", "end": "17:00", "title": "Board Prep", "location": "remote"},
            ],
            "tasks": [
                {"id": "t1", "title": "Write Q3 board deck", "estimated_hours": 2, "priority": "high", "requires_focus": True},
                {"id": "t2", "title": "Review legal contract", "estimated_hours": 1, "priority": "high", "requires_focus": True},
                {"id": "t3", "title": "Approve expense reports", "estimated_hours": 0.5, "priority": "low", "requires_focus": False},
                {"id": "t4", "title": "Review PRs on GitHub", "estimated_hours": 0.5, "priority": "medium", "requires_focus": False},
            ],
            "urgent_emails": [
                {"id": "e1", "from": "ceo@bigclient.com", "subject": "Contract signing — today"},
                {"id": "e2", "from": "board@company.com", "subject": "Board deck request"},
                {"id": "e3", "from": "hr@company.com",   "subject": "Salary review deadline today"},
            ],
            "travel": {
                "required": True,
                "reason": "Office visit for afternoon meetings",
                "travel_time_minutes": 30,
                "departure_latest": "13:15",
            },
            "constraints": [
                "Must include 30-min lunch",
                "Deep work before 12pm",
                "Batch emails — no more than 2 email sessions",
                "All fixed meetings must be included unchanged",
            ],
        },
        "expected_actions": [
            {
                "type": "plan",
                "schedule": [
                    {"start_time": "08:00", "end_time": "09:00",  "type": "task",    "title": "Write Q3 board deck (Part 1)"},
                    {"start_time": "09:00", "end_time": "09:30",  "type": "meeting", "title": "Daily Standup"},
                    {"start_time": "09:30", "end_time": "11:00",  "type": "task",    "title": "Write Q3 board deck (Part 2)"},
                    {"start_time": "11:00", "end_time": "12:00",  "type": "meeting", "title": "Investor Call"},
                    {"start_time": "12:00", "end_time": "12:30",  "type": "email",   "title": "Email batch 1: urgent replies"},
                    {"start_time": "12:30", "end_time": "13:00",  "type": "break",   "title": "Lunch"},
                    {"start_time": "13:00", "end_time": "13:30",  "type": "travel",  "title": "Travel to office"},
                    {"start_time": "13:30", "end_time": "14:00",  "type": "task",    "title": "Review legal contract"},
                    {"start_time": "14:00", "end_time": "15:00",  "type": "meeting", "title": "1:1 with Engineering Lead"},
                    {"start_time": "15:00", "end_time": "15:30",  "type": "task",    "title": "Approve expense reports + Review PRs"},
                    {"start_time": "15:30", "end_time": "16:00",  "type": "email",   "title": "Email batch 2: remaining"},
                    {"start_time": "16:30", "end_time": "17:00",  "type": "meeting", "title": "Board Prep"},
                ],
            }
        ],
        "grading_criteria": {
            "all_meetings_included": 0.25,
            "no_time_overlaps": 0.20,
            "lunch_included": 0.10,
            "deep_work_morning": 0.15,
            "travel_block_present": 0.15,
            "email_batched": 0.15,
        },
    },
}
