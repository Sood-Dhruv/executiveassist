"""
Graders for ExecutiveAssist-Env.

Each grader returns (reward: float, feedback: str, done: bool).
Rewards are in [0.0, 1.0] and support partial credit.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, time
import re


def grade_action(
    task_id: str,
    task: Dict,
    action: Dict,
    step: int,
    history: List[Dict],
) -> Tuple[float, str, bool]:
    """
    Route to the correct grader based on task_id.
    Returns (reward, feedback, done).
    """
    graders = {
        "schedule_meeting":        grade_schedule_meeting,
        "confirm_slot":            grade_confirm_slot,
        "cancel_meeting":          grade_cancel_meeting,
        "inbox_triage":            grade_inbox_triage,
        "reschedule_conflict":     grade_reschedule_conflict,
        "draft_reply":             grade_draft_reply,
        "multi_party_schedule":    grade_multi_party_schedule,
        "meeting_notes_extraction": grade_meeting_notes_extraction,
        "full_day_plan":           grade_full_day_plan,
    }

    grader = graders.get(task_id)
    if grader is None:
        return 0.0, f"Unknown task '{task_id}'.", True

    try:
        return grader(task, action, step, history)
    except Exception as e:
        return 0.0, f"Grader error: {str(e)}", False


# ──────────────────────────────────────────────────────────────────────────────
# EASY TASK GRADERS
# ──────────────────────────────────────────────────────────────────────────────

def grade_schedule_meeting(task, action, step, history):
    expected = task["expected_actions"][0]
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    # Must be a schedule action
    if action.get("type") != "schedule":
        return 0.0, "Expected action type 'schedule'.", False

    score += criteria["correct_type"]
    notes.append("✓ Correct action type")

    # Check date
    if action.get("date") == expected["date"]:
        score += criteria["correct_date"]
        notes.append(f"✓ Correct date ({expected['date']})")
    else:
        notes.append(f"✗ Wrong date. Got '{action.get('date')}', expected '{expected['date']}'")

    # Check time (allow ±30 min flexibility for start time)
    act_start = action.get("start_time", "")
    exp_start = expected["start_time"]
    if _times_close(act_start, exp_start, tolerance_min=30):
        score += criteria["correct_time"]
        notes.append(f"✓ Correct start time (expected ~{exp_start})")
    else:
        notes.append(f"✗ Start time off. Got '{act_start}', expected ~'{exp_start}'")

    # Check attendees
    action_attendees = set(a.lower() for a in action.get("attendees", []))
    expected_attendees = set(a.lower() for a in expected["attendees"])
    overlap = len(action_attendees & expected_attendees)
    if overlap == len(expected_attendees):
        score += criteria["all_attendees"]
        notes.append("✓ All attendees included")
    elif overlap > 0:
        partial = criteria["all_attendees"] * (overlap / len(expected_attendees))
        score += partial
        notes.append(f"~ Partial attendees ({overlap}/{len(expected_attendees)})")
    else:
        notes.append("✗ No matching attendees")

    done = score >= 0.7
    feedback = "\n".join(notes) + f"\n\nScore: {score:.2f}"
    return round(score, 2), feedback, done


def grade_confirm_slot(task, action, step, history):
    expected = task["expected_actions"][0]
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    if action.get("type") != "schedule":
        return 0.0, "Expected action type 'schedule'.", False

    # Check if chosen slot is conflict-free (slot 3: July 16, 11:00-12:00)
    chosen_date = action.get("date", "")
    chosen_start = action.get("start_time", "")

    no_conflict = _check_no_conflict(
        chosen_date, chosen_start, action.get("end_time", ""),
        task["context"]["calendars"].get("Alex Chen", [])
    )

    if chosen_date == expected["date"] and _times_close(chosen_start, expected["start_time"], 15):
        score += criteria["correct_slot_chosen"]
        notes.append("✓ Correct conflict-free slot chosen")
    elif no_conflict:
        score += criteria["correct_slot_chosen"] * 0.5
        notes.append("~ A valid conflict-free slot chosen (not optimal)")
    else:
        notes.append("✗ Chosen slot has a conflict")

    if no_conflict:
        score += criteria["no_conflict"]
        notes.append("✓ No calendar conflict")
    else:
        notes.append("✗ Slot conflicts with existing calendar")

    score += criteria["correct_type"]
    notes.append("✓ Correct action type")

    done = score >= 0.7
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


def grade_cancel_meeting(task, action, step, history):
    expected_actions = task["expected_actions"]
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    # Collect all actions submitted so far including this one
    all_actions = [h["action"] for h in history] + [action]

    cancel_actions = [a for a in all_actions if a.get("type") == "cancel"]
    reply_actions  = [a for a in all_actions if a.get("type") == "reply"]

    # Grade cancel
    if cancel_actions:
        best_cancel = cancel_actions[-1]
        if best_cancel.get("event_id") == "evt-002":
            score += criteria["correct_event_cancelled"]
            notes.append("✓ Correct event cancelled (evt-002: Vendor Review)")
        else:
            notes.append(f"✗ Wrong event cancelled. Got '{best_cancel.get('event_id')}'")
    else:
        notes.append("✗ No cancel action found")

    # Grade reply
    if reply_actions:
        best_reply = reply_actions[-1]
        score += criteria["reply_sent"]
        notes.append("✓ Reply action sent")
        body = (best_reply.get("body") or "").lower()
        if any(w in body for w in ["sorry", "apologize", "apology", "regret"]):
            score += criteria["reply_has_apology"]
            notes.append("✓ Reply contains apology")
        else:
            notes.append("✗ Reply lacks apology language")
        if any(w in body for w in ["reschedule", "rebook", "another time", "new time"]):
            score += criteria["reply_mentions_reschedule"]
            notes.append("✓ Reply mentions rescheduling")
        else:
            notes.append("✗ Reply doesn't mention rescheduling")
    else:
        notes.append("~ No reply action yet (will check on next step)")

    done = len(cancel_actions) > 0 and len(reply_actions) > 0 and score >= 0.6
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


# ──────────────────────────────────────────────────────────────────────────────
# MEDIUM TASK GRADERS
# ──────────────────────────────────────────────────────────────────────────────

def grade_inbox_triage(task, action, step, history):
    if action.get("type") != "triage":
        return 0.0, "Expected action type 'triage'.", False

    expected = task["expected_actions"][0]["assignments"]
    given    = action.get("assignments", {})
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    # URGENT: email-1, email-8
    urgent_expected = {"email-1", "email-8"}
    urgent_given    = {eid for eid, cat in given.items() if cat == "URGENT"}
    u_correct = len(urgent_expected & urgent_given) / len(urgent_expected)
    score += u_correct * criteria["urgent_correct"]
    notes.append(f"{'✓' if u_correct == 1.0 else '~'} Urgent ({u_correct*100:.0f}% correct)")

    # ARCHIVE: email-2, email-6
    archive_expected = {"email-2", "email-6"}
    archive_given    = {eid for eid, cat in given.items() if cat == "ARCHIVE"}
    a_correct = len(archive_expected & archive_given) / len(archive_expected)
    score += a_correct * criteria["archive_correct"]
    notes.append(f"{'✓' if a_correct == 1.0 else '~'} Archive ({a_correct*100:.0f}% correct)")

    # IMPORTANT: email-3, email-5
    imp_expected = {"email-3", "email-5"}
    imp_given    = {eid for eid, cat in given.items() if cat == "IMPORTANT"}
    i_correct = len(imp_expected & imp_given) / len(imp_expected)
    score += i_correct * criteria["important_correct"]
    notes.append(f"{'✓' if i_correct == 1.0 else '~'} Important ({i_correct*100:.0f}% correct)")

    # DELEGATE: email-4, email-7
    del_expected = {"email-4", "email-7"}
    del_given    = {eid for eid, cat in given.items() if cat == "DELEGATE"}
    d_correct = len(del_expected & del_given) / len(del_expected)
    score += d_correct * criteria["delegate_correct"]
    notes.append(f"{'✓' if d_correct == 1.0 else '~'} Delegate ({d_correct*100:.0f}% correct)")

    done = score >= 0.75
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


def grade_reschedule_conflict(task, action, step, history):
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    all_actions = [h["action"] for h in history] + [action]
    reschedule_actions = [a for a in all_actions if a.get("type") == "reschedule"]
    reply_actions = [a for a in all_actions if a.get("type") == "reply"]

    if reschedule_actions:
        r = reschedule_actions[-1]
        event_moved = r.get("event_id", "")

        # Correct: move evt-B (internal), not evt-A (external client)
        if event_moved == "evt-B":
            score += criteria["correct_event_rescheduled"]
            notes.append("✓ Correct event rescheduled (internal sprint retro)")
        elif event_moved == "evt-A":
            notes.append("✗ Wrong: should not move external client pitch")
        else:
            notes.append(f"✗ Unknown event moved: {event_moved}")

        # External not moved
        if event_moved != "evt-A":
            score += criteria["external_not_moved"]
            notes.append("✓ External client meeting preserved")

        # New slot conflict-free
        new_start = r.get("new_start_time", "")
        new_end   = r.get("new_end_time", "")
        new_date  = r.get("new_date", "2025-07-16")
        if new_start and _check_no_conflict(new_date, new_start, new_end,
                                            [e for e in task["context"]["calendars"]["Alex Chen"]
                                             if e["id"] != event_moved]):
            score += criteria["new_slot_conflict_free"]
            notes.append("✓ New slot is conflict-free")
        else:
            notes.append("~ Could not verify new slot is conflict-free")
    else:
        notes.append("✗ No reschedule action found")

    if reply_actions:
        score += criteria["notification_sent"]
        notes.append("✓ Notification reply sent")
    else:
        notes.append("~ No reply/notification sent yet")

    done = len(reschedule_actions) > 0 and score >= 0.65
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


def grade_draft_reply(task, action, step, history):
    if action.get("type") != "reply":
        return 0.0, "Expected action type 'reply'.", False

    criteria = task["grading_criteria"]
    score = 0.0
    notes = []
    body = (action.get("body") or "").lower()
    subject = (action.get("subject") or "").lower()

    # Acknowledges delay
    if any(w in body for w in ["delay", "late", "behind schedule", "missed", "overdue"]):
        score += criteria["acknowledges_delay"]
        notes.append("✓ Acknowledges the delay")
    else:
        notes.append("✗ Does not acknowledge the delay")

    # Contains apology
    if any(w in body for w in ["apologize", "sorry", "sincerely regret", "apologies"]):
        score += criteria["contains_apology"]
        notes.append("✓ Contains genuine apology")
    else:
        notes.append("✗ Missing apology")

    # Gives concrete date/next step — flexible matching for any reasonable date/timeline expression
    import re as _re
    concrete_date_patterns = [
        r"july\s*1[5-9]", r"jul\s*1[5-9]",           # july 15–19
        r"17th|16th|18th|19th|15th",                   # ordinals
        r"\bby\s+(end|eod|monday|tuesday|wednesday|thursday|friday|next)",
        r"within\s+\d+\s+(day|hour|business)",         # within N days/hours
        r"deliver(y|ed|ing)?\s+(by|on|before)",        # delivery by/on
        r"(complete|ready|shipped|resolved)\s+(by|on|before)",
        r"commit\s+to",                                 # commit to a date
        r"\d{4}-\d{2}-\d{2}",                          # ISO date
        r"(monday|tuesday|wednesday|thursday|friday)\s+(morning|afternoon|eod|by)",
        r"next\s+(week|monday|business day)",
    ]
    has_concrete = any(_re.search(p, body) for p in concrete_date_patterns)
    if has_concrete:
        score += criteria["gives_concrete_date"]
        notes.append("✓ Provides concrete next step / date")
    else:
        notes.append("✗ No concrete timeline given (mention a specific date or commitment)")

    # Professional tone (no ALL CAPS rants, reasonable length)
    defensive_words = ["not our fault", "impossible", "you should have", "blame", "unfair"]
    if not any(w in body for w in defensive_words):
        score += criteria["professional_tone"]
        notes.append("✓ Professional tone maintained")
    else:
        notes.append("✗ Contains defensive language")

    # No defensive language (separate check)
    if not any(w in body for w in defensive_words):
        score += criteria["no_defensive_language"]
        notes.append("✓ No defensive language")

    done = score >= 0.70
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


# ──────────────────────────────────────────────────────────────────────────────
# HARD TASK GRADERS
# ──────────────────────────────────────────────────────────────────────────────

def grade_multi_party_schedule(task, action, step, history):
    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    all_actions = [h["action"] for h in history] + [action]
    schedule_actions = [a for a in all_actions if a.get("type") == "schedule"]
    reply_actions    = [a for a in all_actions if a.get("type") == "reply"]

    if not schedule_actions:
        return 0.0, "No schedule action found.", False

    sched = schedule_actions[-1]
    start_utc = sched.get("start_time_utc") or sched.get("start_time", "")
    end_utc   = sched.get("end_time_utc")   or sched.get("end_time", "")
    date      = sched.get("date", "")

    attendees_ctx = task["context"]["attendees"]

    # Check working hours for all attendees — award partial credit for compromise slots
    in_hours_count = 0
    for att in attendees_ctx:
        offset = att["utc_offset"]
        local_start = _add_hours_to_time(start_utc, offset)
        local_end   = _add_hours_to_time(end_utc, offset)
        if _time_gte(local_start, "07:00") and _time_lte(local_end, "20:00"):
            # Within extended reasonable range (7am–8pm local)
            in_hours_count += 1
        if not (_time_gte(local_start, "09:00") and _time_lte(local_end, "18:00")):
            notes.append(f"~ {att['name']}: local time {local_start}–{local_end} (outside strict 9-6)")

    hours_ratio = in_hours_count / len(attendees_ctx)
    if hours_ratio == 1.0:
        score += criteria["slot_within_working_hours_all"]
        notes.append("✓ Slot within working hours for ALL attendees")
    elif hours_ratio >= 0.6:
        score += criteria["slot_within_working_hours_all"] * hours_ratio
        notes.append(f"~ Best-compromise slot: {in_hours_count}/{len(attendees_ctx)} attendees within reasonable hours")

    # Check 90-min duration
    if start_utc and end_utc:
        start_m = _time_to_minutes(start_utc)
        end_m   = _time_to_minutes(end_utc)
        dur = end_m - start_m
        if abs(dur - 90) <= 5:
            score += criteria["correct_duration"]
            notes.append("✓ Correct 90-minute duration")
        else:
            notes.append(f"✗ Duration is {dur} min, expected 90")
    else:
        notes.append("~ Could not determine duration (missing start/end UTC)")

    # No conflicts
    calendars = task["context"]["calendars_utc"]
    conflict = False
    for att in attendees_ctx:
        cal = calendars.get(att["email"], [])
        if not _check_no_conflict(date, start_utc, end_utc, cal, utc=True):
            conflict = True
            notes.append(f"✗ Conflict for {att['name']}")
    if not conflict:
        score += criteria["no_conflicts"]
        notes.append("✓ No scheduling conflicts")

    # Invites sent
    if len(reply_actions) >= 3:
        score += criteria["invites_sent"]
        notes.append("✓ Invites sent to attendees")
    elif len(reply_actions) > 0:
        score += criteria["invites_sent"] * 0.5
        notes.append("~ Some invites sent")
    else:
        notes.append("✗ No invites sent")

    # Local times in invites
    bodies_combined = " ".join((a.get("body") or "") for a in reply_actions).lower()
    tz_words = ["local time", "your time", "utc", "+5", "+9", "+2", "-5", "-4"]
    if any(w in bodies_combined for w in tz_words):
        score += criteria["local_times_in_invites"]
        notes.append("✓ Local times referenced in invites")
    else:
        notes.append("~ Local times not found in invite bodies")

    done = score >= 0.70
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


def grade_meeting_notes_extraction(task, action, step, history):
    all_actions = [h["action"] for h in history] + [action]
    extract_actions = [a for a in all_actions if a.get("type") == "extract"]
    reply_actions   = [a for a in all_actions if a.get("type") == "reply"]

    criteria = task["grading_criteria"]
    score = 0.0
    notes = []

    if not extract_actions:
        if action.get("type") == "reply":
            notes.append("~ Reply sent but no extract action yet")
            return 0.0, "\n".join(notes), False
        return 0.0, "Expected action type 'extract'.", False

    ext = extract_actions[-1]
    action_items   = ext.get("action_items", [])
    decisions      = ext.get("decisions", [])
    open_questions = ext.get("open_questions", [])

    expected_items = task["expected_actions"][0]["action_items"]
    expected_decisions = task["expected_actions"][0]["decisions"]
    expected_questions = task["expected_actions"][0]["open_questions"]

    # Score action items found (by task keyword matching)
    keywords_per_item = [
        ["design", "ren", "mobile"],
        ["backend", "chris", "develop"],
        ["onboarding", "maya", "delivery"],
        ["analytics", "sdk", "cost", "maya"],
        ["board", "update", "alex", "q3"],
    ]

    items_found = 0
    has_owners = 0
    has_dates = 0
    items_combined = " ".join(str(i).lower() for i in action_items)

    for kws in keywords_per_item:
        if any(kw in items_combined for kw in kws):
            items_found += 1

    for item in action_items:
        if isinstance(item, dict):
            if item.get("owner") or item.get("assigned_to"):
                has_owners += 1
            if item.get("due_date") or item.get("deadline") or item.get("date"):
                has_dates += 1

    items_ratio = items_found / len(keywords_per_item)
    score += items_ratio * criteria["action_items_found"]
    notes.append(f"{'✓' if items_ratio==1 else '~'} Action items: {items_found}/{len(keywords_per_item)} found")

    if action_items:
        owner_ratio = has_owners / len(action_items)
        if owner_ratio >= 0.6:
            score += criteria["action_items_have_owners"]
            notes.append("✓ Action items have owners")
        else:
            notes.append(f"~ Only {has_owners}/{len(action_items)} items have owners")

        date_ratio = has_dates / len(action_items)
        if date_ratio >= 0.6:
            score += criteria["action_items_have_dates"]
            notes.append("✓ Action items have due dates")
        else:
            notes.append(f"~ Only {has_dates}/{len(action_items)} items have dates")

    # Score decisions
    dec_combined = " ".join(str(d).lower() for d in decisions)
    dec_keywords = [["july 28", "28th", "mobile", "ship"], ["animation", "scope", "premium", "no cut"]]
    dec_found = sum(1 for kws in dec_keywords if any(kw in dec_combined for kw in kws))
    dec_ratio = dec_found / len(dec_keywords)
    score += dec_ratio * criteria["decisions_found"]
    notes.append(f"{'✓' if dec_ratio==1 else '~'} Decisions: {dec_found}/{len(dec_keywords)} found")

    # Score open questions
    oq_combined = " ".join(str(q).lower() for q in open_questions)
    oq_keywords = [["web", "version", "launch"], ["analytics", "sdk", "third-party"]]
    oq_found = sum(1 for kws in oq_keywords if any(kw in oq_combined for kw in kws))
    oq_ratio = oq_found / len(oq_keywords)
    score += oq_ratio * criteria["open_questions_found"]
    notes.append(f"{'✓' if oq_ratio==1 else '~'} Open questions: {oq_found}/{len(oq_keywords)} found")

    done = score >= 0.70
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


def grade_full_day_plan(task, action, step, history):
    all_actions = [h["action"] for h in history] + [action]
    plan_actions = [a for a in all_actions if a.get("type") == "plan"]

    if not plan_actions:
        return 0.0, "Expected action type 'plan'.", False

    criteria = task["grading_criteria"]
    plan = plan_actions[-1]
    schedule = plan.get("schedule", [])
    score = 0.0
    notes = []

    # All fixed meetings included
    fixed_ids = {"Daily Standup", "Investor Call", "1:1 with Engineering Lead", "Board Prep"}
    titles_in_plan = {b.get("title", "").lower() for b in schedule}
    fixed_found = sum(
        1 for f in fixed_ids
        if any(f.lower() in t for t in titles_in_plan)
    )
    meetings_ratio = fixed_found / len(fixed_ids)
    score += meetings_ratio * criteria["all_meetings_included"]
    notes.append(f"{'✓' if meetings_ratio==1 else '~'} Fixed meetings: {fixed_found}/{len(fixed_ids)} included")

    # No time overlaps
    no_overlap = _check_schedule_no_overlaps(schedule)
    if no_overlap:
        score += criteria["no_time_overlaps"]
        notes.append("✓ No time overlaps in plan")
    else:
        notes.append("✗ Time overlaps detected in plan")

    # Lunch included
    has_lunch = any(
        b.get("type") == "break" or "lunch" in b.get("title", "").lower()
        for b in schedule
    )
    if has_lunch:
        score += criteria["lunch_included"]
        notes.append("✓ Lunch/break included")
    else:
        notes.append("✗ No lunch break in plan")

    # Deep work in morning (before 12:00)
    morning_blocks = [
        b for b in schedule
        if b.get("type") == "task" and b.get("requires_focus", True)
        and _time_to_minutes(b.get("end_time", "12:00")) <= _time_to_minutes("12:00")
    ]
    deep_work_blocks = [b for b in schedule if "board deck" in b.get("title","").lower()
                        or b.get("type") == "task"]
    any_deep_work_morning = any(
        _time_to_minutes(b.get("end_time", "13:00")) <= _time_to_minutes("12:00")
        for b in deep_work_blocks
    )
    if any_deep_work_morning:
        score += criteria["deep_work_morning"]
        notes.append("✓ Deep work scheduled in morning")
    else:
        notes.append("✗ No deep work in morning hours")

    # Travel block present
    has_travel = any(b.get("type") == "travel" or "travel" in b.get("title","").lower()
                     for b in schedule)
    if has_travel:
        score += criteria["travel_block_present"]
        notes.append("✓ Travel block included")
    else:
        notes.append("✗ No travel block in plan")

    # Email batched (≤ 2 email sessions)
    email_blocks = [b for b in schedule if b.get("type") == "email" or "email" in b.get("title","").lower()]
    if 1 <= len(email_blocks) <= 2:
        score += criteria["email_batched"]
        notes.append(f"✓ Emails batched ({len(email_blocks)} session(s))")
    elif len(email_blocks) > 2:
        notes.append(f"✗ Too many email sessions ({len(email_blocks)}), should be ≤2")
    else:
        notes.append("✗ No email sessions in plan")

    done = score >= 0.70
    return round(score, 2), "\n".join(notes) + f"\n\nScore: {score:.2f}", done


# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _time_to_minutes(t: str) -> int:
    """Convert 'HH:MM' to total minutes from midnight."""
    try:
        parts = str(t).strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        return h * 60 + m
    except Exception:
        return 0


def _times_close(t1: str, t2: str, tolerance_min: int = 15) -> bool:
    """Check if two time strings are within tolerance_min minutes of each other."""
    return abs(_time_to_minutes(t1) - _time_to_minutes(t2)) <= tolerance_min


def _time_gte(t: str, ref: str) -> bool:
    return _time_to_minutes(t) >= _time_to_minutes(ref)


def _time_lte(t: str, ref: str) -> bool:
    return _time_to_minutes(t) <= _time_to_minutes(ref)


def _add_hours_to_time(t: str, hours: float) -> str:
    """Add decimal hours offset to HH:MM string."""
    mins = _time_to_minutes(t) + int(hours * 60)
    mins = mins % (24 * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _check_no_conflict(date: str, start: str, end: str, calendar: list, utc: bool = False) -> bool:
    """
    Returns True if [start, end) on `date` does not overlap any event in calendar.
    """
    start_m = _time_to_minutes(start)
    end_m   = _time_to_minutes(end)
    if end_m <= start_m:
        end_m += 24 * 60  # cross-midnight

    for event in calendar:
        if event.get("date") != date:
            continue
        e_start = _time_to_minutes(event.get("start_utc" if utc else "start", "00:00"))
        e_end   = _time_to_minutes(event.get("end_utc"   if utc else "end",   "00:00"))
        # Overlap check
        if start_m < e_end and end_m > e_start:
            return False
    return True


def _check_schedule_no_overlaps(schedule: list) -> bool:
    """Check that no two time blocks in the plan overlap."""
    sorted_blocks = sorted(schedule, key=lambda b: _time_to_minutes(b.get("start_time", "00:00")))
    for i in range(len(sorted_blocks) - 1):
        this_end  = _time_to_minutes(sorted_blocks[i].get("end_time", "00:00"))
        next_start = _time_to_minutes(sorted_blocks[i+1].get("start_time", "00:00"))
        if this_end > next_start:
            return False
    return True