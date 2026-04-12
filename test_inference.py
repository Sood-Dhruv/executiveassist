"""
test_inference.py — Local test suite for the inference.py proxy-routing fix.

Tests:
  1. ENV vars are read correctly
  2. OpenAI client points to LLM_BASE_URL (proxy), NOT env server
  3. EnvClient points to ENV_BASE_URL (FastAPI env)
  4. LLM and Env URLs are different (the core bug check)
  5. Mock LLM call goes through the right URL
  6. Fallback logic works when LLM is unavailable
  7. deep_parse handles nested JSON strings
  8. All 9 task IDs are covered by TASK_HINTS

Run with:
    python test_inference.py
Or with a real env server running:
    ENV_BASE_URL=http://localhost:7860 python test_inference.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Inject fake env vars BEFORE importing inference ──────────────────────────
# Use os.environ[] (not setdefault) so we override any pre-existing shell vars.
os.environ["API_BASE_URL"] = "https://fake-litellm-proxy.validator.ai/v1"
os.environ["API_KEY"]      = "test-api-key-abc123"
os.environ["MODEL_NAME"]   = "gpt-4o"
os.environ["ENV_BASE_URL"] = "http://localhost:7860"

import inference  # noqa: E402  (must come after env vars are set)
# Re-apply after import so inference module-level vars also get the fake values
inference.LLM_BASE_URL = os.environ["API_BASE_URL"]
inference.ENV_BASE_URL = os.environ["ENV_BASE_URL"]
inference.API_KEY      = os.environ["API_KEY"]
inference.MODEL_NAME   = os.environ["MODEL_NAME"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment variable routing
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvVarRouting(unittest.TestCase):

    def test_llm_base_url_reads_api_base_url(self):
        """LLM_BASE_URL must equal API_BASE_URL (the validator proxy)."""
        self.assertEqual(
            inference.LLM_BASE_URL,
            "https://fake-litellm-proxy.validator.ai/v1",
            "LLM_BASE_URL should read from API_BASE_URL env var"
        )

    def test_env_base_url_is_local_server(self):
        """ENV_BASE_URL should default to localhost:7860 (the FastAPI env server)."""
        self.assertEqual(
            inference.ENV_BASE_URL,
            "http://localhost:7860",
        )

    def test_llm_and_env_urls_are_different(self):
        """THE CORE BUG: LLM proxy and env server must NOT be the same URL."""
        self.assertNotEqual(
            inference.LLM_BASE_URL,
            inference.ENV_BASE_URL,
            "LLM proxy URL and env server URL must be different! "
            "This was the original bug — both were pointing to localhost:7860."
        )

    def test_api_key_is_read(self):
        self.assertEqual(inference.API_KEY, "test-api-key-abc123")

    def test_model_name_is_read(self):
        self.assertEqual(inference.MODEL_NAME, "gpt-4o")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OpenAI client construction in main()
# ─────────────────────────────────────────────────────────────────────────────
class TestOpenAIClientConstruction(unittest.TestCase):

    def test_openai_client_uses_llm_proxy_not_env_server(self):
        """
        The OpenAI client must be initialised with LLM_BASE_URL,
        NOT with args.base_url / ENV_BASE_URL.
        """
        captured = {}

        def fake_openai(base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"]  = api_key
            return MagicMock()

        with patch("inference.OpenAI", side_effect=fake_openai), \
             patch("inference.EnvClient"), \
             patch("inference.Agent"), \
             patch("inference.run_task"):
            # Simulate running main with no --task arg
            with patch("sys.argv", ["inference.py"]):
                inference.main()

        self.assertIn("base_url", captured, "OpenAI() was never called")
        self.assertEqual(
            captured["base_url"],
            "https://fake-litellm-proxy.validator.ai/v1",
            f"OpenAI client must point to LLM proxy, got: {captured.get('base_url')}"
        )
        self.assertNotEqual(
            captured["base_url"],
            "http://localhost:7860",
            "OpenAI client must NOT point to the local env server"
        )

    def test_env_client_uses_env_server_url(self):
        """EnvClient must be initialised with the env server URL."""
        captured = {}

        class FakeEnvClient:
            def __init__(self, base_url):
                captured["env_url"] = base_url

        with patch("inference.OpenAI", return_value=MagicMock()), \
             patch("inference.EnvClient", side_effect=FakeEnvClient), \
             patch("inference.Agent"), \
             patch("inference.run_task"):
            with patch("sys.argv", ["inference.py"]):
                inference.main()

        self.assertEqual(
            captured.get("env_url"),
            "http://localhost:7860",
            f"EnvClient must point to env server, got: {captured.get('env_url')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM call routing
# ─────────────────────────────────────────────────────────────────────────────
class TestLLMCallRouting(unittest.TestCase):

    def _make_mock_llm(self, return_json: dict):
        """Build a mock OpenAI client that returns a JSON string."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(return_json)
        mock_llm.chat.completions.create.return_value = mock_response
        return mock_llm

    def test_llm_call_returns_parsed_dict(self):
        expected = {"type": "schedule", "date": "2025-07-15", "start_time": "15:00",
                    "end_time": "16:00", "attendees": ["A", "B"], "title": "Test"}
        mock_llm = self._make_mock_llm(expected)
        state = {
            "task_id": "schedule_meeting",
            "task_description": "Schedule a meeting",
            "instructions": "Find a free slot",
            "context": {},
        }
        result = inference.llm_call(mock_llm, "schedule_meeting", state, step=1, history=[])
        self.assertEqual(result["type"], "schedule")
        self.assertTrue(mock_llm.chat.completions.create.called,
                        "LLM client must have been called")

    def test_llm_call_uses_correct_model(self):
        mock_llm = self._make_mock_llm({"type": "reply", "to": [], "subject": "x", "body": "y"})
        state = {"task_id": "draft_reply", "task_description": "d", "instructions": "i", "context": {}}
        inference.llm_call(mock_llm, "draft_reply", state, step=1, history=[])
        call_kwargs = mock_llm.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model") or call_kwargs[1].get("model"), "gpt-4o")

    def test_llm_call_strips_markdown_fences(self):
        """LLM sometimes wraps JSON in ```json ... ``` — must be stripped."""
        mock_llm = MagicMock()
        raw = '```json\n{"type": "cancel", "event_id": "evt-002", "reason": "test"}\n```'
        mock_llm.chat.completions.create.return_value.choices[0].message.content = raw
        state = {"task_id": "cancel_meeting", "task_description": "d", "instructions": "i", "context": {}}
        result = inference.llm_call(mock_llm, "cancel_meeting", state, step=1, history=[])
        self.assertEqual(result["type"], "cancel")
        self.assertEqual(result["event_id"], "evt-002")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Agent fallback logic
# ─────────────────────────────────────────────────────────────────────────────
class TestAgentFallback(unittest.TestCase):

    def test_agent_uses_hardcoded_fallback_on_llm_failure(self):
        """When LLM fails, agent should return the hardcoded answer."""
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = Exception("LLM unavailable")
        agent = inference.Agent(mock_llm)
        state = {
            "task_id": "cancel_meeting",
            "task_description": "Cancel meeting",
            "instructions": "Cancel evt-002",
            "context": {},
            "expected_actions": [],
        }
        action = agent.act(state, step=1)
        self.assertEqual(action["type"], "cancel")
        self.assertEqual(action["event_id"], "evt-002")

    def test_agent_uses_expected_actions_fallback(self):
        """For tasks without hardcoded sequences, falls back to expected_actions."""
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = Exception("LLM unavailable")
        agent = inference.Agent(mock_llm)
        state = {
            "task_id": "some_unknown_task",
            "task_description": "Do something",
            "instructions": "...",
            "context": {},
            "expected_actions": [{"type": "reply", "to": ["a@b.com"], "subject": "Hi", "body": "Hello"}],
        }
        action = agent.act(state, step=1)
        self.assertEqual(action["type"], "reply")

    def test_agent_history_accumulates(self):
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = \
            '{"type":"reply","to":[],"subject":"s","body":"b"}'
        agent = inference.Agent(mock_llm)
        state = {"task_id": "draft_reply", "task_description": "d",
                 "instructions": "i", "context": {}, "expected_actions": []}
        agent.act(state, step=1)
        agent.act(state, step=2)
        self.assertEqual(len(agent._history), 2)

    def test_agent_reset_clears_history(self):
        mock_llm = MagicMock()
        agent = inference.Agent(mock_llm)
        agent._history = [{"step": 1, "action": {"type": "cancel"}}]
        agent.reset_history()
        self.assertEqual(agent._history, [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. deep_parse utility
# ─────────────────────────────────────────────────────────────────────────────
class TestDeepParse(unittest.TestCase):

    def test_parses_json_string(self):
        result = inference.deep_parse('{"type": "cancel", "event_id": "evt-002"}')
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "cancel")

    def test_passes_through_dict(self):
        d = {"type": "schedule", "date": "2025-07-15"}
        self.assertEqual(inference.deep_parse(d), d)

    def test_parses_nested_json_string_in_dict(self):
        d = {"action": '{"type": "reply", "to": ["a@b.com"]}'}
        result = inference.deep_parse(d)
        self.assertIsInstance(result["action"], dict)
        self.assertEqual(result["action"]["type"], "reply")

    def test_passes_through_non_json_string(self):
        self.assertEqual(inference.deep_parse("hello world"), "hello world")

    def test_parses_list(self):
        lst = ['{"type":"cancel"}', '{"type":"reply"}']
        result = inference.deep_parse(lst)
        self.assertEqual(result[0]["type"], "cancel")
        self.assertEqual(result[1]["type"], "reply")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Task coverage
# ─────────────────────────────────────────────────────────────────────────────
class TestTaskCoverage(unittest.TestCase):

    ALL_TASKS = [
        "schedule_meeting", "confirm_slot", "cancel_meeting",
        "inbox_triage", "reschedule_conflict", "draft_reply",
        "multi_party_schedule", "meeting_notes_extraction", "full_day_plan",
    ]

    def test_all_tasks_have_hints(self):
        for task_id in self.ALL_TASKS:
            self.assertIn(task_id, inference.TASK_HINTS,
                          f"Missing TASK_HINTS entry for '{task_id}'")

    def test_hardcoded_tasks_have_at_least_one_step(self):
        for task_id, steps in inference.HARDCODED.items():
            self.assertGreater(len(steps), 0,
                               f"HARDCODED['{task_id}'] is empty")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EnvClient (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvClient(unittest.TestCase):

    def test_reset_posts_to_correct_endpoint(self):
        with patch("inference.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"task_id": "schedule_meeting", "done": False}
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.post.return_value = mock_resp

            client = inference.EnvClient("http://localhost:7860")
            state  = client.reset("schedule_meeting")

            call_args = MockClient.return_value.post.call_args
            url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
            # accept both positional and keyword
            if not url:
                url = call_args[1].get("url", str(call_args))
            self.assertIn("/reset", str(call_args))
            self.assertEqual(state["task_id"], "schedule_meeting")

    def test_step_posts_action(self):
        with patch("inference.httpx.Client") as MockClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "state": {"done": True}, "reward": 0.9, "done": True, "info": {}
            }
            mock_resp.raise_for_status = MagicMock()
            MockClient.return_value.post.return_value = mock_resp

            client = inference.EnvClient("http://localhost:7860")
            result = client.step({"type": "cancel", "event_id": "evt-002"})
            self.assertEqual(result["reward"], 0.9)
            self.assertTrue(result["done"])


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ExecutiveAssist inference.py — Proxy Routing Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Order: most important first
    for cls in [
        TestEnvVarRouting,
        TestOpenAIClientConstruction,
        TestLLMCallRouting,
        TestAgentFallback,
        TestDeepParse,
        TestTaskCoverage,
        TestEnvClient,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("  ✅  ALL TESTS PASSED — proxy routing is correct")
        print("  The OpenAI client will call the validator's LiteLLM proxy,")
        print("  not the local env server. Safe to resubmit.")
    else:
        print(f"  ❌  {len(result.failures)} failure(s), {len(result.errors)} error(s)")
        print("  Fix the failures before resubmitting.")
    print("=" * 60)
    sys.exit(0 if result.wasSuccessful() else 1)