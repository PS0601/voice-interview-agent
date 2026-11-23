"""Smoke tests for agent.py configuration (no LiveKit connection needed)."""
import importlib


def test_agent_module_imports_and_defines_server():
    agent = importlib.import_module("agent")
    assert agent.server is not None
    assert "interviewer" in agent.InterviewAgent().instructions.lower()


def test_automatic_dispatch_and_no_avatar_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_NAME", raising=False)
    monkeypatch.delenv("BEY_AVATAR_ID", raising=False)
    import agent
    agent = importlib.reload(agent)
    # load_dotenv may supply values from a developer's .env.local; only check the defaults when unset there
    import os
    if "AGENT_NAME" not in os.environ:
        assert agent.AGENT_NAME == ""
    if "BEY_AVATAR_ID" not in os.environ:
        assert agent.BEY_AVATAR_ID == ""
