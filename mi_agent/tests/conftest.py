"""Shared pytest configuration for the mi_agent test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_llm: runs the case through the real LLM parser; skipped unless "
        "ANTHROPIC_API_KEY is set (CI stays deterministic).")
