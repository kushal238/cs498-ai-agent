"""
pytest configuration for the benchmark test suite.

Marks:
    integration — tests that make live network calls to external APIs.
                  Run with: pytest -m integration
                  Skip with: pytest -m "not integration"
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring network access",
    )
