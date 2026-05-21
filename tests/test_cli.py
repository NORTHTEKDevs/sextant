"""Smoke tests for the CLI (sextant/__main__.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sextant", *args],
        capture_output=True, text=True, env=env, check=False,
    )


def test_cli_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    assert "cove" in r.stdout
    assert "self_consistency" in r.stdout


@pytest.mark.parametrize("subcmd", ["cove", "self_consistency", "best_of_n"])
def test_cli_subcommand_help(subcmd: str):
    r = _run(subcmd, "--help")
    assert r.returncode == 0


def test_cli_self_consistency_offline_works(monkeypatch):
    import os
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = _run("self_consistency", "What is 2+2?", "--n", "4", "--json", env=env)
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert "answer" in body
    assert body["n"] == 4
    assert "confidence" in body


def test_cli_best_of_n_offline_works():
    import os
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = _run("best_of_n", "test", "--n", "3", "--json", env=env)
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert "answer" in body
    assert "scores" in body
    assert len(body["scores"]) == 3


def test_cli_drift_needs_openai_key():
    import os
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = _run("drift", "hello", env=env)
    assert r.returncode != 0
    assert "OPENAI_API_KEY" in r.stderr
