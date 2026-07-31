# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stall-restart wrapper that guards each OSMO Run's experiment runner."""

import subprocess
import sys
from pathlib import Path

import pytest

WATCHDOG_SCRIPT = Path(__file__).parents[1] / "evaluation" / "experiment_runner_watchdog.py"


def _run_watchdog(command: list[str], *, watchdog_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run the watchdog over a shell command and return its completed process."""
    return subprocess.run(
        [
            sys.executable,
            str(WATCHDOG_SCRIPT),
            "--poll-interval-seconds",
            "0.2",
            *(watchdog_args or []),
            "--",
            "bash",
            "-c",
            *command,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_propagates_a_clean_exit_without_restarting(exit_code):
    """Return the wrapped command's own exit code, however it exits."""
    result = _run_watchdog(
        [f"echo working; exit {exit_code}"],
        watchdog_args=["--stall-timeout-seconds", "30"],
    )

    assert result.returncode == exit_code
    assert "working" in result.stdout
    assert "Restart" not in result.stdout


def test_restarts_a_stalled_command_and_gives_up_after_max_restarts(tmp_path):
    """Kill a silent command, relaunch it on a cleared output directory, then give up."""
    stale_output = tmp_path / "stale_run_output.txt"
    stale_output.write_text("output from the stalled attempt", encoding="utf-8")

    result = _run_watchdog(
        ["echo starting; sleep 300"],
        watchdog_args=[
            "--stall-timeout-seconds",
            "1",
            "--startup-timeout-seconds",
            "1",
            "--max-restarts",
            "1",
            "--output-directory",
            str(tmp_path),
        ],
    )

    assert result.returncode == 1
    assert "Restart 1/1 after a stall." in result.stdout
    assert "giving up" in result.stdout
    # The relaunched command requires an empty output directory, so the stalled attempt's output goes.
    assert not stale_output.exists()
    assert list(tmp_path.iterdir()) == []


def test_carriage_return_progress_counts_as_liveness():
    """Treat a command that only emits carriage returns as alive, not stalled."""
    # Six 0.5 s progress updates, none of them newline-terminated, over a 1 s stall timeout.
    result = _run_watchdog(
        ["for i in $(seq 6); do printf 'progress %s\\r' \"$i\"; sleep 0.5; done; exit 0"],
        watchdog_args=["--stall-timeout-seconds", "1", "--startup-timeout-seconds", "1"],
    )

    assert result.returncode == 0
    assert "Restart" not in result.stdout
    assert "progress 6" in result.stdout


def test_startup_grace_covers_a_silent_cold_start():
    """Tolerate a long silence before the first output, then hold the command to the stall timeout."""
    result = _run_watchdog(
        ["sleep 2; echo started; exit 0"],
        watchdog_args=["--stall-timeout-seconds", "1", "--startup-timeout-seconds", "30"],
    )

    assert result.returncode == 0
    assert "started" in result.stdout
    assert "Restart" not in result.stdout


def test_rejects_an_empty_command():
    """Fail loudly when no wrapped command is given."""
    result = subprocess.run(
        [sys.executable, str(WATCHDOG_SCRIPT), "--stall-timeout-seconds", "30"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "No command to run" in result.stderr
