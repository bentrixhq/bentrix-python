"""**``python -m bentrix``**: the CI tape written, a strategy backtested on it and the battery
run on it, each one JSON object and an exit code — the commands a CI job runs."""

import json

import pytest

pytest.importorskip("bentrix._native")

from bentrix.__main__ import main  # noqa: E402
from bentrix.backtest import Strategy  # noqa: E402


class Hold(Strategy):
    """Buys at the first candle and holds."""

    def on_bar(self, bar, ctx):
        if not ctx.book().get("positions"):
            ctx.buy(0.01)


def _out(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_tape_a_backtest_and_the_battery_run_headless(tmp_path, capsys):
    """The tape is written and named; a strategy given as ``module:Class`` runs on it to its bar limit and
    holds its position; the battery runs to its end with no check failed (exit 0); a missing argument
    and a strategy that does not import exit 2.

    Mutation (2026-11-01): a usage error exiting 0 (``return 0`` for argparse's exit) — red,
    the missing ``--strategy`` did not exit 2. The battery's own pass and fail are the Rust walk's."""
    assert main(["tape", str(tmp_path / "t"), "--last", "2024-03-07"]) == 0
    tape = _out(capsys)
    assert tape["ran"] and tape["workspace"].endswith("ws")
    args = ["--workspace", tape["workspace"], "--store", tape["store"]]
    assert main(["backtest", *args, "--strategy", "test_cli:Hold", "--max-bars", "5"]) == 0
    bt = _out(capsys)
    assert bt["bars"] == 5 and len(bt["book"]["positions"]) == 1, bt
    assert main(["battery", *args]) == 0
    battery = _out(capsys)
    assert len(battery["checks"]) == 16 and battery["failed_checks"] == 0, battery
    assert main(["backtest", *args]) == 2
    assert main(["backtest", *args, "--strategy", "no_such_module:Nothing"]) == 2
