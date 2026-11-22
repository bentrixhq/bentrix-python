"""**``python -m bentrix``: BENTRIX headless, for a terminal and for CI**.

    python -m bentrix tape DIR [--first 2024-03-04] [--last 2024-03-08] [--symbol XAUUSD]
    python -m bentrix backtest --workspace WS --store STORE --strategy module:Class [--bench URI]
                              [--interval H1] [--max-bars N]
    python -m bentrix battery --workspace WS --store STORE [--bench URI]

Every answer is one JSON object on standard output. **Exit codes**: ``0`` done; ``1`` done and a check
failed (``battery``); ``2`` it could not run (an argument, a workspace held by a window, a bench that
is not there). The tape is the CI tape (``bentrix._native.make_tape``): deterministic and nobody's
prices, so a job anywhere gets the same trades.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys


def _local(args):
    from bentrix.local import Local

    return Local(workspace=args.workspace, store=args.store)


def _bench(local, given):
    if given:
        return given
    labs = local.call("get_state").get("laboratories", [])
    for lab in labs:
        for bench in lab.get("benches", []):
            return bench["uri"]
    raise SystemExit(_fail("the workspace holds no laboratory workbench"))


def _fail(why: str) -> int:
    print(json.dumps({"ran": False, "why": why}))
    return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bentrix", description="Bentrix headless.")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tape", help="write the CI tape and a workspace over it")
    t.add_argument("directory")
    t.add_argument("--first", default="2024-03-04")
    t.add_argument("--last", default="2024-03-08")
    t.add_argument("--symbol", default="XAUUSD")
    for name in ("backtest", "battery"):
        s = sub.add_parser(name)
        s.add_argument("--workspace", required=True)
        s.add_argument("--store", required=True)
        s.add_argument("--bench")
        if name == "backtest":
            s.add_argument("--strategy", required=True, help="module:Class, a bentrix.backtest.Strategy")
            s.add_argument("--interval", default="H1")
            s.add_argument("--max-bars", type=int)
    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        if args.command == "tape":
            from bentrix import _native

            ws, store = _native.make_tape(args.directory, args.first, args.last, args.symbol)
            print(json.dumps({"ran": True, "workspace": ws, "store": store}))
            return 0
        local = _local(args)
        bench = _bench(local, args.bench)
        if args.command == "backtest":
            module, _, cls = args.strategy.partition(":")
            strategy = getattr(importlib.import_module(module), cls)()
            from bentrix.backtest import run

            out = run(strategy, bench, interval=args.interval, local=local, max_bars=args.max_bars)
            print(json.dumps({"ran": True, "bench": bench, **out}))
            return 0
        local.call("battery_run", bench=bench)
        while True:
            got = local.call("battery_get")
            if got.get("checks") is not None:
                print(json.dumps({"ran": True, **got}))
                return 1 if got.get("failed_checks") else 0
            local.settle(50)
    except (RuntimeError, ValueError, ImportError, AttributeError) as e:
        return _fail(str(e))


if __name__ == "__main__":
    sys.exit(main())
