"""**``python -m bentrix``: Bentrix headless, for a terminal and for CI**.

    python -m bentrix tape DIR [--first 2024-03-04] [--last 2024-03-08] [--symbol XAUUSD]
    python -m bentrix backtest --workspace WS --store STORE --strategy module:Class [--bench URI]
                              [--interval H1] [--max-bars N]
    python -m bentrix battery --workspace WS --store STORE [--bench URI]
    python -m bentrix doctor
    python -m bentrix new env FILE

``new env`` writes an environment of your own into ``FILE`` (refused if it exists): a Gymnasium
environment over the engine, ninety lines yours to change - your observation, actions and reward,
not a base class. Run as it stands, it checks itself with Gymnasium's own ``check_env`` on a
practice tape.

``doctor`` says, one check a line of its answer, what a first run meets: the engine library found and
its build; the package's catalogue beside the engine's; whether the engine opens in your program on
your account's package, and the package; whether your own workspace is held, and by which process;
and the first steps (a practice tape, an order at an H1 close, the book and the account) run in a
temporary folder. It exits ``1`` when a check failed.

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


def _check(checks, name, ok, said, **more):
    checks.append({"check": name, "ok": bool(ok), "said": said, **more})
    return ok


def doctor() -> int:
    """The checks (module doc), each said; ``0`` when all passed, ``1`` when one failed."""
    import os
    import tempfile

    checks: list = []
    try:
        from bentrix import _native
    except ImportError as e:
        _check(checks, "library", False, str(e))
        print(json.dumps({"ran": True, "ok": False, "checks": checks}))
        return 1
    _check(checks, "library", True, f"the engine library {getattr(_native._lib, '_name', '?')}, build {_native.build()}", build=_native.build())
    from bentrix import CATALOGUE_VERSION, practice

    try:
        held = _native.workspace_held(None)
        if held["held"]:
            _check(checks, "workspace", False, held["said"] + " Close it, or open another workspace (Local(workspace=...)).", **{k: held[k] for k in ("workspace", "by")})
        else:
            _check(checks, "workspace", True, f"your workspace ({held['workspace']}) is not held by another process" + ("" if held["exists"] else "; it does not exist yet"), workspace=held["workspace"])
    except RuntimeError as e:
        _check(checks, "workspace", False, str(e))
    unpaid = os.environ.get("BENTRIX_ENGINE_UNPAID")
    with tempfile.TemporaryDirectory(prefix="bentrix-doctor-") as where:
        p = practice(os.path.join(where, "practice"))
        try:
            local = p.open()
        except RuntimeError as e:
            _check(checks, "package", False, str(e))
            print(json.dumps({"ran": True, "ok": False, "checks": checks}))
            return 1
        try:
            account = local.call("account_read")
            tier = account.get("tier")
            said = "the engine opens in your program" + (f"; your package is {tier}" if tier else "")
            if unpaid:
                said += "; BENTRIX_ENGINE_UNPAID is set, so the package was not asked"
            _check(checks, "package", True, said, tier=tier, asked=not unpaid)
            bench = p.bench(local)
            engine = local.call("bench_manifest", bench=bench)["document"]["engine"]["catalogue"]
            _check(checks, "catalogue", engine == CATALOGUE_VERSION, f"the package's catalogue is {CATALOGUE_VERSION}, the engine's {engine}" + ("" if engine == CATALOGUE_VERSION else ": install the package that came with this Bentrix"), package=CATALOGUE_VERSION, engine=engine)
            local.arm(bench)
            close = local.next_close(bench, "H1", ["XAUUSD"])
            local.call("order_market", bench=bench, symbol="XAUUSD", side="buy", volume=0.1)
            local.next_close(bench, "H1", ["XAUUSD"])
            book = local.book(bench)
            home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
            equity = local.read("trade_account", home=home, owner=owner, bench=name).get("equity")
            positions = book.get("positions") or []
            ok = close is not None and len(positions) == 1 and equity is not None
            _check(checks, "first_steps", ok, f"a practice tape made, an H1 close read, a buy filled: {len(positions)} position, equity {equity}" if ok else f"the first steps did not end as they should: {len(positions)} positions, equity {equity}")
        except RuntimeError as e:
            _check(checks, "first_steps", False, str(e))
        finally:
            local.close()
    ok = all(c["ok"] for c in checks)
    print(json.dumps({"ran": True, "ok": ok, "checks": checks}))
    return 0 if ok else 1


TEMPLATES = {"env": "my_env.py"}


def new(kind: str, path: str) -> int:
    """Write the template of ``kind`` into ``path``, never over a file that is there."""
    import os
    from importlib import resources

    if kind not in TEMPLATES:
        return _fail(f"{kind} is not a template; one of {', '.join(TEMPLATES)}")
    if os.path.exists(path):
        return _fail(f"{path} is there already; name another file")
    text = (resources.files("bentrix") / "_examples" / TEMPLATES[kind]).read_text(encoding="utf-8")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(json.dumps({"ran": True, "wrote": os.path.abspath(path), "template": TEMPLATES[kind]}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bentrix", description="Bentrix headless.")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tape", help="write the CI tape and a workspace over it")
    t.add_argument("directory")
    t.add_argument("--first", default="2024-03-04")
    t.add_argument("--last", default="2024-03-08")
    t.add_argument("--symbol", default="XAUUSD")
    n = sub.add_parser("new", help="write a template of your own: env, an environment over the engine")
    n.add_argument("kind", choices=sorted(TEMPLATES))
    n.add_argument("path")
    sub.add_parser("doctor", help="say what a first run meets: the library, the package, the workspace, the first steps")
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
        if args.command == "doctor":
            return doctor()
        if args.command == "new":
            return new(args.kind, args.path)
        if args.command == "tape":
            from bentrix import _native

            ws, store = _native.make_tape(args.directory, args.first, args.last, args.symbol)
            print(json.dumps({"ran": True, "workspace": ws, "store": store}))
            return 0
        if args.command == "backtest":
            # The strategy acts on the workbench object, as `bench.run` hands it (SDK 4.0).
            module, _, cls = args.strategy.partition(":")
            strategy = getattr(importlib.import_module(module), cls)()
            import bentrix
            from bentrix.backtest import run

            with bentrix.open(args.workspace, store=args.store) as bx:
                bench = bx.bench(args.bench)
                out = run(strategy, bench, interval=args.interval, max_bars=args.max_bars)
                print(json.dumps({"ran": True, "bench": bench.uri, **out}))
            return 0
        local = _local(args)
        bench = _bench(local, args.bench)
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
