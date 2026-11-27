"""**Bentrix in your own Python process**.

``Local`` opens your Bentrix workspace with no window and no server — the extension module
``bentrix._native`` holds the engine in this process — and answers the same tools an agent calls,
by the same names and in the same shapes (the generated client's catalogue). Its reach is read,
laboratory and paper: never a real account, whose orders the person confirms in the window. A
window or a ``--serve`` already on the workspace refuses it, by name.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class Profile:
    """**Where a block's time went** (``Local.profile``): its wall time, the time inside the engine's native calls -
    each method's calls and seconds - and the rest, which is Python's: features, the policy, JSON, the loop."""

    def __init__(self) -> None:
        self.methods: Dict[str, List[float]] = {}
        self.wall = 0.0

    @property
    def engine(self) -> float:
        """Seconds inside the engine's native calls."""
        return sum(t for _, t in self.methods.values())

    @property
    def python(self) -> float:
        """Seconds outside them: the block's own work."""
        return max(self.wall - self.engine, 0.0)

    def __str__(self) -> str:
        rows = [f"{'wall':16s} {self.wall * 1000:10.1f} ms",
                f"{'engine':16s} {self.engine * 1000:10.1f} ms",
                f"{'python':16s} {self.python * 1000:10.1f} ms"]
        for name, (n, t) in sorted(self.methods.items(), key=lambda kv: -kv[1][1]):
            rows.append(f"  {name:14s} {t * 1000:10.1f} ms  {int(n):7d} calls  {t * 1e6 / max(n, 1):8.1f} us each")
        return "\n".join(rows)


class _Timed:
    """The session, each of its methods timed into a Profile."""

    def __init__(self, inner: Any, profile: Profile):
        self._inner, self._profile = inner, profile

    def __getattr__(self, name: str) -> Any:
        f = getattr(self._inner, name)
        if not callable(f):
            return f

        def timed(*a: Any, **k: Any) -> Any:
            began = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                row = self._profile.methods.setdefault(name, [0, 0.0])
                row[0] += 1
                row[1] += time.perf_counter() - began

        return timed


class Local:
    """An embedded Bentrix over ``workspace`` (yours when ``None``) and its recordings."""

    def __init__(self, workspace: Optional[str] = None, store: Optional[str] = None):
        from bentrix import _native  # the extension module; built into the wheel

        self._s = _native.Session(workspace, store)

    def close(self) -> None:
        """**Write the workspace and let it go.** The workbench's state is written when its session
        ends; an interpreter killed before that writes nothing, so a run that matters closes it — or
        uses ``with Local(...) as local:``. Closed, every call is refused."""
        self._s.close()

    def __enter__(self) -> "Local":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def call(self, tool: str, **args: Any) -> Any:
        """One tool by name; its answer, or its refusal raised as ``RuntimeError``."""
        return json.loads(self._s._call(tool, json.dumps({k: v for k, v in args.items() if v is not None})))

    def arm(self, bench: str, max_frames: int = 20000) -> None:
        """Arm ``bench``'s engine — the venue over its recording — waiting until it is ready."""
        self._s.arm(bench, max_frames)

    def advance(self, bench: str, to_ms: int) -> int:
        """Advance ``bench`` a tick at a time to ``to_ms`` on its clock, every tick fed to its venue."""
        return self._s.advance(bench, to_ms)

    def candle_span(self, bench: str, interval: str, at_ms: int) -> "tuple[int, int]":
        """The candle of ``interval`` holding ``at_ms`` on ``bench``'s own cut, as its chart cuts it:
        ``(opens, closes)`` in UTC ms — a D1 at the bench's zone's midnight, a W1 from Monday."""
        return tuple(self._s.candle_span(bench, interval, at_ms))

    def next_close(self, bench: str, interval: str, symbols: "list[str]", end_ms: Optional[int] = None) -> Any:
        """**One close of ``interval``, in one call**: the workbench advanced to the close of the candle
        its clock stands in, the venue's answers collected, each market's closed candle read —
        ``(reached, opened, [(symbol, [time_ms, open, high, low, close, volume])])``, or ``None`` at
        ``end_ms`` or where the clock moves no further."""
        return self._s.next_close(bench, interval, list(symbols), end_ms)

    def run_until(self, bench: str, interval: str, n: int, symbols: "list[str]", end_ms: Optional[int] = None) -> "list[Any]":
        """``next_close`` ``n`` times, or until the work ends, in one call: the closes in order — for a
        reader that acts on none of them (a dataset, a scan)."""
        return self._s.run_until(bench, interval, list(symbols), n, end_ms)

    def step(self, bench: str, acts: "list[tuple[str, dict]]" = (), until: Any = None, reads: "list[str]" = ()) -> Any:
        """**A whole environment step in one call**: each ``(tool, args)`` of ``acts`` called in order (a refusal
        answered in its place, as ``{"error": ...}``), then the clock moved by ``until`` — ``("close", interval,
        [symbols])`` to the next close, ``("to", ms)`` to a moment, or ``None`` — then each resource of ``reads`` read
        after it. Answers ``{"acts", "close", "cursor_ms", "reads"}``: the same as the calls one by one, in one crossing."""
        spec: dict = {"acts": [{"tool": t, "args": {k: v for k, v in a.items() if v is not None}} for t, a in acts], "reads": list(reads)}
        if until is not None:
            if until[0] == "close":
                spec["until"] = {"close": {"interval": until[1], "symbols": list(until[2])}}
            elif until[0] == "to":
                spec["until"] = {"to_ms": until[1]}
            else:
                raise ValueError("until is (\"close\", interval, symbols) or (\"to\", ms)")
        return json.loads(self._s.step(bench, json.dumps(spec)))

    def engine_stats(self, reset: bool = False) -> Any:
        """**The engine's counters**: each tool's and read's calls and milliseconds, the frames turned, the ticks fed to
        every armed workbench, how many are armed, and this process's resident bytes. ``reset`` starts the calls and
        frames again after answering."""
        return json.loads(self._s.stats(reset))

    def profile(self) -> "_Profiling":
        """**Where a block's time goes**: ``with local.profile() as p: ...`` then ``print(p)`` - the block's wall time,
        the time inside the engine's native calls by method, and the rest, which is the block's own Python."""
        return _Profiling(self)

    def snapshot(self, bench: str) -> int:
        """**Keep where a workbench stands** — its moment and every act of its venue — so ``restore``
        brings it back and a sweep branches from one state without arming again. Kept in this
        process; its id."""
        return self._s.snapshot(bench)

    def restore(self, bench: str, snapshot: int) -> int:
        """**Bring a workbench back to a snapshot**: its book rebuilt from the kept acts at the kept
        moment — every order, fill and close made after it gone. Answers where it stands."""
        return self._s.restore(bench, snapshot)

    def keep(self, bench: str, name: str) -> str:
        """**Keep where a workbench stands, to watch it on the chart**: a clone born in its state - its book, ledger and
        clock - named ``name`` in the rail (``"ppo-17 seed 3"``). Close your program, open Bentrix, select it and replay
        it with every trade drawn; this workbench goes on apart. Answers the kept workbench's uri."""
        return self.call("bench_clone", bench=bench, state=True, name=name)["uri"]

    def state(self, bench: str) -> Any:
        """A workbench as it stands (the ``workbench_state`` resource): its cursor, days, zone and
        markets, read without moving its clock."""
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        return self.read("workbench_state", home=home, owner=owner, bench=name)

    def settle(self, max_frames: int = 1000) -> None:
        self._s.settle(max_frames)

    def read(self, resource: str, **vars: str) -> Any:
        """One resource by its catalogue name, its template's variables named."""
        return json.loads(self._s._read(resource, json.dumps(vars)))

    def book(self, bench: str) -> Any:
        """A workbench's open positions and pending orders (the ``trade_book`` resource)."""
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        return self.read("trade_book", home=home, owner=owner, bench=name)


class _Profiling:
    """``with local.profile() as p``: the session timed for the block, put back after it."""

    def __init__(self, local: Local):
        self._local, self._profile = local, Profile()

    def __enter__(self) -> Profile:
        self._inner = self._local._s
        self._local._s = _Timed(self._inner, self._profile)
        self._began = time.perf_counter()
        return self._profile

    def __exit__(self, *exc: Any) -> None:
        self._profile.wall = time.perf_counter() - self._began
        self._local._s = self._inner
