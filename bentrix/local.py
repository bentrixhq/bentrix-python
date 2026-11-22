"""**BENTRIX in your own Python process**.

``Local`` opens your BENTRIX workspace with no window and no server — the extension module
``bentrix._native`` holds the engine in this process — and answers the same tools an agent calls,
by the same names and in the same shapes (the generated client's catalogue). Its reach is read,
laboratory and paper: never a real account, whose orders the person confirms in the window. A
window or a ``--serve`` already on the workspace refuses it, by name.
"""

from __future__ import annotations

import json
from typing import Any, Optional


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
        return json.loads(self._s.call(tool, json.dumps({k: v for k, v in args.items() if v is not None})))

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

    def snapshot(self, bench: str) -> int:
        """**Keep where a workbench stands** — its moment and every act of its venue — so ``restore``
        brings it back and a sweep branches from one state without arming again. Kept in this
        process; its id."""
        return self._s.snapshot(bench)

    def restore(self, bench: str, snapshot: int) -> int:
        """**Bring a workbench back to a snapshot**: its book rebuilt from the kept acts at the kept
        moment — every order, fill and close made after it gone. Answers where it stands."""
        return self._s.restore(bench, snapshot)

    def state(self, bench: str) -> Any:
        """A workbench as it stands (the ``workbench_state`` resource): its cursor, days, zone and
        markets, read without moving its clock."""
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        return self.read("workbench_state", home=home, owner=owner, bench=name)

    def settle(self, max_frames: int = 1000) -> None:
        self._s.settle(max_frames)

    def read(self, resource: str, **vars: str) -> Any:
        """One resource by its catalogue name, its template's variables named."""
        return json.loads(self._s.read(resource, json.dumps(vars)))

    def book(self, bench: str) -> Any:
        """A workbench's open positions and pending orders (the ``trade_book`` resource)."""
        home, owner, _, name = bench.removeprefix("bentrix://").split("/")[:4]
        return self.read("trade_book", home=home, owner=owner, bench=name)
