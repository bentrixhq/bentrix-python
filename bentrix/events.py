"""Bind a function to a workbench's ticks or to its bars closing, and receive each one as an event.

    import bentrix
    b = bentrix.connect()
    events = bentrix.Events(b)

    @events.on_bar_close("bentrix://lab/1/bench/2", "EURUSD", "M5")
    def closed(bar):
        print(bar["time_ms"], bar["close"])

    @events.on_tick("bentrix://lab/1/bench/2", "EURUSD")
    def tick(t):
        ...

    events.run()

**What each binding delivers.** ``on_tick`` every tick of one market, in order, none skipped.
``on_bar_close`` each bar of one market and interval **once, when it closes** — never while it
forms. ``on_bar_update`` the bar still forming, each time it moves, and each bar again at its close.
An event is a dict keyed by the stream's own column names.

**On the workbench's clock, not this machine's.** On a laboratory bench a bar closes when the
replay passes its end — under Play, a step or an agent's ``replay_step`` alike. A rewind behind a
binding's cursor calls its ``on_rewind`` (when given) with the stream's new cursor, and the ticks or
bars after it are delivered again as the clock passes them again.

**Where the handlers run.** Each binding reads its stream on a thread of its own through
``wait_for``, so nothing is lost while a handler works; every handler is called on the thread that
called :meth:`Events.run`, one event at a time, in the order the events arrived. A handler that
raises stops :meth:`run` with that exception.

Reading a stream as an iterator (``b.data.live_ticks(...)``, ``b.data.closed_bars(...)``) stays
available; this module is the same streams with the waiting done for you.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Dict, List, Optional

from ._runtime import BentrixError, Client

Event = Dict[str, Any]
Handler = Callable[[Event], None]

_STOP = object()


class _Binding:
    def __init__(self, uri: str, handler: Handler, on_rewind: Optional[Handler]) -> None:
        self.uri = uri
        self.handler = handler
        self.on_rewind = on_rewind


def _row(columns: Any, item: Any) -> Event:
    """A stream's row as a dict: an array row keyed by the page's columns; an object row as it is."""
    if isinstance(item, list) and isinstance(columns, list):
        return dict(zip(columns, item))
    if isinstance(item, dict):
        return dict(item)
    return {"value": item}


class Events:
    """Bindings of functions to a running Bentrix's streams. See the module's documentation."""

    def __init__(self, client: Client, *, timeout_ms: int = 30_000) -> None:
        # A ``bentrix.Bentrix`` from ``connect`` is accepted for its connection (the application's door).
        door = getattr(client, "_door", None)
        self._c = getattr(door, "client", client)
        self._timeout_ms = timeout_ms
        self._bindings: List[_Binding] = []
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._stop = threading.Event()

    # --- binding -----------------------------------------------------------------------------------

    @staticmethod
    def _bench(bench: str) -> str:
        bench = bench.rstrip("/")
        if "/bench/" not in bench or not bench.startswith("bentrix://"):
            raise BentrixError(f"{bench!r} is not a workbench's uri (bentrix://…/bench/<id>)")
        return bench

    def _bind(self, uri: str, handler: Optional[Handler], on_rewind: Optional[Handler]) -> Any:
        def register(fn: Handler) -> Handler:
            self._bindings.append(_Binding(uri, fn, on_rewind))
            return fn

        return register if handler is None else register(handler)

    def on_tick(self, bench: str, symbol: str, handler: Optional[Handler] = None, *, on_rewind: Optional[Handler] = None) -> Any:
        """Call ``handler`` with every tick of ``symbol`` on ``bench``. Usable as a decorator."""
        return self._bind(f"{self._bench(bench)}/ticks/{symbol}/live", handler, on_rewind)

    def on_bar_close(
        self, bench: str, symbol: str, timeframe: str, handler: Optional[Handler] = None, *, on_rewind: Optional[Handler] = None
    ) -> Any:
        """Call ``handler`` once with each bar of ``symbol`` and ``timeframe`` (M1, H1, …) as it closes."""
        return self._bind(f"{self._bench(bench)}/bars/{symbol}/{timeframe}/closed", handler, on_rewind)

    def on_bar_update(self, bench: str, symbol: str, timeframe: str, handler: Optional[Handler] = None) -> Any:
        """Call ``handler`` with the forming bar each time it moves, and with each bar again as it closes."""
        return self._bind(f"{self._bench(bench)}/bars/{symbol}/{timeframe}/live", handler, None)

    # --- running -----------------------------------------------------------------------------------

    def stop(self) -> None:
        """End :meth:`run` after the event being handled; safe from a handler or another thread."""
        self._stop.set()
        self._queue.put(_STOP)

    def run(self, *, until: Optional[Callable[[], bool]] = None) -> None:
        """Deliver every binding's events until :meth:`stop`, ``until()`` turning true, or a handler raising."""
        if not self._bindings:
            raise BentrixError("nothing is bound: call on_tick, on_bar_close or on_bar_update first")
        self._stop.clear()
        threads = [threading.Thread(target=self._pump, args=(b,), daemon=True, name=f"bentrix-events {b.uri}") for b in self._bindings]
        for t in threads:
            t.start()
        try:
            while not self._stop.is_set():
                if until is not None and until():
                    break
                try:
                    got = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if got is _STOP:
                    break
                if isinstance(got, BaseException):
                    raise got
                fn, event = got
                fn(event)
        finally:
            self._stop.set()

    def _pump(self, b: _Binding) -> None:
        """One binding's reader: the present first, then every page since its cursor."""
        try:
            cursor = self._c._read_uri(b.uri).get("next")
            while not self._stop.is_set():
                page = self._c.call("wait_for", {"uri": b.uri, "timeout_ms": self._timeout_ms}, {"after": cursor})
                if page.get("rewound") and b.on_rewind is not None:
                    self._queue.put((b.on_rewind, {"next": page.get("next")}))
                columns = page.get("columns")
                for item in page.get("items") or []:
                    self._queue.put((b.handler, _row(columns, item)))
                if page.get("next") is not None:
                    cursor = page["next"]
        except BaseException as e:  # handed to run(), which raises it on the caller's thread
            if not self._stop.is_set():
                self._queue.put(e)
