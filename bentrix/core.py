"""**Bentrix in two objects**: ``Bentrix``, through either door, and ``Bench``, the workbench a program works on.

    import bentrix

    bx = bentrix.open()                 # the engine in this process, over your workspace
    bx = bentrix.open("folder")         # ...a workspace in a folder (a practice tape's, or a workspace)
    bx = bentrix.open(bentrix.practice())
    bx = bentrix.connect()              # the Bentrix open on this computer

    bench = bx.bench()                  # the selected workbench, else the first; bx.bench("name"); bx.benches
    answer = bench.buy(0.10, sl=2095.0)
    if not answer.ok:
        print(answer.reason)            # the venue's words; answer.kind, answer.remedy, answer.docs
    bench.step("M1")                    # to the next close of M1
    position = bench.positions[0]
    bench.modify(position, sl=2097.0)   # only what is named changes
    bench.close(position, lots=0.04)

**One door, two places.** ``open`` runs the engine in this process (``bentrix.local.Local``); ``connect`` talks to the
running application over its agents' connection. Both answer the same tools and resources, so everything here is
written once, against ``call``: a resource is read through the ``resource_read`` tool, which answers what
``resources/read`` gives for its URI. ``bx.tools`` holds the generated typed methods by domain
(``bx.tools.trade.order_market(...)``) over either door. Only the engine steps in process (``Bench.step`` is one
call there); on the application the workbench's own clock moves, and ``step`` asks the replay to move it.

**An order never raises for a refusal**: ``Answer.ok`` is the check, ``Answer.reason`` the venue's words beside its
``kind``, ``remedy`` and Docs page. ``bentrix.open(..., raise_refusals=True)`` raises ``Refused`` instead. Every
error is a ``bentrix.Error``.

**Arguments a newcomer would guess**: ``lots`` for a volume; ``symbol`` optional - the workbench's market when it has
one, refused by name when it has several; a position or an order wherever a ticket is taken; a moment as
milliseconds, a ``datetime``, or a string read in UTC (``"2024-03-04 07:40"``).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote


from ._runtime import Error, ToolError


class Held(Error):
    """The workspace is held by another program - a Bentrix window or a ``--serve`` - and is not opened twice."""


class NotSignedIn(Error):
    """The engine in a program's own process is Plus and Pro: this computer is not signed in to Bentrix with either."""


class Refused(Error):
    """An order the venue refused, raised only under ``raise_refusals=True``; ``answer`` holds its ``Answer``."""

    def __init__(self, answer: "Answer"):
        super().__init__(answer.reason or "refused")
        self.answer = answer


Moment = Union[int, float, str, _dt.datetime, _dt.date]


def ms(at: Moment) -> int:
    """**A moment as milliseconds since 1970 (UTC)**: an integer as given; a ``datetime`` (a naive one read in UTC);
    a ``date`` at its midnight; a string ``"2024-03-04"``, ``"2024-03-04 07:40"`` or ISO 8601, read in UTC."""
    if isinstance(at, bool):
        raise TypeError("a moment is not a bool")
    if isinstance(at, (int, float)):
        return int(at)
    if isinstance(at, str):
        text = at.strip().replace("T", " ").rstrip("Z")
        for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                at = _dt.datetime.strptime(text, f)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"{at!r} is not a moment: write 2024-03-04 or 2024-03-04 07:40 (UTC)")
    if isinstance(at, _dt.datetime):
        if at.tzinfo is None:
            at = at.replace(tzinfo=_dt.timezone.utc)
        return int(at.timestamp() * 1000)
    if isinstance(at, _dt.date):
        return ms(_dt.datetime(at.year, at.month, at.day))
    raise TypeError(f"{at!r} is not a moment")


def iso(at_ms: int) -> str:
    """Milliseconds since 1970 as ``YYYY-MM-DD HH:MM:SS`` in UTC (with milliseconds when there are any)."""
    t = _dt.datetime.fromtimestamp(at_ms / 1000, tz=_dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S") + (f".{at_ms % 1000:03d}" if at_ms % 1000 else "")


class _Record(dict):
    """**A record of the workbench's**: a dict whose keys read as attributes too (``position.sl`` and
    ``position["sl"]``). Each kind declares its fields - an editor completes them, and a typo fails at the attribute,
    naming the fields there are."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            fields = ", ".join(_fields(type(self)) or sorted(self))
            raise AttributeError(f"{type(self).__name__} has no {name!r}; it has {fields}") from None

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | set(_fields(type(self))) | set(self))


def _fields(kind: type) -> List[str]:
    """A record kind's declared fields, in order, from its annotations."""
    out: List[str] = []
    for k in reversed(kind.__mro__):
        out += [f for f in k.__dict__.get("__annotations__", {}) if f not in out]
    return out


# Each kind's fields are the engine's own answer, field for field: sdk/python/tests/test_bench.py's census reads them
# from a workbench and refuses one declared and not answered, or answered and not declared.


class Position(_Record):
    """An open position, as the workbench's book holds it."""

    ticket: int
    symbol: str
    side: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    #: The opening side's charge, money out negative as a deal's and the account's.
    commission: float
    swap: float
    opened_ms: int
    identifier: int
    magic: int
    tag: Optional[str]
    group: Optional[int]
    trail: Optional[float]
    best: Optional[float]
    worst: Optional[float]
    best_price: Optional[float]
    worst_price: Optional[float]

    @property
    def lots(self) -> float:
        """Its volume, in lots."""
        return self["volume"]

    @property
    def entry(self) -> float:
        """The price it opened at (``price_open``)."""
        return self["price_open"]


class Order(_Record):
    """A resting order, as the workbench's book holds it."""

    ticket: int
    symbol: str
    kind: str
    volume: float
    price: float
    stop_limit: Optional[float]
    sl: float
    tp: float
    expires_ms: Optional[int]
    magic: int
    tag: Optional[str]
    group: Optional[int]

    @property
    def lots(self) -> float:
        """Its volume, in lots."""
        return self["volume"]


class Account(_Record):
    """The workbench's account, in its currency: what it holds now and its totals since the workbench began."""

    bench: str
    source: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level_pct: Optional[float]
    stop_out_pct: Optional[float]
    leverage: Optional[int]
    profit: Optional[float]
    realized: Optional[float]
    commission: Optional[float]
    swap: Optional[float]
    peak_equity: Optional[float]
    drawdown: Optional[float]
    max_drawdown: Optional[float]

    @property
    def free(self) -> float:
        """The free margin."""
        return self["free_margin"]

    @property
    def level(self) -> Optional[float]:
        """The margin level, in per cent."""
        return self["margin_level_pct"]

    @property
    def peak(self) -> Optional[float]:
        """The highest equity since the workbench began."""
        return self["peak_equity"]


class Deal(_Record):
    """A deal: an entry ``in``, ``out``, ``in_out`` or ``out_by``, at ``time_ms`` and ``price``, and what it made."""

    ticket: int
    order: int
    position: int
    symbol: str
    side: str
    entry: str
    volume: float
    price: float
    time_ms: int
    profit: float
    commission: float
    swap: float
    magic: int
    tag: Optional[str]
    group: Optional[int]
    best: Optional[float]
    worst: Optional[float]
    best_price: Optional[float]
    worst_price: Optional[float]
    #: Why it was made: ``order`` (an order made it), ``sl``, ``tp`` or ``stop_out`` (the venue's own stop, target or
    #: stop-out), ``venue`` (the venue by itself); ``None`` on a real account's.
    reason: Optional[str]

    @property
    def lots(self) -> float:
        """Its volume, in lots."""
        return self["volume"]


class Event(_Record):
    """One entry of the venue's event log: its ``type`` and ``reason``, at ``time_ms``, numbered by ``seq``."""

    seq: int
    type: str
    reason: Optional[str]
    time_ms: int
    symbol: Optional[str]
    side: Optional[str]
    volume: Optional[float]
    price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    amount: Optional[float]
    order: Optional[int]
    position: Optional[int]
    trigger: Optional[str]
    tag: Optional[str]
    group: Optional[int]


class Quote(_Record):
    """A market's bid and ask at the clock, ``at_ms`` the tick's moment (none before the first)."""

    symbol: str
    bid: float
    ask: float
    digits: int
    at_ms: Optional[int]

    @property
    def time(self) -> Optional[int]:
        """The tick's moment (``at_ms``)."""
        return self["at_ms"]


class Tick(_Record):
    """One tick: its moment and prices, the venue's ``last`` and ``volume``, MetaTrader's ``flags``."""

    time_ms: int
    bid: float
    ask: float
    last: float
    volume: float
    flags: int


class Candle(_Record):
    """A closed candle: its open ``time`` (UTC ms) and prices."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def time_ms(self) -> int:
        """Its open time, as a strategy's ``Bar`` names it."""
        return self["time"]


class Exposure(_Record):
    """What the workbench holds against its equity: ``total``, and ``by_symbol`` one line per market."""

    bench: str
    total: Dict[str, Any]
    by_symbol: List[Dict[str, Any]]

    # **Every market's together is money, never lots**: a lot of gold is
    # 100 ounces and a lot of silver 5,000, so their sum means nothing; a market's own lots are ``lots(symbol)``.
    @property
    def long(self) -> float:
        """The face value held long, every market, in the account's currency."""
        return self["total"]["long_notional"]

    @property
    def short(self) -> float:
        """The face value held short, every market, in the account's currency."""
        return self["total"]["short_notional"]

    @property
    def net(self) -> float:
        """Long less short, in the account's currency."""
        return self["total"]["net_notional"]

    @property
    def gross(self) -> float:
        """Long and short together, in the account's currency."""
        return self["total"]["gross_notional"]

    def lots(self, symbol: str) -> Dict[str, float]:
        """One market's lots: ``long``, ``short``, ``net`` and ``gross``; zero for a market the book does not hold."""
        row = next((r for r in self["by_symbol"] if r.get("symbol") == symbol), None) or {}
        return {k: float(row.get(f"{k}_lots", 0.0)) for k in ("long", "short", "net", "gross")}


@dataclass
class Answer:
    """**What an order came to.** ``ok`` unless refused; ``outcome`` is the venue's word (accepted, filled, placed,
    modified, closed, cancelled, refused; on a real account sent or awaiting_person); a refusal's ``reason`` is the
    venue's own sentence, ``kind`` the word a program branches on, ``remedy`` what to change and ``docs`` the page.
    ``raw`` is the whole answer, the book included."""

    outcome: str
    reason: Optional[str]
    kind: Optional[str]
    remedy: Optional[str]
    docs: Optional[str]
    ticket: Optional[int]
    order: Optional[int]
    raw: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.outcome != "refused"

    @property
    def partial(self) -> bool:
        """**Only a first part filled**: the rest fills on the ticks to come - a workbench's execution model filling an
        order in parts, or a broker's partial fill (``partial: true`` beside ``filled``)."""
        last = self.raw.get("last_answer") or {}
        return bool(self.raw.get("partial", last.get("partial")))

    @classmethod
    def of(cls, raw: Dict[str, Any]) -> "Answer":
        last = raw.get("last_answer") or {}
        get = lambda k: raw.get(k, last.get(k))  # noqa: E731
        return cls(str(get("outcome") or "sent"), get("reason"), get("kind"), get("remedy"), get("docs"), get("ticket"), get("order"), raw)

    def __str__(self) -> str:
        from bentrix.refusals import explain

        return explain(self.raw) or self.outcome


class Bars:
    """**Candles as NumPy columns**: ``time`` (UTC ms), ``open``, ``high``, ``low``, ``close``, ``volume`` - by
    attribute or by key - one row a closed candle, oldest first; ``forming`` the candle still forming, when asked for."""

    COLUMNS = ("time", "open", "high", "low", "close", "volume")

    def __init__(self, rows: Sequence[Sequence[float]], forming: Optional[Sequence[float]] = None):
        import numpy as np  # here, not at import: bentrix itself depends on nothing; candles as columns need NumPy

        a = np.asarray([r[:6] for r in rows], dtype=np.float64).reshape(-1, 6) if rows else np.zeros((0, 6))
        self._columns = {name: a[:, i] for i, name in enumerate(self.COLUMNS)}
        self._columns["time"] = self._columns["time"].astype(np.int64)
        self.rows = [list(r) for r in rows]
        self.forming = list(forming) if forming else None

    def __getattr__(self, name: str) -> "np.ndarray":
        try:
            return self.__dict__["_columns"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> "np.ndarray":
        return self._columns[name]

    def __len__(self) -> int:
        return len(self.rows)


# ---- the doors -------------------------------------------------------------------------------------------


class _EngineDoor:
    """The engine in this process: ``bentrix.local.Local``."""

    def __init__(self, local: Any):
        self.local = local

    def call(self, tool: str, **args: Any) -> Any:
        try:
            return self.local.call(tool, **args)
        except Error:
            raise
        except RuntimeError as e:
            # The engine's refusal in Bentrix's own words, raised as the application's door raises it.
            raise ToolError(tool, str(e)) from None

    def close(self) -> None:
        self.local.close()


class _AppDoor:
    """The running application, over its agents' connection (the generated client)."""

    def __init__(self, client: Any):
        self.client = client

    def call(self, tool: str, **args: Any) -> Any:
        # A tool's refusal is the connection's ``ToolError``, already a ``bentrix.Error``.
        return self.client.call(tool, {k: v for k, v in args.items() if v is not None})

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


class _Typed:
    """What the generated typed methods call: ``call`` and ``read_resource`` over a door."""

    def __init__(self, door: Any):
        self._door = door

    def call(self, name: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._door.call(name, **{**required, **{k: v for k, v in (optional or {}).items() if v is not None}})

    def read_resource(self, template: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from bentrix._runtime import expand_template

        return self._door.call("resource_read", uri=expand_template(template, required, optional))

    def read_pages(self, template: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]], parse: Any) -> Iterator[Any]:
        from bentrix._runtime import expand_template

        uri: Optional[str] = expand_template(template, required, optional)
        while uri:
            page = self._door.call("resource_read", uri=uri)
            yield parse(page)
            uri = page.get("next") if isinstance(page.get("next"), str) and page.get("next", "").startswith("bentrix://") else None


def _engine_said(fn: Callable[..., Any], *args: Any) -> Any:
    """One of the engine's own calls, its refusal raised as ``bentrix.Error`` - the one tree a workbench's every
    refusal is in, whichever path it took."""
    try:
        return fn(*args)
    except Error:
        raise
    except RuntimeError as refusal:
        raise Error(str(refusal)) from None


def _every_bench(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every workbench of a ``get_state``: a laboratory's, then a REAL account's (paper among them)."""
    return [b for area in ("laboratories", "accounts") for home in state.get(area, []) for b in home.get("benches", [])]


class _Tools:
    """The generated typed methods, one attribute a domain (``trade``, ``data``, ``replay``, ...), over a door."""

    def __init__(self, door: Any):
        from bentrix import _generated

        client = door.client if isinstance(door, _AppDoor) else _Typed(door)
        for name, cls in vars(_generated).items():
            if name.startswith("_") and name.endswith("Api") and isinstance(cls, type):
                setattr(self, name[1:-3].lower(), cls(client))


# ---- Bentrix and its workbenches -------------------------------------------------------------------------


class Bentrix:
    """**Bentrix through a door** (module doc) - ``bentrix.open`` and ``bentrix.connect`` answer one."""

    def __init__(self, door: Any, raise_refusals: bool = False):
        self._door = door
        self.raise_refusals = raise_refusals
        self._tools: Optional[_Tools] = None
        self._armed: set = set()

    # the door
    def call(self, tool: str, args: Optional[Dict[str, Any]] = None, optional: Optional[Dict[str, Any]] = None, /, **kw: Any) -> Any:
        """One tool by name, its arguments as keywords or as one dict (the connection's form of before, a dict of
        the required and one of the optional, is taken too); its answer, or its refusal raised as
        ``bentrix.Error``. A ``Bench`` is taken wherever a tool takes a workbench's address."""
        given = {**(args or {}), **{k: v for k, v in (optional or {}).items() if v is not None}, **kw}
        for v in given.values():
            if isinstance(v, Bench):
                # Armed on its first use, as its own methods arm it. **A refusal to arm is not the tool's**: a sealed
                # part is refused an engine and is still what `bench_unseal` is called on, so the tool answers for itself.
                try:
                    v._arm()
                except Error:
                    pass
            elif isinstance(v, str) and v.startswith("bentrix://") and "/bench/" in v and self.engine is not None and v not in self._armed:
                # **A workbench named by its address is armed as its object is**: `bench_equity` on a reopened
                # workbench was refused *its engine is not armed* while every read of the workbench object armed first.
                # Kept only when it armed: a sealed part refused now arms after `bench_unseal`.
                try:
                    self.engine.arm(v)
                    self._armed.add(v)
                except RuntimeError:
                    pass
        return self._door.call(tool, **{k: (v.uri if isinstance(v, Bench) else v) for k, v in given.items()})

    def read(self, uri: str) -> Any:
        """One resource by its URI, as ``resources/read`` answers it."""
        return self._door.call("resource_read", uri=uri)

    @property
    def tools(self) -> _Tools:
        """The generated typed methods by domain: ``bx.tools.trade.order_market(...)``."""
        if self._tools is None:
            self._tools = _Tools(self._door)
        return self._tools

    @property
    def engine(self) -> Any:
        """The engine in this process (``bentrix.local.Local``), or ``None`` through the application."""
        return self._door.local if isinstance(self._door, _EngineDoor) else None

    # the workspace
    @property
    def laboratories(self) -> List[Dict[str, Any]]:
        return self.call("get_state").get("laboratories", [])

    @property
    def benches(self) -> List["Bench"]:
        """Every workbench, in the rail's order: the laboratories', then a REAL account's (paper among them)."""
        return [Bench(self, b["uri"], b.get("name")) for b in _every_bench(self.call("get_state"))]

    def bench(self, name: Optional[str] = None) -> "Bench":
        """**A workbench**: the one named (its name or its address), else the selected one, else the first."""
        state = self.call("get_state")
        every = _every_bench(state)
        if name is not None:
            for b in every:
                if name in (b.get("uri"), b.get("name")):
                    return Bench(self, b["uri"], b.get("name"))
            raise Error(f"no workbench is named {name!r}; there are {', '.join(repr(b.get('name')) for b in every) or 'none'}")
        selected = state.get("selected")
        for b in every:
            if selected and b.get("uri") == selected:
                return Bench(self, b["uri"], b.get("name"))
        if not every:
            raise Error("the workspace holds no workbench")
        return Bench(self, every[0]["uri"], every[0].get("name"))

    def close(self) -> None:
        """Let the door go: the engine closes (its workspace written), the connection ends."""
        self._door.close()

    def __getattr__(self, name: str) -> Any:
        """**What the connection had before 4.0 is still here**: through the application, every name of the generated
        client (``workspace``, ``agent``, ``trade`` and the other domains, ``call_job``, ``watch``, ``read_resource``,
        ``server_catalogue_version``); through the engine, the domains of ``tools``."""
        if name.startswith("_"):
            raise AttributeError(name)
        client = getattr(self._door, "client", None)
        if client is not None and hasattr(client, name):
            return getattr(client, name)
        tools = self.tools
        if hasattr(tools, name):
            return getattr(tools, name)
        raise AttributeError(f"'Bentrix' object has no attribute {name!r}")

    def __enter__(self) -> "Bentrix":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


_UNIT_MS = {"S": 1_000, "M": 60_000, "H": 3_600_000, "D": 86_400_000, "W": 604_800_000, "MN": 2_592_000_000}


def _interval_ms(name: str) -> int:
    """An interval's length, to choose the finest of several (``MN1`` counted as thirty days)."""
    unit = "MN" if name.startswith("MN") else name[:1]
    count = name[len(unit):]
    if unit not in _UNIT_MS or not count.isdigit():
        raise Error(f"{name} is not an interval: S, M, H, D, W or MN and a count (M5, H1, D1, MN1)")
    return _UNIT_MS[unit] * int(count)


def _ticket(x: Any) -> int:
    if isinstance(x, dict):
        return int(x["ticket"])
    return int(x)


class Bench:
    """**A workbench, the object a program works on** (module doc). It knows its address; nothing splits a URI."""

    def __init__(self, bx: Bentrix, uri: str, name: Optional[str] = None):
        self.bx, self.uri, self.name = bx, uri, name
        self._armed = False
        self._markets: Optional[List[str]] = None
        self._on_close: List[Tuple[str, Optional[str], Callable[[Candle], Any]]] = []
        self._on_tick: List[Tuple[Optional[str], Callable[[Quote], Any]]] = []
        self._on_event: List[Tuple[str, Callable[[Event], Any]]] = []

    def __repr__(self) -> str:
        return f"Bench({self.name or self.uri!r})"

    # -- through the door
    def _call(self, tool: str, **args: Any) -> Any:
        self._arm()
        return self.bx.call(tool, bench=self.uri, **args)

    def _read(self, tail: str, **query: Any) -> Any:
        self._arm()
        q = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in query.items() if v is not None)
        return self.bx.read(f"{self.uri}/{tail}" + (f"?{q}" if q else ""))

    def _arm(self) -> None:
        """**Nothing to arm**: the engine is armed on first use. In this process it is armed here; through the
        application a workbench with no engine mounted is selected - the window shows it, which is how Bentrix mounts
        one - and waited for until its engine is ready (at most a minute)."""
        if self._armed:
            return
        engine = self.bx.engine
        if engine is not None:
            try:
                engine.arm(self.uri)
            except RuntimeError as refusal:
                # A live or real workbench has no replay to arm; its tools say what it is. Any other refusal - a sealed
                # test part, a recording that will not open - is the workbench's answer, said as it is.
                # Read through the door, not ``_state``, which arms first.
                if not self.bx.read(self.uri).get("live"):
                    raise Error(str(refusal)) from None
            self._armed = True
            return
        self._armed = True
        import time

        if self.bx.read(self.uri).get("engine") != "none":
            return
        self.bx.call("select_bench", bench=self.uri)
        for _ in range(600):
            if self.bx.read(self.uri).get("engine") != "none":
                return
            time.sleep(0.1)

    def _symbol(self, symbol: Optional[str]) -> str:
        if symbol is not None:
            return symbol
        markets = self.markets
        if len(markets) == 1:
            return markets[0]
        raise Error(f"this workbench holds {len(markets)} markets ({', '.join(markets)}): name the symbol")

    def _answer(self, raw: Dict[str, Any]) -> Answer:
        a = Answer.of(raw)
        if not a.ok and self.bx.raise_refusals:
            raise Refused(a)
        return a

    # -- what it is
    @property
    def state(self) -> Dict[str, Any]:
        """The workbench as it stands (the ``workbench_state`` resource): its clock, days, zone and markets."""
        return self._state()

    def _state(self) -> Dict[str, Any]:
        self._arm()
        return self.bx.read(self.uri)

    def _facts(self) -> Dict[str, Any]:
        """**Its document's facts - its days and markets - read without its engine**: a sealed test part says where it
        starts and ends and what it holds while its clock and data stay behind the seal."""
        return self.bx.read(self.uri)

    @property
    def markets(self) -> List[str]:
        """The workbench's markets."""
        if self._markets is None:
            ms_ = self._facts().get("markets") or []
            self._markets = [m if isinstance(m, str) else m.get("symbol") for m in ms_]
        return list(self._markets)

    def manifest(self) -> Dict[str, Any]:
        """Its conditions and their digest (``bench_manifest``)."""
        return self._call("bench_manifest")

    def spec(self, symbol: Optional[str] = None) -> _Record:
        """A market's facts as numbers (``symbol_info``)."""
        return _Record(self._call("symbol_info", symbol=self._symbol(symbol)))

    # -- the clock
    @property
    def now(self) -> int:
        """Where its clock stands (UTC ms)."""
        return int(self._state().get("cursor_ms"))

    @property
    def start(self) -> int:
        """Where its days start (UTC ms) - read without its engine, as a sealed part's are."""
        return int((self._facts().get("bounds") or {}).get("from_ms"))

    @property
    def end(self) -> int:
        """Where its days end (UTC ms) - read without its engine, as a sealed part's are."""
        return int((self._facts().get("bounds") or {}).get("to_ms"))

    def step(self, interval: str = "M1", symbol: Optional[str] = None, *, ticks: Optional[int] = None, all: bool = False) -> Any:  # noqa: A002
        """**To the next close of ``interval``**: the candle that closed (a ``Candle``), or ``None`` where the clock moves
        no further; ``all=True``, a ``dict`` of every market's candle by symbol. ``ticks=n`` moves ``n`` ticks instead and
        answers where the clock stands.

        A workbench that has lived through no tick yet first lives through its first (a jump never invents a first
        quote - the product's own rule, *press Play*): from a tape's start both doors reach the first candle that holds
        ticks alike. The engine reads the close in one call; through the application the replay steps a tick, when it
        must, then a candle."""
        if ticks is not None:
            moved = self._call("replay_step", unit="tick", n=int(ticks))
            return int(moved.get("cursor_ms", self.now))
        syms = self.markets if all else [self._symbol(symbol)]
        engine = self.bx.engine
        if engine is not None:
            self._arm()
            got = _engine_said(engine.next_close, self.uri, interval, syms)
            if got is None:
                return None
            _, _, candles = got
            # A candle's time is a whole number of milliseconds at both doors (the engine's row carries floats).
            closed = {s: Candle(zip(Bars.COLUMNS, [int(row[0]), *row[1:]])) for s, row in candles}
        else:
            stood = self._state().get("cursor_ms")
            if any((self.quote(s) or {}).get("at_ms") is None for s in syms):
                self._call("replay_step", unit="tick", n=1)
            now = self._call("replay_step", unit="candle", n=1, timeframe=interval).get("cursor_ms")
            if now is None or now == stood:
                return None
            closed = {}
            for s in syms:
                rows = self.window([interval], 1, s)[interval].rows
                if rows:
                    closed[s] = Candle(zip(Bars.COLUMNS, rows[-1]))
        return closed if all else closed.get(syms[0])

    def candle(self, interval: str, at: Moment) -> Tuple[int, int]:
        """The candle of ``interval`` holding ``at``, as the workbench's chart cuts it: ``(opens, closes)`` in UTC ms - a
        D1 at its zone's midnight, a W1 from Monday (the engine)."""
        self._arm()
        return tuple(_engine_said(self._engine("a candle's span").candle_span, self.uri, interval, ms(at)))

    def advance(self, to: Moment) -> int:
        """Forward to ``to``, tick by tick, at most the days' last moment; where the clock stands."""
        target = ms(to)
        engine = self.bx.engine
        if engine is not None:
            self._arm()
            return _engine_said(engine.advance, self.uri, target)
        return self._call("replay_seek", to_ms=target, stop_at_end=True).get("cursor_ms", target)

    def rewind(self, to: Moment, undo_deals: bool = False) -> Dict[str, Any]:
        """Back to ``to``; refused with its door when that would undo a deal, unless ``undo_deals``."""
        return self._call("replay_seek", to_ms=ms(to), undo_deals=undo_deals)

    # -- orders
    def buy(self, lots: float, symbol: Optional[str] = None, *, sl: Optional[float] = None, tp: Optional[float] = None, tag: Optional[str] = None, group: Optional[int] = None) -> Answer:
        """A market buy of ``lots``; ``sl``/``tp`` as prices."""
        return self._answer(self._call("order_market", symbol=self._symbol(symbol), side="buy", volume=lots, sl=sl, tp=tp, tag=tag, group=group))

    def sell(self, lots: float, symbol: Optional[str] = None, *, sl: Optional[float] = None, tp: Optional[float] = None, tag: Optional[str] = None, group: Optional[int] = None) -> Answer:
        """A market sell of ``lots``; ``sl``/``tp`` as prices."""
        return self._answer(self._call("order_market", symbol=self._symbol(symbol), side="sell", volume=lots, sl=sl, tp=tp, tag=tag, group=group))

    def pending(self, kind: str, price: float, lots: float, symbol: Optional[str] = None, *, sl: Optional[float] = None, tp: Optional[float] = None, stop_limit: Optional[float] = None, expires: Optional[Moment] = None, tag: Optional[str] = None, group: Optional[int] = None) -> Answer:
        """A pending order: ``kind`` buy_limit, sell_limit, buy_stop, sell_stop, buy_stop_limit or sell_stop_limit."""
        return self._answer(self._call(
            "order_pending", symbol=self._symbol(symbol), kind=kind, price=price, volume=lots, sl=sl, tp=tp,
            stop_limit=stop_limit, expires_ms=ms(expires) if expires is not None else None, tag=tag, group=group,
        ))

    def modify(self, ticket: Any, *, sl: Optional[float] = None, tp: Optional[float] = None, price: Optional[float] = None) -> Answer:
        """**Only what is named changes**: a position's or an order's stop, target, or a pending order's price;
        ``0`` removes a level."""
        return self._answer(self._call("order_modify", ticket=_ticket(ticket), sl=sl, tp=tp, price=price))

    def close(self, ticket: Any, lots: Optional[float] = None) -> Answer:
        """Close a position (all of it, or ``lots``), or cancel a pending order."""
        return self._answer(self._call("order_close", ticket=_ticket(ticket), volume=lots))

    cancel = close

    def close_all(self, symbol: Optional[str] = None, group: Optional[int] = None) -> Answer:
        """Close every position, or one market's, or one group's; nothing open is an answer, not a refusal."""
        return self._answer(self._call("order_close_all", symbol=symbol, group=group))

    def reverse(self, ticket: Any) -> Answer:
        """Close a position and open the same size the other way, as one act (``order_reverse``)."""
        return self._answer(self._call("order_reverse", ticket=_ticket(ticket)))

    def untrail(self, ticket: Any) -> Answer:
        """Stop following a position's trailing stop; its stop stays where the trail left it (``order_untrail``)."""
        return self._answer(self._call("order_untrail", ticket=_ticket(ticket)))

    def group(self, label: str) -> int:
        """A group of orders under ``label`` (``group_open``): its number, for ``buy(..., group=)`` and
        ``close_all(group=)``."""
        return int(self._call("group_open", label=label)["group"])

    def trail(self, ticket: Any, points: float) -> Answer:
        """A trailing stop at ``points`` from the price, the venue's own."""
        return self._answer(self._call("order_trail", ticket=_ticket(ticket), points=points))

    def cancel(self, ticket: Any) -> Answer:
        """Cancel a pending order; the same act as ``close`` on an order's ticket."""
        return self.close(ticket)

    def preview(self, side: str, lots: float, symbol: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        """What an order would cost and hold, sending nothing (``order_preview``)."""
        return self._call("order_preview", symbol=self._symbol(symbol), side=side, volume=lots, **kw)

    # -- reading
    @property
    def book(self) -> Dict[str, Any]:
        return self._read("book")

    @property
    def positions(self) -> List[Position]:
        return [Position(p) for p in self.book.get("positions", [])]

    @property
    def orders(self) -> List[Order]:
        return [Order(o) for o in self.book.get("orders", [])]

    @property
    def account(self) -> Account:
        return Account(self._read("account"))

    def _stream(self, tail: str, kind: type = _Record, after: Optional[int] = None) -> List[Any]:
        """Every item of a stream, from its start - or past the first ``after`` items: pages read until one is empty."""
        out: List[Any] = []
        after = None if not after else str(int(after))
        while True:
            page = self._read(tail, after=after)
            items = page.get("items", [])
            if not items:
                return out
            out += [kind(x) for x in items]
            after = page.get("next")
            if after is None:
                return out

    def deals(self, since: Optional[Moment] = None) -> List[Deal]:
        """Every deal, oldest first; from ``since`` on when it is given."""
        every = self._stream("deals", Deal)
        return every if since is None else [d for d in every if d.time_ms >= ms(since)]

    def order_history(self) -> List[_Record]:
        """Every order the workbench was sent, oldest first: what it became and when."""
        return self._stream("orders")

    def ticks(self, day: Optional[Moment] = None, symbol: Optional[str] = None) -> List["Tick"]:
        """A day's ticks that have happened at the clock (the clock's day when ``day`` is not given), oldest first -
        every page read. Only a day the replay holds in memory is answered."""
        sym = self._symbol(symbol)
        when = iso(ms(day))[:10] if day is not None else iso(self.now)[:10]
        page = self._read(f"ticks/{quote(sym, safe='')}", day=when)
        out: List[Tick] = []
        while True:
            cols = page.get("columns") or []
            out += [Tick(zip(cols, row)) for row in page.get("rows", [])]
            if not page.get("next"):
                return out
            page = self.bx.read(page["next"])

    @property
    def exposure(self) -> Exposure:
        """What it holds against its equity: ``total`` and ``by_symbol``."""
        return Exposure(self._read("exposure"))

    def quote(self, symbol: Optional[str] = None) -> Optional[Quote]:
        """The market's bid and ask at the clock."""
        sym = self._symbol(symbol)
        for q in self._read("quotes").get("items", []):
            if q.get("symbol") == sym:
                return Quote(q)
        return None

    def window(self, frames: Sequence[str], limit: int = 100, symbol: Optional[str] = None) -> Dict[str, Bars]:
        """**Several timeframes at once**, closed candles only: ``{frame: Bars}``, each with its ``forming`` candle."""
        sym = self._symbol(symbol)
        got = self._call("bars_window", symbol=sym, frames=list(frames), limit=limit, include_forming=True)
        market = got["markets"][sym]
        return {f: Bars(market[f]["rows"], market[f].get("forming")) for f in frames}

    def bars(self, interval: str = "M1", limit: int = 100, symbol: Optional[str] = None) -> Bars:
        """The last ``limit`` closed candles of ``interval``."""
        return self.window([interval], limit, symbol)[interval]

    def events(self, since: Optional[int] = None) -> List[Event]:
        """The venue's event log (``trade_events``), oldest first; with ``since``, only the events numbered past it -
        read from there, never the whole log again."""
        return self._stream("events", Event, after=since)

    # -- state
    def snapshot(self) -> int:
        """Keep where it stands, in this process (the engine only); its id."""
        engine = self._engine("snapshot")
        self._arm()
        return _engine_said(engine.snapshot, self.uri)

    def restore(self, snapshot: int) -> int:
        """Back to a snapshot; where it stands."""
        return _engine_said(self._engine("restore").restore, self.uri, snapshot)

    def save(self, name: str, extra: bytes = b"") -> Dict[str, Any]:
        """A checkpoint to a file (``checkpoint_save``), with bytes of your own (``extra``) handed back by ``load``."""
        import base64

        return self._call("checkpoint_save", path=name, extra_b64=base64.b64encode(extra).decode("ascii") if extra else None)

    def load(self, name: str) -> Dict[str, Any]:
        """A checkpoint back (``checkpoint_load``), refused under other conditions; ``extra`` the bytes saved with it."""
        import base64

        got = self._call("checkpoint_load", path=name)
        got["extra"] = base64.b64decode(got.get("extra_b64") or "")
        return got

    def clone(self, state: bool = True, name: Optional[str] = None, profile: Optional[str] = None) -> "Bench":
        """A workbench alike, in this one's state by default - or a sibling at the start of its days under a profile's
        account, costs, spread and fills (``bench_profile``'s text; a profile takes ``state=False``)."""
        made = self._call("bench_clone", state=False if profile is not None else state, name=name, profile=profile)
        return Bench(self.bx, made["uri"], made.get("name"))

    def keep(self, name: str) -> "Bench":
        """Keep where it stands, to watch on the chart: a clone in its state, under ``name``."""
        return self.clone(True, name)

    def split(self, parts: Dict[str, Any], gap_days: Optional[int] = None) -> Dict[str, "Bench"]:
        """Train, validation and test parts (``bench_split``): ``{role: Bench}``."""
        made = self._call("bench_split", parts={k: {"from_ms": ms(v[0]), "to_ms": ms(v[1])} if isinstance(v, (tuple, list)) else v for k, v in parts.items()}, gap_days=gap_days)
        return {p["role"]: Bench(self.bx, p["uri"], p.get("name")) for p in made["parts"]}

    def unseal(self, reason: str) -> Dict[str, Any]:
        """**Open a split's sealed test part** (``bench_unseal``): the reason is written in the trial ledger first, which
        never removes it, then the part arms. Answers the ledger line's number (``ledger_seq``) and the part's ``looks``."""
        # Straight through the door: the part is sealed, so it is not armed before the tool opens it.
        got = self.bx.call("bench_unseal", bench=self.uri, reason=reason)
        self._armed = False
        return got

    def delete(self) -> Dict[str, Any]:
        """**Delete this workbench** (``bench_delete``, confirmed): its document and charts go, the recording it replays
        stays."""
        return self.bx._door.call("bench_delete", bench=self.uri, confirm=True)

    def walk(self, train_days: int, test_days: int, gap_days: int = 5, keep: bool = True) -> Iterator[Dict[str, "Bench"]]:
        """**The rolling form of ``split``**: from the workbench's first day, a training part of
        ``train_days``, a gap of ``gap_days`` and a sealed test part of ``test_days``; then the same again ``test_days``
        later, while a whole window fits in its days. Each window is a ``split`` - its parts made when it is reached,
        its test part sealed until ``bench_unseal`` writes why - and is yielded as ``{"train": Bench, "test": Bench}``.

        Each window is two workbenches, and they stay. With ``keep=False`` a window's two are deleted when the walk
        is asked for the next one, and the last when it ends - for a loop that reads each window as it comes; a window
        kept beyond that (``list(walk(...))``) is gone."""
        if min(train_days, test_days) < 1 or gap_days < 0:
            raise ValueError("a walk's training and test parts are a day or more, and its gap is no fewer than 0 days")
        day = 86_400_000
        start, end = self.start, self.end
        while start + (train_days + gap_days + test_days) * day <= end:
            train = (start, start + train_days * day)
            test_from = train[1] + gap_days * day
            window = self.split({"train": train, "test": (test_from, test_from + test_days * day)}, gap_days=gap_days)
            try:
                yield window
            finally:
                if not keep:
                    for part in window.values():
                        part.delete()
            start += test_days * day

    # -- running
    def on_close(self, interval: str, symbol: Optional[str] = None) -> Callable[[Callable[[Candle], Any]], Callable[[Candle], Any]]:
        """**A handler of every close of ``interval``** (a decorator), handed the ``Candle`` that closed - the market's
        named, else the workbench's one. ``run`` calls it."""
        def add(fn: Callable[[Candle], Any]) -> Callable[[Candle], Any]:
            self._on_close.append((interval, symbol, fn))
            return fn
        return add

    def on_tick(self, symbol: Optional[str] = None) -> Callable[[Callable[[Quote], Any]], Callable[[Quote], Any]]:
        """**A handler of every tick** (a decorator), handed the market's ``Quote``; with one, ``run`` steps tick by tick."""
        def add(fn: Callable[[Quote], Any]) -> Callable[[Quote], Any]:
            self._on_tick.append((symbol, fn))
            return fn
        return add

    def on(self, event: str) -> Callable[[Callable[[Event], Any]], Callable[[Event], Any]]:
        """**A handler of one kind of the venue's events** (a decorator): ``fill``, ``position_opened``,
        ``position_closed``, ``stop_loss``, ``take_profit``, ``order_rejected``, ... (the event log's ``type``), handed
        each ``Event`` as it happens; ``"*"`` every one."""
        def add(fn: Callable[[Event], Any]) -> Callable[[Event], Any]:
            self._on_event.append((event, fn))
            return fn
        return add

    def run(self, strategy: Any = None, interval: str = "H1", *, to: Optional[Moment] = None, max_steps: Optional[int] = None) -> int:
        """**Run the workbench's clock**, through either door, until its days end (or ``to``, or ``max_steps``), and
        answer how many steps it took.

        With ``strategy`` (a ``bentrix.backtest.Strategy``), at each close of ``interval`` its ``on_bar`` is handed the
        candle and this workbench - the context IS the workbench (``on_start`` and ``on_end`` around them); on a
        workbench of several markets its ``on_bars`` is handed each close's candles by symbol (the engine only). Without, the
        handlers registered by ``on_close``, ``on_tick`` and ``on`` are called: the clock steps by the finest interval a
        handler names, or tick by tick when one is ``on_tick``; each close, tick and event is handed to its own."""
        until = None if to is None else ms(to)
        if strategy is not None and len(self.markets) > 1:
            # Several markets on one clock: each close's candles by symbol to on_bars, as bentrix.backtest.run does -
            # on the engine in this process, which hands every market's candle of one close together.
            if self.bx.engine is None:
                raise Error("a strategy over several markets runs on the engine in this process (bentrix.open); through the application, run it on a workbench of one market")
            from .backtest import run as _run

            return int(_run(strategy, self, interval=interval, symbols=self.markets, max_bars=max_steps, to=until)["bars"])
        closes = list(self._on_close)
        if strategy is not None:
            closes.append((interval, None, lambda c: strategy.on_bar(c, self)))
            strategy.on_start(self)
        if not closes and not self._on_tick and not self._on_event:
            raise Error("nothing to run: pass a strategy, or register a handler with on_close, on_tick or on")
        finest = min((c[0] for c in closes), key=_interval_ms, default="M1")
        seen_close = {(iv, sym): self._last_close(iv, sym) for iv, sym, _ in closes}
        seen_event = max((e.seq for e in self.events()), default=0) if self._on_event else 0
        steps, stood = 0, self.now
        while max_steps is None or steps < max_steps:
            moved = self.step(ticks=1) if self._on_tick else self.step(finest, all=True)
            if moved is None or (self._on_tick and moved == stood):
                break  # the end of its days: a close that never comes, or a tick that moves nothing
            stood = self.now
            steps += 1
            for sym, fn in self._on_tick:
                q = self.quote(sym)
                if q is not None:
                    fn(q)
            # A new close is found once per interval and market, then handed to every handler of it, in order.
            fresh = {}
            for key in seen_close:
                last = self._last_close(*key)
                if last is not None and last.time != (seen_close[key] or {}).get("time"):
                    seen_close[key] = fresh[key] = last
            for iv, sym, fn in closes:
                if (iv, sym) in fresh:
                    fn(fresh[(iv, sym)])
            if self._on_event:
                # From the last event handed on, not the log's start: a run of ten thousand steps read it whole at every one.
                for e in self.events(since=seen_event):
                    if e.seq > seen_event:
                        seen_event = e.seq
                        for kind, fn in self._on_event:
                            if kind in ("*", e.type):
                                fn(e)
            if until is not None and self.now >= until:
                break
        if strategy is not None:
            strategy.on_end(self)
        return steps

    def _last_close(self, interval: str, symbol: Optional[str]) -> Optional[Candle]:
        rows = self.window([interval], 1, symbol)[interval].rows
        return Candle(zip(Bars.COLUMNS, rows[-1])) if rows else None

    def _engine(self, what: str) -> Any:
        engine = self.bx.engine
        if engine is None:
            raise Error(f"{what} keeps the state in this process: it is the engine's (bentrix.open), not the application's - save and load are both's")
        return engine
