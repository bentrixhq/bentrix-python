"""**Bentrix's engine in this Python process, through its C library** (``libbentrix``).

The engine is Bentrix's own and ships compiled with the application (Plus and Pro): ``libbentrix.dll`` in Bentrix's
folder on Windows. This module finds it and answers the same calls the package's ``Local``, ``backtest``, ``data``
and ``gym`` make, through the library's C functions (``bentrix.h``) and nothing else — no compiler, no dependency.

Where the library is looked for, in order:

1. the file ``BENTRIX_FFI`` names;
2. ``libbentrix`` in the folder ``BENTRIX_FFI_DIR`` names;
3. on Windows, Bentrix's own folder: ``%LOCALAPPDATA%\\Programs\\BENTRIX``;
4. the system's library path.

Where none is found, importing this module raises ``ImportError`` saying so, and the client of a running Bentrix
(``bentrix.connect``) works as it always does.

A session is used from the thread that opened it. A refusal is raised as ``RuntimeError`` in Bentrix's own words; an
argument that is not JSON as ``ValueError``.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

_NAME = {"win32": "libbentrix.dll", "darwin": "libbentrix.dylib"}.get(sys.platform, "libbentrix.so")


def _candidates() -> Iterator[str]:
    file = os.environ.get("BENTRIX_FFI", "").strip()
    if file:
        yield file
    folder = os.environ.get("BENTRIX_FFI_DIR", "").strip()
    if folder:
        yield str(Path(folder) / _NAME)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            yield str(Path(local) / "Programs" / "BENTRIX" / _NAME)
    yield _NAME


def _load() -> ctypes.CDLL:
    tried = []
    for path in _candidates():
        if os.path.sep in path and not os.path.isfile(path):
            tried.append(path)
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as e:
            tried.append(f"{path} ({e})")
    raise ImportError(
        "Bentrix's engine library (" + _NAME + ") was not found. It is installed with Bentrix; "
        "or set BENTRIX_FFI to the file, or BENTRIX_FFI_DIR to its folder. Looked at: " + "; ".join(tried)
    )


_lib = _load()

_c = ctypes.c_char_p
_vp = ctypes.c_void_p
_i64 = ctypes.c_int64


def _sig(name: str, restype: Any, *argtypes: Any) -> None:
    f = getattr(_lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)


# The library hands out text it owns until ``bx_free``: those functions answer a plain pointer, read and freed here.
_sig("bx_version", _c)
_sig("bx_last_error", _c)
_sig("bx_free", None, _vp)
_sig("bx_open", _vp, _c, _c)
_sig("bx_close", None, _vp)
_sig("bx_call", _vp, _vp, _c, _c)
_sig("bx_read", _vp, _vp, _c, _c)
_sig("bx_arm", ctypes.c_int, _vp, _c, ctypes.c_size_t)
_sig("bx_advance", ctypes.c_int, _vp, _c, _i64, ctypes.POINTER(_i64))
_sig("bx_frame", ctypes.c_int, _vp)
_sig("bx_settle", ctypes.c_int, _vp, ctypes.c_size_t)
_sig("bx_interval_ms", _i64, _c)
_sig("bx_candle_span", ctypes.c_int, _vp, _c, _c, _i64, ctypes.POINTER(_i64), ctypes.POINTER(_i64))
_sig("bx_next_close", _vp, _vp, _c, _c, _c, _i64, ctypes.c_int)
_sig("bx_run_until", _vp, _vp, _c, _c, _c, ctypes.c_size_t, _i64, ctypes.c_int)
_sig("bx_snapshot", ctypes.c_int, _vp, _c, ctypes.POINTER(ctypes.c_uint64))
_sig("bx_restore", ctypes.c_int, _vp, _c, ctypes.c_uint64, ctypes.POINTER(_i64))
_sig("bx_make_markets", ctypes.c_int, _c, _c, _c, _c, ctypes.POINTER(_vp), ctypes.POINTER(_vp))


def _b(s: Optional[str]) -> Optional[bytes]:
    return None if s is None else s.encode("utf-8")


def _why() -> str:
    w = _lib.bx_last_error()
    return w.decode("utf-8", "replace") if w else "Bentrix answered with no words"


def _text(p: Optional[int]) -> str:
    """An owned answer read and handed back to the library; a NULL raised in Bentrix's words."""
    if not p:
        raise RuntimeError(_why())
    try:
        return ctypes.cast(p, _c).value.decode("utf-8")
    finally:
        _lib.bx_free(p)


def _ok(code: int) -> None:
    if code != 0:
        raise RuntimeError(_why())


def _json(text: str, what: str) -> None:
    try:
        json.loads(text)
    except ValueError as e:
        raise ValueError(f"{what} are not JSON: {e}") from None


Bars = List[Tuple[str, List[float]]]
Close = Tuple[int, int, Bars]


def _close(c: dict) -> Close:
    return (int(c["reached"]), int(c["opened"]), [(str(s), [float(x) for x in r]) for s, r in c["bars"]])


class Session:
    """An embedded Bentrix over ``workspace`` (yours when ``None``) and its recordings."""

    def __init__(self, workspace: Optional[str] = None, store: Optional[str] = None):
        s = _lib.bx_open(_b(workspace), _b(store))
        if not s:
            raise RuntimeError(_why())
        self._s: Optional[int] = s

    def _open(self) -> int:
        if not self._s:
            raise RuntimeError("this session is closed")
        return self._s

    def call(self, tool: str, args: str) -> str:
        _json(args, "the arguments")
        return _text(_lib.bx_call(self._open(), _b(tool), _b(args)))

    def read(self, resource: str, vars: str) -> str:
        _json(vars, "the variables")
        return _text(_lib.bx_read(self._open(), _b(resource), _b(vars)))

    def arm(self, bench: str, max_frames: int = 20000) -> None:
        _ok(_lib.bx_arm(self._open(), _b(bench), max_frames))

    def advance(self, bench: str, to_ms: int) -> int:
        reached = _i64(0)
        _ok(_lib.bx_advance(self._open(), _b(bench), to_ms, ctypes.byref(reached)))
        return reached.value

    def candle_span(self, bench: str, interval: str, at_ms: int) -> Tuple[int, int]:
        opens, closes = _i64(0), _i64(0)
        _ok(_lib.bx_candle_span(self._open(), _b(bench), _b(interval), at_ms, ctypes.byref(opens), ctypes.byref(closes)))
        return (opens.value, closes.value)

    def next_close(self, bench: str, interval: str, symbols: Sequence[str], end_ms: Optional[int] = None) -> Optional[Close]:
        out = json.loads(_text(_lib.bx_next_close(self._open(), _b(bench), _b(interval), _b(json.dumps(list(symbols))), end_ms or 0, int(end_ms is not None))))
        return None if out is None else _close(out)

    def run_until(self, bench: str, interval: str, symbols: Sequence[str], n: int, end_ms: Optional[int] = None) -> List[Close]:
        out = json.loads(_text(_lib.bx_run_until(self._open(), _b(bench), _b(interval), _b(json.dumps(list(symbols))), n, end_ms or 0, int(end_ms is not None))))
        return [_close(c) for c in out]

    def snapshot(self, bench: str) -> int:
        n = ctypes.c_uint64(0)
        _ok(_lib.bx_snapshot(self._open(), _b(bench), ctypes.byref(n)))
        return n.value

    def restore(self, bench: str, snapshot: int) -> int:
        reached = _i64(0)
        _ok(_lib.bx_restore(self._open(), _b(bench), snapshot, ctypes.byref(reached)))
        return reached.value

    def frame(self) -> bool:
        w = _lib.bx_frame(self._open())
        if w < 0:
            raise RuntimeError(_why())
        return w == 1

    def settle(self, max_frames: int = 1000) -> None:
        _ok(_lib.bx_settle(self._open(), max_frames))

    def close(self) -> None:
        if self._s:
            _lib.bx_close(self._s)
            self._s = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build() -> str:
    """The build the library was made from (``0.1.380+<commit>``)."""
    return _lib.bx_version().decode("utf-8")


def interval_ms(label: str) -> Optional[int]:
    """An interval's fixed length in ms (``H1`` → 3,600,000); ``None`` for one with none."""
    n = _lib.bx_interval_ms(_b(label))
    return n or None


def make_tape(directory: str, first: str = "2024-03-04", last: str = "2024-03-08", symbol: str = "XAUUSD",
              also: Sequence[Tuple[str, float]] = ()) -> Tuple[str, str]:
    """**The CI tape** in ``directory``: a small, deterministic recording of ``symbol`` (and ``also``, each with its
    scale) from ``first`` to ``last``, and a workspace over it — its two paths."""
    ws, store = _vp(), _vp()
    markets = json.dumps([[symbol, 1.0]] + [[s, float(k)] for s, k in also])
    _ok(_lib.bx_make_markets(_b(directory), _b(first), _b(last), _b(markets), ctypes.byref(ws), ctypes.byref(store)))
    return (_text(ws.value), _text(store.value))
