"""**The practice tape**: a small recording to learn on, made on your own computer.

    import bentrix
    p = bentrix.practice()                     # a fresh folder of its own
    with p.open() as local:                    # the engine in this process, over it
        bench = p.bench(local)

The tape is deterministic and nobody's prices: a market moving by a fixed formula, the same on
every computer, so an example run anywhere makes the same trades. Its markets are named as real
ones (``XAUUSD`` at the scale of gold); a second market is the same formula at its own scale.
Nothing is downloaded and nothing of yours is touched - it is written only into ``folder``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union


class Practice(NamedTuple):
    """Where a practice tape was written: its folder, and the workspace and the store inside it."""

    folder: str
    workspace: str
    store: str

    def open(self) -> Any:
        """The engine in this process over the tape (``bentrix.local.Local``); close it, or use ``with``."""
        from bentrix.local import Local

        return Local(workspace=self.workspace, store=self.store)

    def bench(self, local: Any) -> str:
        """The tape's workbench, as ``local`` lists it."""
        for lab in local.call("get_state").get("laboratories", []):
            for b in lab.get("benches", []):
                return b["uri"]
        raise RuntimeError("the practice workspace holds no workbench")


Market = Union[str, Tuple[str, float]]


def practice(
    folder: Optional[str] = None,
    markets: Sequence[Market] = ("XAUUSD",),
    first: str = "2024-03-04",
    last: str = "2024-03-08",
) -> Practice:
    """**A practice tape in ``folder``** (a fresh temporary folder of its own when ``None``): ``markets`` each
    a name or ``(name, scale)`` - the first the main market, a plain name at scale 1 - over the days
    ``first`` to ``last`` (``YYYY-MM-DD``, UTC). Answers its folder, workspace and store."""
    from bentrix import _native

    if not markets:
        raise ValueError("a practice tape has at least one market")
    pairs = [(m, 1.0) if isinstance(m, str) else (str(m[0]), float(m[1])) for m in markets]
    where = folder if folder is not None else tempfile.mkdtemp(prefix="bentrix-practice-")
    os.makedirs(where, exist_ok=True)
    (main, scale), also = pairs[0], pairs[1:]
    if scale != 1.0:
        raise ValueError(f"the main market {main} is at scale 1; a second market takes its own scale")
    try:
        ws, store = _native.make_tape(where, first, last, main, also)
    except _native.OpenRefused as e:
        # **The engine's refusal in the package's error tree**: the first
        # line of every example raised a bare RuntimeError where ``bentrix.open`` raises ``NotSignedIn``.
        from bentrix.core import Error, Held, NotSignedIn

        raise {"held": Held, "package": NotSignedIn}.get(e.kind, Error)(str(e)) from None
    return Practice(where, ws, store)
