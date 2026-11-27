"""**Bentrix for a program**: the engine in this process, or the Bentrix open on this computer, as one object.

    import bentrix
    bx = bentrix.open()            # the engine in this process: your workspace (bentrix.open("folder"), bentrix.open(bentrix.practice()))
    bx = bentrix.connect()         # the Bentrix open on this computer, paired with nothing typed
    bench = bx.bench()             # a workbench: bench.buy(0.1), bench.step("H1"), bench.positions, bench.account
    bx.tools.trade.order_market(bench.uri, "XAUUSD", "buy", 0.1)   # the generated typed methods, by domain

``bentrix.core`` is the object model (``Bentrix``, ``Bench``, ``Answer``); ``_generated.py`` the typed methods from
the application's capability catalogue; ``_runtime.py`` the connection; ``bentrix.pairing`` finding the application.
Every error the SDK raises is a ``bentrix.Error``: a tool's refusal a ``ToolError`` through either door.
"""

from pathlib import Path
from typing import Any, Optional, Union

from ._discovery import (
    ENDPOINT_FILE,
    SECRET_FILE,
    NotRunning,
    PairingRefused,
    agents_dir,
    default_name,
    discover,
    pair,
    read_endpoint,
    read_secret,
)
from ._generated import CATALOGUE_DIGEST, CATALOGUE_VERSION, Bentrix as _Client
# The object model over either door.
from .core import Account, Answer, Bench, Bentrix, Candle, Deal, Error, Event, Exposure, Held, NotSignedIn, Order, Position, Quote, Refused, iso, ms
from .events import Events
# `@bentrix.model`: a model file is the same code in the full SDK as under Bentrix's model host.
from .model import model, export_manifest, OnnxModel, Model as ModelFile
# A model's outputs of any kind: the builders and the contract's refusal.
from .model import Output, OutputRefused
# The practice tape: made on this computer, no download (the engine library is loaded only when it is made).
from .practice import Practice, practice
# A refused order in one line: its words, what to change, the Docs' page.
from .refusals import explain
from ._runtime import (
    BentrixError,
    IncompatibleCatalogue,
    InputRequired,
    Job,
    JobFailed,
    RpcError,
    ToolError,
)

# The first ten minutes' surface: the two doors, a workbench, an answer, the errors; the rest by its module
# (``bentrix.pairing``, ``bentrix.model``, ``bentrix.gym``, ...). The names below it stay importable.
__all__ = [
    "open", "connect", "practice", "Practice", "Bentrix", "Bench", "Answer", "ms", "iso", "explain", "model", "Events",
    "Position", "Order", "Account", "Deal", "Event", "Quote", "Candle", "Exposure",
    "Error", "Refused", "Held", "NotSignedIn", "ToolError", "RpcError", "NotRunning", "PairingRefused", "IncompatibleCatalogue",
    "InputRequired", "JobFailed", "Job", "CATALOGUE_VERSION",
]


def open(where: Any = None, store: Optional[Union[str, Path]] = None, *, raise_refusals: bool = False) -> Bentrix:  # noqa: A001
    """**The engine in this process** (``bentrix.local.Local``): your own workspace when ``where`` is ``None``; a
    ``Practice`` (``bentrix.practice()``); a folder - a practice tape's (its ``ws`` and ``store`` inside) or a
    workspace (its recordings in ``store``, else yours). Refused while a window or another program holds the
    workspace. ``raise_refusals`` raises ``Refused`` for an order the venue refuses instead of answering it."""
    from .core import _EngineDoor
    from .local import Local

    if isinstance(where, Practice):
        ws, st = where.workspace, where.store
    elif where is None:
        ws, st = None, store
    else:
        folder = Path(where)
        if (folder / "ws").is_dir() and (folder / "store").is_dir() and store is None:
            ws, st = str(folder / "ws"), str(folder / "store")
        else:
            ws, st = str(folder), store
    try:
        local = Local(workspace=ws, store=None if st is None else str(st))
    except RuntimeError as e:
        # The engine says why beside its words (`bx_last_error_kind`); its sentence is never parsed.
        raise {"held": Held, "package": NotSignedIn}.get(getattr(e, "kind", ""), Error)(str(e)) from None
    return Bentrix(_EngineDoor(local), raise_refusals=raise_refusals)


def connect(
    url: Optional[str] = None,
    token: Optional[str] = None,
    *,
    agents_folder: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Bentrix:
    """Open a connection.

    * ``connect(url, token)`` uses exactly what it is given and reads no file.
    * ``connect()`` finds the running application through ``endpoint.json`` in Bentrix's agents
      folder (``agents_folder`` overrides where), pairs with it using ``local.secret`` under ``name``
      (default: ``BENTRIX_AGENT_NAME``, else this script's name) and connects with the token it gets.
      It raises :class:`NotRunning` when there is no live application and :class:`PairingRefused`
      when the application says no.
    * ``connect(url)`` pairs with that endpoint; ``connect(token=...)`` finds the endpoint only.
    """
    folder = Path(agents_folder) if agents_folder is not None else None
    if token is None and url is None:
        url, token = discover(folder, name)
    elif token is None:
        token = pair(url, read_secret(folder or agents_dir()), name or default_name())
    elif url is None:
        url = read_endpoint(folder or agents_dir())["url"]
    from .core import _AppDoor

    raise_refusals = bool(kwargs.pop("raise_refusals", False))
    return Bentrix(_AppDoor(_Client(url, token, **kwargs)), raise_refusals=raise_refusals)
