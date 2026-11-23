"""The hand-written half of the BENTRIX SDK: the connection and everything that does not change
when a capability is added.

It speaks MCP revision 2026-07-28 over Streamable HTTP with the standard library only: every request is
its own POST carrying the protocol version, the client's identity and its (empty) capabilities in
``_meta``; there is no session and no ``initialize``. The typed surface -- one method per operation,
one reader per resource, one iterator per event, one dataclass per document -- is ``_generated.py``,
written from the capability catalogue; it subclasses :class:`Client` and passes its compiled-in
``CATALOGUE_VERSION`` to it.

Why not the official ``mcp`` package: its client is asynchronous and built around a session, and pulls
an HTTP stack and a validation library; this surface is one synchronous POST per call and a script
should be able to ``pip install`` it with nothing else. (Whether the package speaks 2026-07-28 yet is
UNVERIFIED here; the wire below is the revision's own text.)

The client declares NO optional capability (no elicitation, no tasks), so the server answers with the
tools-only fallbacks: a question for the person is asked in BENTRIX's own window, and a long operation
returns a ``{"job": handle}`` that :class:`Job` polls.
"""

from __future__ import annotations

import base64
import dataclasses
import itertools
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Any, Callable, Dict, Generic, Iterator, Optional, Tuple, TypeVar

PROTOCOL_VERSION = "2026-07-28"
CLIENT_NAME = "bentrix-python-sdk"
CLIENT_VERSION = "0.1.382"

#: **Busy, try again**: the JSON-RPC code Bentrix answers when it refused a request BEFORE doing
#: anything with it — one program's 20 calls a second spent, too many connections or streams, a task
#: it could not read in time. Nothing ran, so the same request is sent again after the wait the
#: answer names (``retryAfterMs``), at most ``BUSY_RETRIES`` times, each wait at most
#: ``BUSY_WAIT_MAX_S``; then the refusal is raised.
BUSY = -40001
BUSY_RETRIES = 10
BUSY_WAIT_MAX_S = 5.0

T = TypeVar("T")

__all__ = [
    "BentrixError", "RpcError", "ToolError", "InputRequired", "IncompatibleCatalogue", "JobFailed",
    "Client", "Job", "plain", "identity", "deprecated", "expand_template",
]


def busy_wait_s(data: Any) -> float:
    """The wait a *busy* answer names (``retryAfterMs``), in seconds: a second when it names none,
    and never more than ``BUSY_WAIT_MAX_S``."""
    ms = data.get("retryAfterMs") if isinstance(data, dict) else None
    ms = ms if isinstance(ms, (int, float)) and ms > 0 else 1000
    return min(ms / 1000.0, BUSY_WAIT_MAX_S)


class BentrixError(Exception):
    """Base class of every error this package raises on purpose."""


class RpcError(BentrixError):
    """A JSON-RPC error: the request itself was refused (unknown tool, bad arguments, no scope...)."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class ToolError(BentrixError):
    """A tool ran and answered ``isError``: the product's own sentence, verbatim. ``machine`` is the
    server's word for the kind (``busy``, ``expired``, ``insufficient_scope``...) when it gave one."""

    def __init__(self, tool: str, message: str, machine: Optional[str] = None) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message
        self.machine = machine


class Rewound(BentrixError):
    """**A stream's clock went back behind the cursor** (a laboratory's rewind): what came after it
    will come again. ``watch`` ends with this rather than handing those items a second time as if new;
    ``cursor`` is where to read on from (``after=``)."""

    def __init__(self, uri: str, cursor: Optional[str]) -> None:
        super().__init__(f"{uri}: the workbench's clock went back; read on from {cursor}")
        self.uri = uri
        self.cursor = cursor


# A stream's call answered with one of these is asked again, after a pause; the next answer decides.
_PASSING = ("busy", "expired")
_PASSING_TRIES = 5
_PASSING_PAUSE_S = 0.5


class InputRequired(BentrixError):
    """The server asked for input this client cannot give (it declared no elicitation)."""


class IncompatibleCatalogue(BentrixError):
    """The server's catalogue is on another major line than the one this SDK was generated from."""


class JobFailed(BentrixError):
    """A job ended failed or cancelled."""


def identity(x: T) -> T:
    return x


def deprecated(name: str, since: str) -> None:
    warnings.warn(
        f"{name} is deprecated since catalogue {since}; see its description for what replaces it",
        DeprecationWarning,
        stacklevel=3,
    )


def plain(value: Any) -> Any:
    """Turn a typed value into the JSON the wire carries."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def expand_template(template: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]] = None) -> str:
    """Expand the RFC 6570 subset the catalogue uses: ``{name}`` and ``{?a,b}`` / ``{&a,b}``.

    A ``None`` optional is left out. A path variable that is missing is an error here, not a bad URI
    on the server.
    """
    values = dict(required)
    values.update({k: v for k, v in (optional or {}).items() if v is not None})
    out = []
    i = 0
    while i < len(template):
        if template[i] != "{":
            out.append(template[i])
            i += 1
            continue
        j = template.index("}", i)
        expr = template[i + 1 : j]
        i = j + 1
        if expr[:1] in ("?", "&"):
            pairs = [
                f"{n}={urllib.parse.quote(_scalar(values[n]), safe='')}" for n in expr[1:].split(",") if n in values
            ]
            if pairs:
                out.append(expr[0] + "&".join(pairs))
        else:
            if expr not in values:
                raise ValueError(f"the URI template {template!r} needs {expr!r}")
            out.append(urllib.parse.quote(_scalar(values[expr]), safe=""))
    return "".join(out)


def _header_value(v: str) -> str:
    """A header value, or the standard Base64 sentinel when plain ASCII cannot carry it."""
    safe = (
        all(0x20 <= ord(c) <= 0x7E for c in v)
        and v == v.strip()
        and not (v.startswith("=?base64?") and v.endswith("?="))
    )
    if safe:
        return v
    return "=?base64?" + base64.b64encode(v.encode("utf-8")).decode("ascii") + "?="


def _parse_version(text: str) -> Tuple[int, int, int]:
    parts = text.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"{text!r} is not a version major.minor.patch")
    return int(parts[0]), int(parts[1]), int(parts[2])


class Job(Generic[T]):
    """A long operation. ``wait()`` returns the typed result; the person may have to act in BENTRIX's
    own window first (the job then reports ``awaiting_person``)."""

    def __init__(self, client: "Client", handle: Optional[str], parse: Callable[[Any], T], result: Any = None) -> None:
        self._c = client
        self.handle = handle
        self._parse = parse
        self._result = result
        self._done = handle is None

    def status(self) -> Dict[str, Any]:
        if self.handle is None:
            return {"status": "completed"}
        return self._c.call("job_get", {"job": self.handle})

    def cancel(self) -> None:
        if self.handle is not None:
            self._c.call("job_cancel", {"job": self.handle})

    def wait(self, timeout: Optional[float] = None, poll: float = 0.5) -> T:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._done:
            st = self.status()
            state = st.get("status")
            if state == "completed":
                self._result = self._parse(st.get("result", {}))
                self._done = True
            elif state in ("failed", "cancelled"):
                raise JobFailed(f"job {self.handle} {state}: {st.get('message', '')}")
            else:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"job {self.handle} is still {state}")
                time.sleep(poll)
        return self._result


# **A call that asks the server to wait is given that long, and 15 s more** (found 2026-11-02, LiveEnv
# on a live paper workbench): `wait_for` with `timeout_ms` beyond the client's own timeout ended as a
# socket timeout here before the server could answer.
LONG_POLL_MARGIN_S = 15.0


def _request_timeout(base: float, args: Dict[str, Any]) -> float:
    wait = args.get("timeout_ms")
    return max(base, wait / 1000.0 + LONG_POLL_MARGIN_S) if isinstance(wait, (int, float)) and not isinstance(wait, bool) else base


class Client:
    """One Bentrix endpoint. Subclassed by the generated ``Bentrix``."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        catalogue_version: str,
        timeout: float = 60.0,
        check_version: bool = True,
    ) -> None:
        self.url = url
        self._token = token
        self._timeout = timeout
        self.catalogue_version = catalogue_version
        self._ids = itertools.count(1)
        self._server_catalogue: Optional[str] = None
        if check_version:
            self.verify_catalogue()

    def __repr__(self) -> str:  # the token is a bearer credential
        return f"<{type(self).__name__} {self.url} catalogue {self.catalogue_version}>"

    def close(self) -> None:
        """Nothing is held open between calls; kept so a client can be used in ``with``."""

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- the wire --------------------------------------------------------------------------------

    def _rpc(self, method: str, params: Dict[str, Any], name: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """One request, sent again while Bentrix answers *busy* (``BUSY``): it ran nothing, so an order
        is never sent twice by this."""
        for attempt in range(BUSY_RETRIES + 1):
            try:
                return self._rpc_once(method, params, name, timeout)
            except RpcError as e:
                if e.code != BUSY or attempt == BUSY_RETRIES:
                    raise
                time.sleep(busy_wait_s(e.data))
        raise AssertionError("unreachable")

    def _rpc_once(self, method: str, params: Dict[str, Any], name: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        params = dict(params)
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        rid = next(self._ids)
        body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": method,
            "Authorization": f"Bearer {self._token}",
        }
        if name is not None:
            headers["Mcp-Name"] = _header_value(name)
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout or self._timeout)
        except urllib.error.HTTPError as e:
            raw = e.read()
            raise self._error_from(raw, e.code) from None
        with resp:
            ctype = resp.headers.get("Content-Type", "")
            if ctype.startswith("text/event-stream"):
                message = self._read_sse(resp, rid)
            else:
                message = json.loads(resp.read().decode("utf-8"))
        if "error" in message:
            err = message["error"]
            raise RpcError(err.get("code", 0), err.get("message", ""), err.get("data"))
        return message["result"]

    @staticmethod
    def _error_from(raw: bytes, status: int) -> BentrixError:
        try:
            err = json.loads(raw.decode("utf-8")).get("error")
        except (ValueError, AttributeError):
            err = None
        if isinstance(err, dict):
            return RpcError(err.get("code", status), err.get("message", ""), err.get("data"))
        return RpcError(status, f"HTTP {status} with no JSON-RPC error body")

    @staticmethod
    def _read_sse(resp: Any, rid: int) -> Dict[str, Any]:
        """The final response on a request-scoped stream; notifications before it are ignored."""
        data: list = []
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8").rstrip("\r\n")
            if text == "":
                if data:
                    try:
                        msg = json.loads("\n".join(data))
                    except ValueError:
                        msg = None
                    data = []
                    if isinstance(msg, dict) and msg.get("id") == rid and ("result" in msg or "error" in msg):
                        return msg
            elif text.startswith(":"):
                continue  # a keep-alive comment
            elif text.startswith("data:"):
                data.append(text[5:].lstrip(" "))
        raise BentrixError("the stream ended without a response")

    # --- tools -----------------------------------------------------------------------------------

    def call(self, name: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool; ``None`` arguments are left out so the server's own default applies."""
        args = {k: plain(v) for k, v in {**required, **(optional or {})}.items() if v is not None}
        res = self._rpc("tools/call", {"name": name, "arguments": args}, name=name, timeout=_request_timeout(self._timeout, args))
        if res.get("resultType") == "input_required":
            raise InputRequired(f"{name} needs input this client cannot give; answer it in Bentrix's window")
        text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("isError"):
            raise ToolError(name, text or "the tool failed", (res.get("_meta") or {}).get("bentrix/machine"))
        structured = res.get("structuredContent")
        if structured is None and text:
            try:
                structured = json.loads(text)
            except ValueError:
                structured = None
        return structured if structured is not None else {}

    def call_job(
        self,
        name: str,
        required: Dict[str, Any],
        optional: Optional[Dict[str, Any]],
        parse: Callable[[Any], T],
    ) -> "Job[T]":
        raw = self.call(name, required, optional)
        handle = raw.get("job") if set(raw) == {"job"} else None
        if isinstance(handle, str):
            return Job(self, handle, parse)
        return Job(self, None, parse, result=parse(raw))

    # --- resources -------------------------------------------------------------------------------

    def read_resource(
        self, template: str, required: Dict[str, Any], optional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._read_uri(expand_template(template, required, optional))

    def _read_uri(self, uri: str) -> Dict[str, Any]:
        res = self._rpc("resources/read", {"uri": uri}, name=uri)
        contents = res.get("contents") or []
        if not contents or "text" not in contents[0]:
            raise BentrixError(f"{uri} answered no text content")
        return json.loads(contents[0]["text"])

    def read_pages(
        self,
        template: str,
        required: Dict[str, Any],
        optional: Optional[Dict[str, Any]],
        parse: Callable[[Any], T],
    ) -> Iterator[T]:
        """Every page of a cursor-paged resource: ``next`` is the URI of the following page."""
        uri: Optional[str] = expand_template(template, required, optional)
        while uri:
            page = self._read_uri(uri)
            yield parse(page)
            uri = page.get("next")
            if uri is not None and "://" not in uri:
                raise BentrixError(f"`next` should be a URI, got {uri!r}")

    def watch(
        self,
        template: str,
        required: Dict[str, Any],
        optional: Optional[Dict[str, Any]],
        parse: Callable[[Any], T],
        after: Optional[str],
        timeout_ms: int,
    ) -> Iterator[T]:
        """Items of a stream, forever: ``wait_for`` long-polls and a page is ``{items, next}``.

        A call answered ``busy`` or ``expired`` is asked again (five times, half a second apart) rather
        than ending the stream; a page saying ``rewound`` ends it with :class:`Rewound`, carrying the
        cursor to read on from, so nothing is handed twice unsaid (the strict audit of 2026-11-04, S3)."""
        uri = expand_template(template, required, optional)
        cursor = after
        while True:
            for attempt in range(_PASSING_TRIES):
                try:
                    page = self.call("wait_for", {"uri": uri, "timeout_ms": timeout_ms}, {"after": cursor})
                    break
                except ToolError as e:
                    if e.machine not in _PASSING or attempt == _PASSING_TRIES - 1:
                        raise
                    time.sleep(_PASSING_PAUSE_S)
            for item in page.get("items", []):
                yield parse(item)
            if page.get("next") is not None:
                cursor = page["next"]
            if page.get("rewound"):
                raise Rewound(uri, cursor)

    # --- the catalogue contract ------------------------------------------------------------------

    def server_catalogue_version(self) -> str:
        """The catalogue version the server reports: ``server/discover``'s ``_meta``, else the
        ``agent_whoami`` tool."""
        if self._server_catalogue is None:
            res = self._rpc("server/discover", {})
            version = (res.get("_meta") or {}).get("bentrix/catalogue")
            if version is None:
                version = self.call("agent_whoami", {}).get("catalogue_version")
            if not isinstance(version, str):
                raise IncompatibleCatalogue("the server does not report its catalogue version")
            self._server_catalogue = version
        return self._server_catalogue

    def verify_catalogue(self) -> None:
        """Refuse a server on another major line; warn about one that is a minor ahead."""
        server = self.server_catalogue_version()
        try:
            s, mine = _parse_version(server), _parse_version(self.catalogue_version)
        except ValueError as e:
            raise IncompatibleCatalogue(str(e)) from None
        if s[0] != mine[0]:
            raise IncompatibleCatalogue(
                f"this SDK was generated from catalogue {self.catalogue_version}; the server's is {server}. "
                "Their major lines differ: install the SDK that matches the application."
            )
        if s[1] > mine[1]:
            warnings.warn(
                f"the server's catalogue {server} is newer than this SDK's {self.catalogue_version}; "
                "new capabilities are not in this SDK until it is updated",
                RuntimeWarning,
                stacklevel=3,
            )
