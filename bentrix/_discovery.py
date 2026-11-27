"""Finding a running Bentrix and getting a token from it.

Bentrix writes two files into its agents folder at every start: ``endpoint.json`` (``{url, pid,
catalogue_version}``, no secret) and ``local.secret`` (this run's same-user secret, readable by the
person alone). A process that can read the second runs as that person, which is the only claim the
secret makes; it presents it once to ``<url>/pair`` with its own name and is answered a token of its
own. From then on the token is a bearer, and the secret is not used again.

Nothing here writes a file, keeps the secret, or puts it in a message: an error names the *file*
and what is wrong with it, never what it held.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ._runtime import BentrixError

ENDPOINT_FILE = "endpoint.json"
SECRET_FILE = "local.secret"
#: Used when no name is given and none is in the environment.
DEFAULT_CLIENT = "python sdk"


class NotRunning(BentrixError):
    """Bentrix is not running (or not reachable): there is nothing to connect to."""


class PairingRefused(BentrixError):
    """Bentrix answered the pairing request and said no (or could not do it)."""


def agents_dir() -> Path:
    """The folder Bentrix writes its endpoint and secret into.

    This mirrors ``bentrix_core::paths::user_dir(StoreKind::Data, "agents")`` and nothing else:
    ``%LOCALAPPDATA%\\BENTRIX\\agents`` when ``LOCALAPPDATA`` is set and non-empty; otherwise
    ``$HOME/.local/share/bentrix/agents`` (the same string on Linux and macOS: that function has no
    ``Library`` branch); otherwise the relative ``bentrix-agents``. It is deliberately not "what the
    platform's convention would be".
    """
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        return Path(local) / "BENTRIX" / "agents"
    home = os.environ.get("HOME", "")
    if home:
        return Path(home) / ".local" / "share" / "bentrix" / "agents"
    return Path("bentrix-agents")


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists. (Windows: ``os.kill(pid, 0)`` would *terminate* it.)"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists, and is not ours
        code = wintypes.DWORD()
        ok = k.GetExitCodeProcess(handle, ctypes.byref(code))
        k.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_endpoint(folder: Path) -> Dict[str, Any]:
    """The running application's endpoint, or :class:`NotRunning` saying why not."""
    path = folder / ENDPOINT_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotRunning(f"Bentrix is not running: there is no {ENDPOINT_FILE} in {folder}") from None
    except (OSError, ValueError) as e:
        raise NotRunning(f"Bentrix is not running: {path} cannot be read ({type(e).__name__})") from None
    url, pid = doc.get("url") if isinstance(doc, dict) else None, doc.get("pid") if isinstance(doc, dict) else None
    if not isinstance(url, str) or not url.startswith("http://127.0.0.1:") or not isinstance(pid, int):
        raise NotRunning(f"Bentrix is not running: {path} is not an endpoint file")
    if not pid_alive(pid):
        raise NotRunning(f"Bentrix is not running: {path} is stale (process {pid} has ended)")
    return doc


def read_secret(folder: Path) -> str:
    path = folder / SECRET_FILE
    try:
        secret = path.read_text(encoding="utf-8").strip()
    except OSError:
        raise PairingRefused(f"there is no readable {SECRET_FILE} in {folder}: Bentrix writes it at every start") from None
    if not secret:
        raise PairingRefused(f"{path} is empty")
    return secret


def default_name() -> str:
    """The name this process pairs under: ``BENTRIX_AGENT_NAME``, else the script's name.

    A pairing is one row per name and pairing again replaces the token, so two scripts that pair
    under one name would end each other's sessions; the script's name keeps them apart.
    """
    env = os.environ.get("BENTRIX_AGENT_NAME", "").strip()
    if env:
        return env
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return f"{DEFAULT_CLIENT} ({script})" if script and not script.startswith("-") else DEFAULT_CLIENT


def pair(url: str, secret: str, name: str, version: Optional[str] = None, timeout: float = 10.0) -> str:
    """Present the secret to ``<url>/pair`` and return the token.

    The secret travels in the body of this one request, over loopback, and is in no URL, header,
    log line or exception message.
    """
    client: Dict[str, Any] = {"name": name}
    if version:
        client["version"] = version
    body = json.dumps({"secret": secret, "client": client}).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/pair", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        code, message = _error_of(e)
        if code == "wrong_secret":
            raise PairingRefused(
                f"Bentrix refused the secret in the secret file ({message}). It writes a new secret at every start: "
                "the file may be from an earlier run, or from another installation."
            ) from None
        if code == "rate_limited":
            raise PairingRefused(f"too many pairing attempts; wait and try again ({message})") from None
        raise PairingRefused(f"Bentrix answered the pairing request with HTTP {e.code} {code}: {message}") from None
    except (urllib.error.URLError, OSError) as e:
        raise NotRunning(f"Bentrix does not answer at {url} ({type(getattr(e, 'reason', e)).__name__})") from None
    token = doc.get("token") if isinstance(doc, dict) else None
    if not isinstance(token, str) or not token:
        raise PairingRefused("Bentrix answered the pairing request with no token")
    return token


def _error_of(e: urllib.error.HTTPError) -> Tuple[str, str]:
    try:
        err = json.loads(e.read().decode("utf-8")).get("error", {})
        return str(err.get("code", "")), str(err.get("message", ""))
    except (ValueError, AttributeError):
        return "", ""


def discover(folder: Optional[Path] = None, name: Optional[str] = None) -> Tuple[str, str]:
    """Find the running application and get a token: ``(url, token)``."""
    folder = Path(folder) if folder is not None else agents_dir()
    url = read_endpoint(folder)["url"]
    return url, pair(url, read_secret(folder), name or default_name())
