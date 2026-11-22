"""The typed Python client of a running BENTRIX.

    import bentrix
    b = bentrix.connect()          # finds the running application and pairs with it
    state = b.workspace.get_state()

Everything under ``b`` is generated from the application's capability catalogue (``_generated.py``);
the connection, pagination and the version refusal are ``_runtime.py``; finding the application and
getting a token is ``_discovery.py``.
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
from ._generated import CATALOGUE_DIGEST, CATALOGUE_VERSION, Bentrix
from .events import Events
# `@bentrix.model`: a model file is the same code in the full SDK as under Bentrix's model host.
from .model import model, export_manifest, OnnxModel, Model as ModelFile
# A model's outputs of any kind: the builders and the contract's refusal.
from .model import Output, OutputRefused
from ._runtime import (
    BentrixError,
    IncompatibleCatalogue,
    InputRequired,
    Job,
    JobFailed,
    RpcError,
    ToolError,
)

__all__ = [
    "connect", "discover", "pair", "agents_dir", "Bentrix", "Events", "CATALOGUE_VERSION", "CATALOGUE_DIGEST",
    "BentrixError", "IncompatibleCatalogue", "InputRequired", "Job", "JobFailed", "NotRunning",
    "PairingRefused", "RpcError", "ToolError", "ENDPOINT_FILE", "SECRET_FILE", "model", "export_manifest", "OnnxModel", "ModelFile",
    "Output", "OutputRefused",
]


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
    * ``connect()`` finds the running application through ``endpoint.json`` in BENTRIX's agents
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
    return Bentrix(url, token, **kwargs)
