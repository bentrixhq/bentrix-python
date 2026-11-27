"""**Finding and pairing with the Bentrix open on this computer**, for a program that does it itself:
``bentrix.connect()`` does all of this, and a newcomer never needs it.

``discover`` finds the application's endpoint and pairs; ``pair`` pairs with an endpoint already known;
``agents_dir`` is where Bentrix keeps its agents' files, ``read_endpoint`` and ``read_secret`` read two of them.
"""

from ._discovery import (  # noqa: F401
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
from ._generated import CATALOGUE_DIGEST, CATALOGUE_VERSION  # noqa: F401

__all__ = [
    "discover", "pair", "agents_dir", "read_endpoint", "read_secret", "default_name", "ENDPOINT_FILE", "SECRET_FILE",
    "NotRunning", "PairingRefused", "CATALOGUE_VERSION", "CATALOGUE_DIGEST",
]
