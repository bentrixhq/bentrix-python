"""Run the runtime's tests:   cd sdk/python && python -m pytest   (pytest 7+, nothing else; stdlib only)."""

import importlib.util
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# The SDK generated from the Rust crate's fixture catalogue. Pinned by a snapshot test over there; here it
# is imported as `bentrix._fixture_sdk` so its relative import of `._runtime` resolves exactly as the real
# generated module's does.
SNAPSHOT = ROOT.parent.parent / "crates" / "bentrix-capabilities" / "tests" / "snapshots" / "fixture_sdk.py"


@pytest.fixture(scope="session")
def fx():
    import bentrix  # noqa: F401  (makes `bentrix` the parent package)

    spec = importlib.util.spec_from_file_location("bentrix._fixture_sdk", SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bentrix._fixture_sdk"] = module
    spec.loader.exec_module(module)
    return module


# ---- the CI tape ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def _tape_made(tmp_path_factory):
    native = pytest.importorskip("bentrix._native")
    d = tmp_path_factory.mktemp("tape")
    ws, store = native.make_tape(str(d))
    return d, ws, store


@pytest.fixture(scope="session")
def _tape2_made(tmp_path_factory):
    native = pytest.importorskip("bentrix._native")
    d = tmp_path_factory.mktemp("tape2")
    ws, store = native.make_tape(str(d), also=[("XAGUSD", 0.0125)])
    return d, ws, store


@pytest.fixture(scope="session")
def _tape3w_made(tmp_path_factory):
    native = pytest.importorskip("bentrix._native")
    d = tmp_path_factory.mktemp("tape3w")
    ws, store = native.make_tape(str(d), first="2024-03-04", last="2024-03-22")
    return d, ws, store


@pytest.fixture
def tape3w(_tape3w_made, tmp_path):
    """**The CI tape over three weeks**, Monday 2024-03-04 to Friday 2024-03-22, opened embedded like
    ``tape`` — two whole weeks close inside it."""
    return _open_copy(_tape3w_made, tmp_path)


@pytest.fixture
def tape(_tape_made, tmp_path):
    """**A fresh copy of the CI tape** — the deterministic recording ``bentrix._native.make_tape`` writes
    and a workspace over it — opened embedded; answers ``(local, bench_uri)``. A copy per test, since an
    embedded workspace is held while it is open."""
    return _open_copy(_tape_made, tmp_path)


@pytest.fixture
def tape2(_tape2_made, tmp_path):
    """**The CI tape with a second market**: ``XAUUSD`` and ``XAGUSD`` (the same formula at 0.0125 of
    its level) in one recording, opened embedded like ``tape``."""
    return _open_copy(_tape2_made, tmp_path)


def _open_copy(made, tmp_path):
    import shutil

    from bentrix.local import Local

    src, ws, store = made
    shutil.copytree(src, tmp_path / "tape")
    was, now = str(src), str(tmp_path / "tape")
    for root, _, files in os.walk(tmp_path / "tape" / "ws"):
        for f in files:
            p = os.path.join(root, f)
            try:
                text = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if was in text:
                open(p, "w", encoding="utf-8").write(text.replace(was, now))
    local = Local(workspace=str(tmp_path / "tape" / "ws"), store=str(tmp_path / "tape" / "store"))
    bench = local.call("get_state")["laboratories"][0]["benches"][0]["uri"]
    return local, bench
