"""**Many environments at once**: a Gymnasium vector environment whose workers are processes.

    from bentrix.pool import pool
    from bentrix.gym import TradingEnv

    def make(local, index):                      # at module level: each worker imports it
        return TradingEnv(BENCH, local=local, window=8)

    envs = pool(make, workers=8, workspace=WS, store=STORE)
    obs, info = envs.reset(seed=[1, 2, 3, 4, 5, 6, 7, 8])
    ...
    envs.close()                                 # every engine closed, every copy removed

Each worker is a process with its own engine (``bentrix.local.Local``) over **a copy of the workspace
of its own**, every copy over the one store - a workspace has one writer, so eight engines need eight
workspaces, and the recordings, which nothing writes, are shared. The copies are made while nothing
holds the workspace (a window or another program holding it is refused, by the engine's own lock),
in a temporary folder, and removed on ``close``. What ``env_factory(local, index)`` makes in a
worker is the environment that worker steps; the pool is Gymnasium's ``AsyncVectorEnv`` over them,
so it speaks Gymnasium's ``VectorEnv`` - batched spaces, ``reset(seed=[...])``, ``step(actions)``,
autoreset - for SB3, CleanRL or a loop of your own.

``env_factory`` must be importable by name (defined at a module's top level), because a worker is
started fresh and imports it; a lambda or a nested function cannot cross into a process.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Callable, List, Optional

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv


class _Owned(gym.Wrapper):
    """A worker's environment, closing its engine when it closes."""

    def __init__(self, env: gym.Env, local: Any):
        super().__init__(env)
        self._local = local

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._local.close()


class _Maker:
    """What a worker runs to make its environment: its own engine over its own copy, then the factory's environment."""

    def __init__(self, factory: Callable[[Any, int], gym.Env], workspace: str, store: Optional[str], index: int, bench: Optional[str] = None):
        self.factory, self.workspace, self.store, self.index, self.bench = factory, workspace, store, index, bench

    def __call__(self) -> gym.Env:
        from bentrix.local import Local

        local = Local(workspace=self.workspace, store=self.store)
        try:
            if self.bench is not None:
                from bentrix.core import Bentrix, _EngineDoor

                return _Owned(self.factory(Bentrix(_EngineDoor(local)).bench(self.bench), self.index), local)
            return _Owned(self.factory(local, self.index), local)
        except BaseException:
            local.close()
            raise


class Pool(AsyncVectorEnv):
    """Gymnasium's ``AsyncVectorEnv`` over the workers (module doc), removing the workspaces' copies on close."""

    def __init__(self, env_fns: List[_Maker], folder: str, **kwargs: Any):
        self.copies_folder = folder
        self.workspaces = [m.workspace for m in env_fns]
        try:
            super().__init__(env_fns, **kwargs)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def close_extras(self, timeout: Optional[float] = None, terminate: bool = False) -> None:
        try:
            super().close_extras(timeout=timeout, terminate=terminate)
        finally:
            shutil.rmtree(self.copies_folder, ignore_errors=True)


def _copy_workspace(workspace: str, to: str) -> None:
    """``workspace`` copied to ``to``, every path in its files that names it renamed to the copy - written plain, or
    as JSON writes it (a Windows path's backslashes doubled)."""
    shutil.copytree(workspace, to)
    was, now = os.path.abspath(workspace), os.path.abspath(to)
    forms = [(was, now), (was.replace("\\", "\\\\"), now.replace("\\", "\\\\"))]
    for root, _, files in os.walk(to):
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, OSError):
                continue
            moved = text
            for a, b in forms:
                if a and a in moved:
                    moved = moved.replace(a, b)
            if moved != text:
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(moved)


def pool(
    env_factory: Callable[[Any, int], gym.Env],
    workers: int = 8,
    *,
    workspace: Any,
    store: Optional[str] = None,
    bench: Optional[str] = None,
    context: Optional[str] = None,
    **vector_kwargs: Any,
) -> Pool:
    """**``workers`` environments, each in a process of its own over a copy of ``workspace``** (module doc):
    ``env_factory(local, index)`` makes each one. Refused when the workspace is held - by a window, ``--serve``
    or an engine in this program - since a copy taken under a writer is not one state. ``context`` is
    ``multiprocessing``'s (``"spawn"`` is what Windows and macOS have; the default is Gymnasium's).

    ``workspace`` may be a practice tape (``bentrix.practice()``), its store then its own. With ``bench`` - a
    workbench's name or address - each worker's ``env_factory(bench, index)`` is handed that workbench of its own
    engine, as a ``bentrix.Bench``, in place of the engine. A held workspace is ``bentrix.Held``."""
    from bentrix import _native
    from bentrix.core import Held
    from bentrix.practice import Practice

    if isinstance(workspace, Practice):
        workspace, store = workspace.workspace, store or workspace.store

    if workers < 1:
        raise ValueError(f"workers {workers} is not 1 or more")
    held = _native.workspace_held(workspace)
    if not held["exists"]:
        raise ValueError(f"there is no workspace at {workspace}")
    if held["held"]:
        raise Held(held["said"] + " Close it before a pool copies the workspace.")
    folder = tempfile.mkdtemp(prefix="bentrix-pool-")
    try:
        from bentrix.local import Local

        makers = []
        for i in range(workers):
            copy = os.path.join(folder, f"worker-{i}")
            _copy_workspace(workspace, copy)
            # **Opened once here, and let go**: an engine a copy cannot open (the package, a held or broken copy) is
            # refused in its own words in this process, where a worker's would reach the caller as a connection
            # reset with nothing said.
            Local(workspace=copy, store=store).close()
            makers.append(_Maker(env_factory, copy, store, i, bench))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return Pool(makers, folder, context=context, **vector_kwargs)
