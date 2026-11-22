"""**Every language's binding reaches the same engine and the same Gym**: one engine behind a C ABI, each language
a thin binding.

Each language's walk (``sdk/<language>/walk``) runs on its own copy of the CI tape and prints, as its
last JSON line, one Gym episode — seed 5, a window of 8, 12 candles, the same twelve discrete actions:
its start, every reward and the ledger hash at every step. The same episode is run here through
``bentrix.gym.TradingEnv``, and the two must agree exactly. A walk whose toolchain or library is absent
is skipped by name, never passed.

Environment: ``BENTRIX_FFI_DIR`` — the folder holding the C ABI's library (``libbentrix.so``,
``libbentrix.dll``); ``BENTRIX_DOTNET`` — the ``dotnet`` to run the .NET walk with.
"""

import json
import os
import pathlib
import subprocess

import pytest

pytest.importorskip("bentrix._native")

from bentrix.gym import TradingEnv  # noqa: E402

SDK = pathlib.Path(__file__).resolve().parents[2]
ACTIONS = [1, 1, 2, 0, 2, 2, 1, 0, 1, 2, 0, 1]


def _python_episode(local, bench):
    env = TradingEnv(bench, local=local, interval="H1", window=8, episode="window", episode_bars=12)
    _, info = env.reset(seed=5)
    rewards, hashes = [], [info["ledger_hash"]]
    for a in ACTIONS:
        _, r, term, trunc, step = env.step(a)
        rewards.append(r)
        hashes.append(step["ledger_hash"])
        if term or trunc:
            break
    return {"start_ms": info["start_ms"], "rewards": rewards, "hashes": hashes}


def _episode_of(stdout):
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    assert lines and stdout.rstrip().endswith("PASS"), stdout[-2000:]
    return json.loads(lines[-1])


def _ffi_env():
    ffi = os.environ.get("BENTRIX_FFI_DIR")
    if not ffi:
        pytest.skip("BENTRIX_FFI_DIR (the C ABI's library) is not set")
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
        env[var] = ffi + os.pathsep + env.get(var, "")
    return env


def test_the_dotnet_binding_runs_the_same_episode(tape, tmp_path):
    """The .NET binding (``sdk/csharp``): its walk passes, and its episode is this one, value for value.

    Mutation (2026-11-01): the .NET ``TradingEnv`` handing ``episode_bars`` one candle more
    than it was given — red, its episode draws another start."""
    dotnet = os.environ.get("BENTRIX_DOTNET")
    if not dotnet:
        pytest.skip("BENTRIX_DOTNET (a dotnet SDK) is not set")
    env = dict(_ffi_env(), DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1", DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1")
    # Built whole, never incrementally: a source restored with an older mtime than the last build's output
    # would otherwise run the last build (measured 2026-11-01, after a mutation was put back).
    project = str(SDK / "csharp" / "walk")
    built = subprocess.run([dotnet, "build", project, "--no-incremental", "-v", "q"], capture_output=True, text=True, env=env, timeout=900)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]
    out = subprocess.run([dotnet, "run", "--no-build", "--project", project, "--", str(tmp_path / "dotnet")], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    theirs = _episode_of(out.stdout)
    local, bench = tape
    ours = _python_episode(local, bench)
    assert theirs == ours, (theirs, ours)


def test_the_node_binding_runs_the_same_episode(tape, tmp_path):
    """The Node.js binding (``sdk/node``, its module built from ``crates/bentrix-node`` and named by
    ``BENTRIX_NODE_ADDON``): its walk passes, and its episode is this one, value for value.

    Mutation (2026-11-01): the Node ``TradingEnv`` handing ``episode_bars`` one candle more
    than it was given — red, its episode draws another start."""
    node, addon = os.environ.get("BENTRIX_NODE"), os.environ.get("BENTRIX_NODE_ADDON")
    if not node or not addon:
        pytest.skip("BENTRIX_NODE and BENTRIX_NODE_ADDON (node, and the built bentrix.node) are not set")
    out = subprocess.run([node, str(SDK / "node" / "walk.js"), str(tmp_path / "node")], capture_output=True, text=True, env=dict(os.environ), timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    theirs = _episode_of(out.stdout)
    local, bench = tape
    assert theirs == _python_episode(local, bench)


def test_the_r_binding_runs_the_same_episode(tape, tmp_path):
    """The R binding (``sdk/r/bentrix``, ``.Call`` over the C ABI): installed afresh into a library of
    this test's own (so a stale install is never what runs), its walk passes, and its episode is this
    one, value for value. ``BENTRIX_R``: the folder holding ``R`` and ``Rscript``; ``BENTRIX_FFI_INCLUDE``
    the header's folder; ``R_MAKEVARS_USER`` passed through where R's own compiler is not the system's.

    Mutation (2026-11-01): the R ``bx_env`` handing ``episode_bars`` one candle more than
    it was given — red, its episode draws another start."""
    rbin = os.environ.get("BENTRIX_R")
    if not rbin or not os.environ.get("BENTRIX_FFI_INCLUDE"):
        pytest.skip("BENTRIX_R and BENTRIX_FFI_INCLUDE (R, and the C ABI's header) are not set")
    env = _ffi_env()
    lib = tmp_path / "rlib"
    lib.mkdir()
    built = subprocess.run([os.path.join(rbin, "R"), "CMD", "INSTALL", "--preclean", "-l", str(lib), str(SDK / "r" / "bentrix")], capture_output=True, text=True, env=env, timeout=900)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]
    out = subprocess.run([os.path.join(rbin, "Rscript"), str(SDK / "r" / "walk.R"), str(tmp_path / "r")], capture_output=True, text=True, env=dict(env, R_LIBS=str(lib)), timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    theirs = _episode_of(out.stdout)
    local, bench = tape
    assert theirs == _python_episode(local, bench)


def test_the_go_binding_runs_the_same_episode(tape, tmp_path):
    """The Go binding (``sdk/go``, cgo over the C ABI; ``BENTRIX_GO`` the ``go`` to build it with): its
    walk passes, and its episode is this one, value for value.

    Mutation (2026-11-01): the Go ``NewTradingEnv`` handing ``episode_bars`` one candle more
    than it was given — red, its episode draws another start."""
    go = os.environ.get("BENTRIX_GO")
    if not go:
        pytest.skip("BENTRIX_GO (a go toolchain) is not set")
    env = _ffi_env()
    env["CGO_LDFLAGS"] = "-L" + os.environ["BENTRIX_FFI_DIR"]
    env["GOFLAGS"] = "-mod=mod"
    out = subprocess.run([go, "run", "./walk", str(tmp_path / "go")], cwd=str(SDK / "go"), capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_cpp_binding_runs_the_same_episode(tape, tmp_path):
    """The C++ binding (``sdk/cpp/bentrix.hpp``, header-only over the C ABI; ``BENTRIX_CXX`` the compiler):
    the walk compiled and run, its episode this one, value for value — a dependency-free JSON parser
    reading every reward the engine wrote.

    Mutation (2026-11-01): the header's number parser reading at most 12 characters —
    red, a reward written at 17 digits is cut short."""
    cxx = os.environ.get("BENTRIX_CXX")
    if not cxx or not os.environ.get("BENTRIX_FFI_INCLUDE"):
        pytest.skip("BENTRIX_CXX and BENTRIX_FFI_INCLUDE (a C++17 compiler, the C ABI's header) are not set")
    env = _ffi_env()
    exe = tmp_path / "walk"
    built = subprocess.run([cxx, "-std=c++17", "-Wall", "-I", os.environ["BENTRIX_FFI_INCLUDE"], str(SDK / "cpp" / "walk.cpp"), "-L", os.environ["BENTRIX_FFI_DIR"], "-lbentrix", "-o", str(exe)], capture_output=True, text=True, env=env, timeout=600)
    assert built.returncode == 0, built.stderr[-3000:]
    out = subprocess.run([str(exe), str(tmp_path / "cpp")], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_rust_sdk_runs_the_same_episode(tape, tmp_path):
    """The Rust SDK (``crates/bentrix-sdk``, the embedded shell behind a typed API; ``BENTRIX_RUST_WALK``
    the built ``walk`` example — ``cargo build -p bentrix-sdk --example walk``): its walk passes, and its
    episode is this one, value for value.

    Mutation (2026-11-01): the Rust ``TradingEnv::reset`` sending a seed one higher than
    it was given — red, its episode draws another start."""
    walk = os.environ.get("BENTRIX_RUST_WALK")
    if not walk:
        pytest.skip("BENTRIX_RUST_WALK (the Rust SDK's built walk) is not set")
    out = subprocess.run([walk, str(tmp_path / "rust")], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_java_binding_runs_the_same_episode(tape, tmp_path):
    """The Java binding (``sdk/java``, the Foreign Function & Memory API over the C ABI, Java 22 or later;
    ``BENTRIX_JAVA`` the folder holding ``javac`` and ``java``): the walk compiled and run, its episode
    this one, value for value — a dependency-free JSON reader and writer on both sides of every call.

    Mutation (2026-11-01): the Java ``TradingEnv`` sending ``episode_bars`` one candle
    more than it was given — red, its episode draws another start."""
    jdk = os.environ.get("BENTRIX_JAVA")
    if not jdk:
        pytest.skip("BENTRIX_JAVA (a JDK 22 or later) is not set")
    env = _ffi_env()
    out_dir = tmp_path / "classes"
    sources = sorted(str(p) for p in (SDK / "java" / "src" / "bentrix").glob("*.java")) + [str(SDK / "java" / "walk" / "Walk.java")]
    built = subprocess.run([os.path.join(jdk, "javac"), "-d", str(out_dir)] + sources, capture_output=True, text=True, env=env, timeout=600)
    assert built.returncode == 0, built.stderr[-3000:]
    run = [os.path.join(jdk, "java"), "--enable-native-access=ALL-UNNAMED", "-Djava.library.path=" + os.environ["BENTRIX_FFI_DIR"], "-cp", str(out_dir), "Walk", str(tmp_path / "java")]
    out = subprocess.run(run, capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_julia_binding_runs_the_same_episode(tape, tmp_path):
    """The Julia binding (``sdk/julia/Bentrix``, ``ccall`` over the C ABI, the standard library only;
    ``BENTRIX_JULIA`` the ``julia`` to run it with): its walk passes, and its episode is this one, value
    for value.

    Mutation (2026-11-01): the Julia ``TradingEnv`` sending ``episode_bars`` one candle
    more than it was given — red, its episode draws another start."""
    julia = os.environ.get("BENTRIX_JULIA")
    if not julia:
        pytest.skip("BENTRIX_JULIA (a julia) is not set")
    env = _ffi_env()
    run = [julia, "--project=" + str(SDK / "julia" / "Bentrix"), str(SDK / "julia" / "walk.jl"), str(tmp_path / "julia")]
    out = subprocess.run(run, capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


# ---- the MCP door: a running BENTRIX (bentrix --serve) over its own copy of the tape ----------------


@pytest.fixture
def serve(tmp_path):
    """**A running BENTRIX** — ``BENTRIX_EXE --serve`` over a CI tape of its own, its agents folder under
    a ``LOCALAPPDATA`` of its own; answers that environment for a client to find it by. Ended after."""
    import time

    exe = os.environ.get("BENTRIX_EXE")
    if not exe:
        pytest.skip("BENTRIX_EXE (a bentrix binary with --serve) is not set")
    from bentrix import _native

    ws, store = _native.make_tape(str(tmp_path / "served"))
    lad = tmp_path / "lad"
    lad.mkdir()
    env = dict(os.environ, LOCALAPPDATA=str(lad))
    proc = subprocess.Popen([exe, "--serve", "--workspace", ws, "--store", store], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    endpoint = lad / "BENTRIX" / "agents" / "endpoint.json"
    for _ in range(600):
        if endpoint.exists() and endpoint.stat().st_size > 0:
            break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("bentrix --serve wrote no endpoint in 60 s")
    yield env
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_the_javascript_client_runs_the_same_episode_over_mcp(serve, tape):
    """The JavaScript MCP client (``sdk/node/mcp``, generated from the catalogue): paired with a running
    BENTRIX with nothing typed, its walk passes and its episode — through the agents' tools, over the
    wire — is the engine's, value for value.

    Mutation (2026-11-01): the runtime ignoring a tool's ``isError`` — red, a refusal reads
    as a success. (A first mutation, nulls kept in the arguments, survived: no tool the walk calls refuses
    a null for an absent optional, so it could not show.)"""
    node = os.environ.get("BENTRIX_NODE")
    if not node:
        pytest.skip("BENTRIX_NODE (node) is not set")
    out = subprocess.run([node, str(SDK / "node" / "mcp" / "walk.js")], capture_output=True, text=True, env=serve, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_python_client_runs_the_same_episode_over_mcp(serve, tape):
    """The Python MCP client (``bentrix.connect``): the same episode over the wire as in-process."""
    import bentrix

    b = bentrix.connect(agents_folder=pathlib.Path(serve["LOCALAPPDATA"]) / "BENTRIX" / "agents", name="crosslang")
    bench = b.workspace.get_state()["laboratories"][0]["benches"][0]["uri"]
    made = b.strategy.env_make(bench, interval="H1", window=8, episode="window", episode_bars=12)
    start = b.strategy.env_reset(made["env"], seed=5)
    rewards, hashes = [], [start["info"]["ledger_hash"]]
    for a in ACTIONS:
        s = b.strategy.env_step(made["env"], a)
        rewards.append(s["reward"])
        hashes.append(s["info"]["ledger_hash"])
        if s["terminated"] or s["truncated"]:
            break
    local, bench_here = tape
    assert {"start_ms": start["info"]["start_ms"], "rewards": rewards, "hashes": hashes} == _python_episode(local, bench_here)


def test_the_csharp_client_runs_the_same_episode_over_mcp(serve, tape, tmp_path):
    """The C# MCP client (``sdk/csharp/mcp``, generated from the catalogue; ``BENTRIX_DOTNET``): paired
    with a running BENTRIX with nothing typed, its walk passes and its episode — over the wire — is the
    engine's, value for value.

    Mutation (2026-11-01): the runtime's URI template writing a path variable as
    nothing — red, the book's URI names no workbench."""
    dotnet = os.environ.get("BENTRIX_DOTNET")
    if not dotnet:
        pytest.skip("BENTRIX_DOTNET (a dotnet SDK) is not set")
    env = dict(serve, DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1", DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1")
    project = str(SDK / "csharp" / "mcpwalk")
    built = subprocess.run([dotnet, "build", project, "--no-incremental", "-v", "q"], capture_output=True, text=True, env=env, timeout=900)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-2000:]
    out = subprocess.run([dotnet, "run", "--no-build", "--project", project], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_go_client_runs_the_same_episode_over_mcp(serve, tape):
    """The Go MCP client (``sdk/go/mcp``, generated from the catalogue; ``BENTRIX_GO``): its walk passes
    and its episode — over the wire — is the engine's, value for value.

    Mutation (2026-11-01): the runtime reading a tool's ``isError`` answer as a success —
    red, a closed environment's step is not refused."""
    go = os.environ.get("BENTRIX_GO")
    if not go:
        pytest.skip("BENTRIX_GO (a go toolchain) is not set")
    env = dict(serve, CGO_ENABLED="0")
    out = subprocess.run([go, "run", "./mcpwalk"], cwd=str(SDK / "go"), capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_r_client_runs_the_same_episode_over_mcp(serve, tape, tmp_path):
    """The R MCP client (``sdk/r/bentrixmcp``, generated from the catalogue; ``BENTRIX_R``, with the
    ``curl`` and ``jsonlite`` packages): installed, its walk passes and its episode — over the wire — is
    the engine's, value for value.

    Mutation (2026-11-01): the hand-written JSON writer writing a whole number as
    ``5.0`` — red, an integer argument is refused."""
    rbin = os.environ.get("BENTRIX_R")
    if not rbin:
        pytest.skip("BENTRIX_R (R, with curl and jsonlite) is not set")
    lib = tmp_path / "rlib"
    lib.mkdir()
    built = subprocess.run([os.path.join(rbin, "R"), "CMD", "INSTALL", "--library=" + str(lib), str(SDK / "r" / "bentrixmcp")], capture_output=True, text=True, env=serve, timeout=900)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    env = dict(serve, R_LIBS=str(lib))
    out = subprocess.run([os.path.join(rbin, "Rscript"), str(SDK / "r" / "mcpwalk.R")], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_java_client_runs_the_same_episode_over_mcp(serve, tape, tmp_path):
    """The Java MCP client (``sdk/java/mcp``, generated from the catalogue, ``java.net.http`` and the
    engine binding's own ``bentrix.Json``; ``BENTRIX_JAVA``): compiled, its walk passes and its episode —
    over the wire — is the engine's, value for value.

    Mutation (2026-11-01): the runtime reading a tool's ``isError`` answer as a success —
    red, a closed environment's step is not refused."""
    jdk = os.environ.get("BENTRIX_JAVA")
    if not jdk:
        pytest.skip("BENTRIX_JAVA (a JDK 22 or later) is not set")
    out_dir = tmp_path / "classes"
    sources = [str(SDK / "java" / "src" / "bentrix" / "Json.java"), str(SDK / "java" / "src" / "bentrix" / "BentrixException.java")]
    sources += sorted(str(p) for p in (SDK / "java" / "mcp" / "src" / "bentrix" / "mcp").glob("*.java")) + [str(SDK / "java" / "mcp" / "walk" / "McpWalk.java")]
    built = subprocess.run([os.path.join(jdk, "javac"), "-d", str(out_dir)] + sources, capture_output=True, text=True, env=serve, timeout=600)
    assert built.returncode == 0, built.stderr[-3000:]
    out = subprocess.run([os.path.join(jdk, "java"), "-cp", str(out_dir), "McpWalk"], capture_output=True, text=True, env=serve, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_julia_client_runs_the_same_episode_over_mcp(serve, tape):
    """The Julia MCP client (``sdk/julia/BentrixMCP``, generated from the catalogue, the standard library
    only; ``BENTRIX_JULIA``): its walk passes and its episode — over the wire — is the engine's, value for
    value.

    Mutation (2026-11-01): the runtime's URI template writing a path variable as
    nothing — red, the book's URI names no workbench. (Keeping a ``nothing`` argument in the call could
    not show: the JSON writer leaves a ``nothing`` out of an object as well.)"""
    julia = os.environ.get("BENTRIX_JULIA")
    if not julia:
        pytest.skip("BENTRIX_JULIA (a julia) is not set")
    run = [julia, "--project=" + str(SDK / "julia" / "BentrixMCP"), str(SDK / "julia" / "mcpwalk.jl")]
    out = subprocess.run(run, capture_output=True, text=True, env=serve, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_cpp_client_runs_the_same_episode_over_mcp(serve, tape, tmp_path):
    """The C++ MCP client (``sdk/cpp/mcp``, header-only, generated from the catalogue, the platform's own
    sockets; ``BENTRIX_CXX``): compiled, its walk passes and its episode — over the wire — is the engine's,
    value for value.

    Mutation (2026-11-01): the runtime reading a tool's ``isError`` answer as a success —
    red, a closed environment's step is not refused."""
    cxx = os.environ.get("BENTRIX_CXX")
    if not cxx:
        pytest.skip("BENTRIX_CXX (a C++17 compiler) is not set")
    exe = tmp_path / "mcpwalk"
    built = subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(SDK / "cpp" / "mcp"), str(SDK / "cpp" / "mcp" / "walk.cpp"), "-o", str(exe)], capture_output=True, text=True, env=serve, timeout=600)
    assert built.returncode == 0, built.stderr[-3000:]
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=serve, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)


def test_the_rust_client_runs_the_same_episode_over_mcp(serve, tape):
    """The Rust MCP client (``crates/bentrix-mcp-client``, generated from the catalogue, ``std::net`` and
    ``serde_json`` only; ``BENTRIX_RUST_MCP_WALK`` the built ``mcpwalk`` example —
    ``cargo build -p bentrix-mcp-client --example mcpwalk``, never ``walk``, which is the engine SDK's
    and would land on the same file): its walk passes and its episode — over the
    wire — is the engine's, value for value.

    Mutation (2026-11-01): the runtime's URI template writing a path variable as
    nothing — red, the book's URI names no workbench."""
    walk = os.environ.get("BENTRIX_RUST_MCP_WALK")
    if not walk:
        pytest.skip("BENTRIX_RUST_MCP_WALK (the Rust MCP client's built walk) is not set")
    out = subprocess.run([walk], capture_output=True, text=True, env=serve, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    local, bench = tape
    assert _episode_of(out.stdout) == _python_episode(local, bench)
