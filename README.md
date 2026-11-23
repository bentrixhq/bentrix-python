# Bentrix SDK for Python

The Python client of [Bentrix](https://bentrix.net) — charts, a replay laboratory and trading through your own MetaTrader — for a program or an agent.

It finds the Bentrix running on this computer through the agents folder Bentrix writes at every start, pairs with it under the name you give — the name you see in ☰ › Settings › **Agents**, where the pairing's permissions are set — and calls every tool of Bentrix's capability catalogue over MCP. Nothing is typed: no address, no key.

The client is generated from the catalogue of the Bentrix release it came with, and checks when it connects that the running Bentrix serves the same major version.

## What you need

Python 3.9 or later, and Bentrix open on the same computer.

## Install

```bash
pip install bentrix
python -c "import bentrix; print(bentrix.CATALOGUE_VERSION)"
```

Or from a clone of this repository: `pip install .`

The client needs nothing else. The engine's modules below — `bentrix.data`, `bentrix.gym`, `bentrix.forecast`, `bentrix.vector` and `bentrix.live` — need NumPy, and the Gym's spaces Gymnasium:

```bash
pip install "bentrix[engine]"    # NumPy
pip install "bentrix[gym]"       # NumPy and Gymnasium
```

## Connect

```python
import bentrix

b = bentrix.connect(name="my-agent")     # finds the running Bentrix and pairs; the name shows in Settings › Agents
state = b.workspace.get_state(depth="benches")
bench = state["laboratories"][0]["benches"][0]["uri"]
home, owner, _, name = bench.removeprefix("bentrix://").split("/")
book = b.trade.book(home, owner, name)
print(len(book["positions"]), "positions open")
```

Every tool is a method grouped by its domain (`b.workspace`, `b.chart`, `b.study`, `b.trade` …). A refusal raises `bentrix.ToolError` with Bentrix's own sentence; *busy* is waited out and the call sent again.

## The engine in your own process

`bentrix.local`, `bentrix.backtest`, `bentrix.data` and `bentrix.gym` without a running Bentrix (the module `bentrix._native`) run Bentrix's engine inside your program, with no window. The engine is Bentrix's own library, `libbentrix`, which Bentrix's installer puts in its folder (`%LOCALAPPDATA%\Programs\BENTRIX\libbentrix.dll` on Windows) and the binding finds there; `BENTRIX_FFI` (the file) or `BENTRIX_FFI_DIR` (its folder) names another. It runs for a computer signed in to Bentrix with Plus or Pro; with Free, connect to the running application as above.

## Documentation

Every method, its arguments, the permission it needs and what it answers: **https://bentrix.net/en/docs/sdk/python/**

## Licence

MIT — see [LICENSE](https://github.com/bentrixhq/bentrix-python/blob/main/LICENSE). It covers this source code only: Bentrix's engine library (`libbentrix`) and the Bentrix application are Bentrix's own, under their own terms.
