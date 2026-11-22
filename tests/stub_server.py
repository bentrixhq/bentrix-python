"""A stub BENTRIX endpoint speaking the 2026-07-28 wire, for the runtime's tests.

It validates what the specification says a server MUST validate (the mirrored headers against the
body, the Accept pair, the per-request metadata) and records every violation, so a test can assert the
client sent none. It answers just enough of the fixture catalogue to drive every runtime path.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = "stub-token"


class Stub:
    def __init__(self, catalogue="1.0.0", discover_meta=True, pair_secret="the-secret", pair_status=None, pair_answer=None):
        self.pair_answer = pair_answer  # replaces the success payload
        self.pair_secret = pair_secret
        self.pair_status = pair_status  # force an HTTP status from the pairing route
        self.pair_requests = []  # (path, headers, parsed body)
        self.catalogue = catalogue
        self.discover_meta = discover_meta
        self.calls = []  # (method, params without _meta)
        self.violations = []
        self.job_polls = 0
        # The next `busy` requests are answered as Bentrix's rate limit answers them: HTTP 429,
        # JSON-RPC -40001 and the wait in `retryAfterMs` -- refused before anything ran.
        self.busy = 0
        self.busy_after_ms = 20
        self.quote_pages = [
            {"items": [{"symbol": "EURUSD", "bid": 1.1, "ask": 1.2}], "next": "c1"},
            {
                "items": [
                    {"symbol": "GBPUSD", "bid": 1.3, "ask": 1.4},
                    {"symbol": "USDJPY", "bid": 150.0, "ask": 150.1},
                ],
                "next": "c2",
            },
        ]
        self.after_seen = []
        self.wait_for_delay = 0.0  # seconds a `wait_for` holds before it answers, as a long poll does
        # **Streams an `Events` binding reads** (uri -> the pages `wait_for` answers in turn; once they
        # run out it answers an empty page at the same cursor). `resources/read` of such a uri answers
        # the present: no item, cursor "c0".
        self.streams = {}
        self.stream_after = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _pair(self, body):
                doc = json.loads(body)
                stub.pair_requests.append((self.path, dict(self.headers), doc))
                if stub.pair_status:
                    status, payload = stub.pair_status, {"error": {"code": "rate_limited", "message": "wait"}}
                elif doc.get("secret") == stub.pair_secret:
                    status, payload = 200, ({"token": TOKEN} if stub.pair_answer is None else stub.pair_answer)
                else:
                    status, payload = 403, {"error": {"code": "wrong_secret", "message": "no"}}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path.endswith("/pair"):
                    return self._pair(body)
                msg = json.loads(body)
                stub._validate(self.headers, msg)
                status, payload, sse = stub._dispatch(msg)
                data = json.dumps(payload).encode()
                if sse:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.write(b"data: " + json.dumps(note).encode() + b"\n\n")
                    # a response to some other request: the client must not take it for its own
                    stray = {"jsonrpc": "2.0", "id": 0, "result": {"structuredContent": {"bench": "STRAY", "uri": "x"}}}
                    self.wfile.write(b"data: " + json.dumps(stray).encode() + b"\n\n")
                    self.wfile.write(b"data: " + data + b"\n\n")
                else:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    # --- what a server must check ---------------------------------------------------------------

    def _validate(self, headers, msg):
        v = self.violations
        params = msg.get("params", {})
        meta = params.get("_meta", {})
        if headers.get("MCP-Protocol-Version") != "2026-07-28":
            v.append("MCP-Protocol-Version header")
        if meta.get("io.modelcontextprotocol/protocolVersion") != headers.get("MCP-Protocol-Version"):
            v.append("header/body protocol version mismatch")
        if "io.modelcontextprotocol/clientCapabilities" not in meta:
            v.append("clientCapabilities missing")
        if "io.modelcontextprotocol/clientInfo" not in meta:
            v.append("clientInfo missing")
        accept = headers.get("Accept", "")
        if "application/json" not in accept or "text/event-stream" not in accept:
            v.append("Accept")
        if headers.get("Mcp-Method") != msg.get("method"):
            v.append("Mcp-Method")
        if msg.get("method") == "tools/call" and headers.get("Mcp-Name") != params.get("name"):
            v.append("Mcp-Name (tool)")
        if msg.get("method") == "resources/read" and headers.get("Mcp-Name") != params.get("uri"):
            v.append("Mcp-Name (resource)")
        if headers.get("Authorization") != f"Bearer {TOKEN}":
            v.append("Authorization")

    # --- the fixture catalogue's answers ---------------------------------------------------------

    def _dispatch(self, msg):
        method, params, rid = msg["method"], msg.get("params", {}), msg["id"]
        clean = {k: v for k, v in params.items() if k != "_meta"}
        if self.busy > 0 and method != "server/discover":
            self.busy -= 1
            self.calls.append(("busy", clean))
            return 429, {"jsonrpc": "2.0", "id": rid, "error": {"code": -40001, "message": "Rate limited: slow down", "data": {"retryAfterMs": self.busy_after_ms}}}, False
        self.calls.append((method, clean))

        def ok(result, sse=False):
            result.setdefault("resultType", "complete")
            return 200, {"jsonrpc": "2.0", "id": rid, "result": result}, sse

        def err(code, message, status=400):
            return status, {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}, False

        def structured(obj, sse=False):
            return ok({"content": [{"type": "text", "text": json.dumps(obj)}], "structuredContent": obj}, sse)

        if method == "server/discover":
            result = {"supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}}}
            if self.discover_meta and self.catalogue is not None:
                result["_meta"] = {"bentrix/catalogue": self.catalogue}
            return ok(result)
        if method == "resources/read":
            uri = params["uri"]
            if uri in self.streams:
                return ok({"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps({"items": [], "next": "c0"})}]})
            u = urlparse(uri)
            if u.path.endswith("/journal"):
                after = parse_qs(u.query).get("after", [None])[0]
                if after is None:
                    page = {"lines": [{"at_ms": 1, "text": "a"}], "next": uri.split("?")[0] + "?after=p2"}
                else:
                    page = {"lines": [{"at_ms": 2, "text": "b"}]}
                return ok({"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(page)}]})
            return err(-32602, f"unknown resource {uri}", 404)
        if method == "tools/call":
            name, args = params["name"], params.get("arguments", {})
            if name == "agent_whoami":
                if self.catalogue is None:
                    return structured({})
                return structured({"catalogue_version": self.catalogue})
            if name == "fx_bench_create":
                return structured({"bench": "b1", "uri": "bentrix://bench/b1"}, sse=args.get("lab") == "sse")
            if name == "fx_replay_step":
                if args.get("n") == 999:
                    return structured({"job": "j1"})
                return structured(
                    {
                        "moment_ms": 5,
                        "moment": "t",
                        "quotes": {"EURUSD": 1.1},
                        "undone": {"deals": 0, "requests": 0},
                        "extra_from_a_newer_server": True,
                    }
                )
            if name == "job_get":
                self.job_polls += 1
                if self.job_polls < 3:
                    return structured({"status": "awaiting_person" if self.job_polls == 1 else "working"})
                return structured(
                    {
                        "status": "completed",
                        "result": {
                            "moment_ms": 9,
                            "moment": "later",
                            "quotes": {},
                            "undone": {"deals": 1, "requests": 2},
                        },
                    }
                )
            if name == "fx_fail":
                return ok({"isError": True, "content": [{"type": "text", "text": "bench b7 is not in lab3"}]})
            if name == "fx_ask":
                return ok({"resultType": "input_required", "inputRequests": {}})
            if name == "wait_for" and args.get("uri") in self.streams:
                seen = self.stream_after.setdefault(args["uri"], [])
                seen.append(args.get("after"))
                pages = self.streams[args["uri"]]
                if pages:
                    page = pages.pop(0)
                    if "__error__" in page:
                        # A passing refusal, as the server answers it (`busy`, `expired`).
                        return ok({"isError": True, "content": [{"type": "text", "text": page["__error__"]}], "_meta": {"bentrix/machine": page["__error__"]}})
                    return structured(page)
                import time

                time.sleep(0.02)
                return structured({"items": [], "next": args.get("after")})
            if name == "wait_for":
                if self.wait_for_delay:
                    import time

                    time.sleep(self.wait_for_delay)
                self.after_seen.append(args.get("after"))
                i = len(self.after_seen) - 1
                return structured(self.quote_pages[min(i, len(self.quote_pages) - 1)])
            return err(-32602, f"Unknown tool: {name}")
        return err(-32601, f"Method not found: {method}", 404)
