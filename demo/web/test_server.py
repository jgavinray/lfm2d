"""Tests for the demo web proxy; no checkpoints or network required."""
import http.client
import json
import re
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import server


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []
    release = threading.Event()   # "hold": the stream's second event waits for this
    hung_up = threading.Event()   # "drip": set when a write finds the reader gone

    def log_message(self, *args):
        pass

    def _stream(self, mode):
        """Server-sent events the way the daemon's /v1/chat sends them:
        no length, one flush per event."""
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        if mode == "die":
            self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        event = lambda name, n: f'event: {name}\ndata: {{"n": {n}}}\n\n'.encode()
        if mode == "die":
            raw = event
            event = lambda name, n: b"%x\r\n%s\r\n" % (len(raw(name, n)), raw(name, n))
        try:
            self.wfile.write(event("checkpoint", 0))
            self.wfile.flush()
            if mode == "die":   # chunked, then gone without the last chunk: a daemon that died mid-turn
                self.wfile.flush()
                return
            if mode == "hold":
                FakeUpstream.release.wait(10)
                self.wfile.write(event("done", 1))
                return
            for n in range(1, 400):   # "drip": a token every 25 ms for up to 10 s
                time.sleep(0.025)
                self.wfile.write(event("token", n))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            FakeUpstream.hung_up.set()

    def _answer(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        FakeUpstream.seen.append((self.command, self.path, body))
        if self.path == "/v1/chat":
            return self._stream(json.loads(body)["mode"])
        if self.path == "/v1/tokenize":
            status, out = 404, {"error": {"message": "no such model", "type": "not_found"}}
        else:
            status, out = 200, {"echo": self.path, "method": self.command}
        data = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _answer


def serve(httpd):
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream))
        up = "http://127.0.0.1:%d" % cls.upstream.server_address[1]
        cls.proxy = serve(server.make_server("127.0.0.1", 0, up))
        cls.base = "http://127.0.0.1:%d" % cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        for httpd in (cls.proxy, cls.upstream):
            httpd.shutdown()
            httpd.server_close()

    def setUp(self):
        FakeUpstream.seen.clear()
        FakeUpstream.release.clear()
        FakeUpstream.hung_up.clear()

    def open_stream(self, mode):
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=5)
        conn.request("POST", "/api/v1/chat", json.dumps({"mode": mode}),
                     {"content-type": "application/json"})
        return conn, conn.getresponse()

    @staticmethod
    def next_event(resp):
        buf = b""
        while not buf.endswith(b"\n\n"):
            byte = resp.read(1)
            if not byte:
                raise EOFError(f"stream ended inside an event: {buf!r}")
            buf += byte
        return buf

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    def test_allowed_call_is_forwarded_with_its_body(self):
        status, body = self.call("POST", "/api/v1/probe", {"text": "x"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"echo": "/v1/probe", "method": "POST"})
        self.assertEqual(FakeUpstream.seen, [("POST", "/v1/probe", b'{"text": "x"}')])

    def test_unlisted_path_never_reaches_upstream(self):
        for method, path in (("POST", "/api/v1/spans"), ("GET", "/api/readyz"),
                             ("POST", "/api/v1/opinion/specs/../../probe")):
            with self.subTest(path=path):
                status, _ = self.call(method, path, {} if method == "POST" else None)
                self.assertEqual(status, 404)
        self.assertEqual(FakeUpstream.seen, [])

    def test_wrong_method_on_a_listed_path_is_refused(self):
        status, _ = self.call("GET", "/api/v1/probe")
        self.assertEqual(status, 404)
        self.assertEqual(FakeUpstream.seen, [])

    def test_an_event_stream_is_relayed_as_it_arrives_not_when_it_ends(self):
        # The upstream holds its second event until released: a proxy that
        # buffers the body never delivers the first one, and the read times out.
        conn, resp = self.open_stream("hold")
        try:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers["content-type"], "text/event-stream")
            self.assertEqual(self.next_event(resp), b'event: checkpoint\ndata: {"n": 0}\n\n')
            FakeUpstream.release.set()
            self.assertEqual(self.next_event(resp), b'event: done\ndata: {"n": 1}\n\n')
            self.assertEqual(resp.read(), b"", "the stream ends when the upstream's does")
        finally:
            resp.close()
            conn.close()

    def test_a_browser_that_hangs_up_hangs_up_the_upstream(self):
        # The daemon cancels a chat turn when its client goes; the proxy is
        # that client, so it must not keep reading for a reader that left.
        conn, resp = self.open_stream("drip")
        self.next_event(resp)
        resp.close()   # an HTTP/1.0 answer: the response holds the only socket
        conn.close()
        self.assertTrue(FakeUpstream.hung_up.wait(5), "upstream still streaming to nobody")

    def test_an_upstream_that_breaks_mid_stream_ends_it_with_an_error_event(self):
        # The 200 is already out, so a second status line in the body would be
        # garbage to the reader: the break arrives as the daemon's own late
        # failures do, an `error` event with a status.
        conn, resp = self.open_stream("die")
        try:
            self.assertEqual(self.next_event(resp), b'event: checkpoint\ndata: {"n": 0}\n\n')
            rest = resp.read()
        finally:
            resp.close()
            conn.close()
        self.assertNotIn(b"HTTP/", rest)
        name, data = rest.strip().split(b"\n")
        self.assertEqual(name, b"event: error")
        self.assertEqual(json.loads(data.removeprefix(b"data: "))["status"], 502)

    def test_upstream_error_status_is_passed_through_not_masked(self):
        status, body = self.call("POST", "/api/v1/tokenize", {"model": "nope", "text": "x"})
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["type"], "not_found")

    def test_dead_upstream_is_a_502(self):
        dead = serve(server.make_server("127.0.0.1", 0, "http://127.0.0.1:1"))
        try:
            base = "http://127.0.0.1:%d" % dead.server_address[1]
            req = urllib.request.Request(base + "/api/v1/adjudicator")
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(e.exception.code, 502)
            e.exception.close()
        finally:
            dead.shutdown()
            dead.server_close()

    def test_static_page_is_served_and_traversal_is_not(self):
        status, body = self.call("GET", "/sour-note")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>", body)
        for path in ("/../server.py", "/%2e%2e/server.py", "/static/../server.py"):
            with self.subTest(path=path):
                status, _ = self.call("GET", path)
                self.assertEqual(status, 404)


class HostTests(unittest.TestCase):
    def test_explicit_host_wins(self):
        self.assertEqual(server.resolve_host("127.0.0.1", run=None), "127.0.0.1")

    def test_wildcard_bind_is_refused(self):
        for host in ("0.0.0.0", "::", ""):
            with self.subTest(host=host), self.assertRaises(SystemExit):
                server.resolve_host(host, run=None)

    def test_default_is_the_tailnet_address(self):
        class Done:
            returncode, stdout = 0, "100.64.0.7\nfd7a::1\n"
        self.assertEqual(server.resolve_host(None, run=lambda *a, **k: Done()), "100.64.0.7")

    def test_no_tailnet_and_no_host_fails_loudly(self):
        def missing(*a, **k):
            raise FileNotFoundError("tailscale")
        with self.assertRaises(SystemExit):
            server.resolve_host(None, run=missing)


# Pages a person uses rather than watches: they reflow to the window (a
# phone included) instead of scaling a fixed 9:16 stage.
INTERACTIVE = {"index.html", "tail.html"}


class PageTests(unittest.TestCase):
    def test_every_page_has_a_title_and_only_calls_the_proxy(self):
        pages = sorted((Path(server.__file__).parent / "static").glob("*.html"))
        self.assertTrue(pages)
        for page in pages:
            text = page.read_text()
            with self.subTest(page=page.name):
                self.assertIn("<title>", text)
                self.assertNotIn("ts.net", text)
                self.assertNotIn("http://", text.replace("http://www.w3.org", ""))
                if page.name in INTERACTIVE:
                    self.assertIn('name="viewport" content="width=device-width', text)
                else:
                    # a scaled stage centred by layout overflows any window shorter than it
                    self.assertIn("translate(-50%, -50%) scale", text)

    def test_every_daemon_call_a_page_makes_is_proxied(self):
        static = Path(server.__file__).parent / "static"
        allowed = {path for _, path in server.ALLOWED}
        for page in sorted(static.glob("*.html")):
            for path in re.findall(r'"(?:/api)?(/v1/[a-z/]+|/embed)"', page.read_text()):
                with self.subTest(page=page.name, path=path):
                    self.assertIn(path, allowed)

    def test_tail_scenarios_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        scenarios = json.loads((static / "tail.json").read_text())["scenarios"]
        self.assertEqual(len({s["name"] for s in scenarios}), len(scenarios))
        for s in scenarios:
            self.assertIsInstance(s["system"], str)
            self.assertTrue(s["first"].strip())
            self.assertTrue(s["read"].strip())
            # the chat renderer refuses control tokens in any turn
            for text in (s["system"], s["first"], s["read"]):
                self.assertNotIn("<|", text)
                self.assertNotIn("<think>", text)

    def test_one_pass_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        spec = json.loads((static / "command-verdict-enum-v1.json").read_text())
        self.assertIsInstance(spec.get("input_label"), str)
        rows = json.loads((static / "one-pass-commands.json").read_text())["rows"]
        inputs = [r["input"] for r in rows]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertTrue(all(isinstance(r["hurts"], bool) for r in rows))
        self.assertEqual({r["hurts"] for r in rows}, {True, False})

    def test_two_worlds_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        cases = json.loads((static / "two-worlds.json").read_text())["cases"]
        self.assertEqual(len({c["input"] for c in cases}), len(cases))
        for c in cases:
            self.assertEqual(len(c["worlds"]), 2)
            for w in c["worlds"]:
                for key in ("name", "before", "output"):
                    self.assertTrue(w[key].strip(), key)
                self.assertTrue(w["fact"].strip() and "\n" not in w["fact"])

    def test_house_rules_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        conf = json.loads((static / "house-rules.json").read_text())
        doc = (static / conf["file"].lstrip("/")).read_text()
        self.assertGreaterEqual(sum(l.startswith("- ") for l in doc.splitlines()), 10)
        self.assertTrue(conf["commands"] and conf["finale"])
        self.assertEqual(len(set(conf["commands"] + conf["finale"])), len(conf["commands"]) + len(conf["finale"]))

    def test_let_you_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        page = (static / "let-you.html").read_text()
        spec_file = re.search(r'const SPEC_FILE = "/([^"]+)"', page).group(1)
        spec = json.loads((static / spec_file).read_text())
        self.assertIsInstance(spec.get("input_label"), str)
        self.assertEqual(sorted(spec["output_schema"]["properties"]["verdict"]["enum"]), ["go", "stop", "wait"])
        # the show types the description, deals the chips, then lights the lamps: all of
        # those must be written before the verdict slot, or the chips are post-hoc
        order = spec["output_schema"]["required"]
        for f in ("effect", "scope", "undo"):
            self.assertLess(order.index(f), order.index("verdict"), f)
        rounds = json.loads((static / "let-you.json").read_text())["rounds"]
        self.assertEqual(len(rounds), len(set(rounds)))


if __name__ == "__main__":
    unittest.main()
