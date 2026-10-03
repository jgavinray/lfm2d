#!/usr/bin/env python3
"""Full lfm2d endpoint exercise on the B70: correctness pass then throughput.

Every response is structure- and value-checked; failures are collected, not
fatal, so one bad endpoint doesn't hide the rest. Throughput numbers are
median wall time per call over N calls after warmup, measured client-side
(loopback, so it includes HTTP overhead — the honest end-to-end number).
"""
import json, statistics, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8931"
FAILS = []
RESULTS = {}

def call(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            try:
                parsed = json.loads(payload) if payload else None
            except json.JSONDecodeError:
                parsed = payload
            return r.status, parsed, dict(r.headers), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        raw = e.read() or b"null"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed, dict(e.headers), time.perf_counter() - t0

def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")
    else:
        print(f"  ok   {name}")

def bench(name, fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        _, _, _, dt = fn()
        times.append(dt * 1e3)
    RESULTS[name] = {"median_ms": statistics.median(times), "p90_ms": sorted(times)[int(0.9 * len(times)) - 1], "iters": iters}
    print(f"  {name}: median {statistics.median(times):.2f} ms  p90 {sorted(times)[int(0.9*len(times))-1]:.2f} ms")

print("== correctness ==")
# healthz/readyz
s, b, h, _ = call("GET", "/healthz")
check("healthz", s == 200 and b == b"ok", f"{s} {b}")
s, b, h, _ = call("GET", "/readyz")
check("readyz", s == 200, f"{s}")

# /v1/models
s, models, _, _ = call("GET", "/v1/models")
check("models 4 heads (adjudicator separate)", s == 200 and len(models) == 4, f"{s} n={len(models) if isinstance(models, list) else models}")
kinds = {m["id"]: m["kind"] for m in models}
check("models kinds", set(kinds.values()) == {"embedder", "router", "token_classifier", "adjudicator"}, str(kinds))
check("models 64-hex hash", all(len(m["weight_hash"]) == 64 for m in models), "")

# /embed — single + batch + query kind
s, b, h, _ = call("POST", "/embed", {"inputs": "A mutex protects shared memory from concurrent writes."})
check("embed single", s == 200 and isinstance(b, list) and len(b) == 1 and len(b[0]) == 1024, f"{s}")
check("embed headers", "x-model-id" in {k.lower() for k in h} and len(h.get("X-Model-Weight-Hash", h.get("x-model-weight-hash", ""))) == 64, str(dict(h).keys()))
s, b, h, _ = call("POST", "/embed", {"inputs": ["hello world", "second input"], "kind": "query"})
check("embed batch query", s == 200 and len(b) == 2 and all(len(v) == 1024 for v in b), f"{s}")

# /v1/route
s, b, _, _ = call("POST", "/v1/route", {"input": "kubectl get pods -n kube-system", "routes": ["shell", "k8s", "general conversation"]})
check("route shape", s == 200 and "routes" in b and len(b["routes"]) == 3 and all("cosine" in r for r in b["routes"]), f"{s} {b}")
cos = {r["route"]: r["cosine"] for r in b["routes"]}
check("route k8s top", max(cos, key=cos.get) == "k8s", str(cos))

# /v1/spans — PII with known entities (byte offsets validated against input)
text = "Contact alice@example.com for the deployment review, or call +1-555-0100."
s, b, _, _ = call("POST", "/v1/spans", {"inputs": text})
check("spans single", s == 200 and isinstance(b, list) and len(b) == 1, f"{s} {b}")
spans = b[0]
check("spans nonempty", len(spans) >= 1, str(spans))
ok_off = all(text[sp["start"]:sp["end"]].strip() for sp in spans if sp["end"] <= len(text))
check("spans byte offsets slice cleanly", ok_off, str([(sp["start"], sp["end"]) for sp in spans]))
check("spans have entity+score", all("entity" in sp and "score" in sp for sp in spans), "")

s, b, _, _ = call("POST", "/v1/spans", {"inputs": ["api_key = sk-1234567890abcdef", "no secrets here"]})
check("spans batch", s == 200 and len(b) == 2, f"{s}")

# /v1/spans/credentials — filtered to credential.*
s, b, _, _ = call("POST", "/v1/spans/credentials", {"inputs": "password=hunter2 and jwt eyJhbGciOi.abc.def and lunch was fine"})
check("credentials filtered", s == 200 and all(sp["entity"].startswith("credential.") or sp["entity"] == "developer.login_credentials" for s2 in b for sp in s2), f"{s} {b}")

# /v1/tokenize — encoder model + adjudicator + context
s, b, _, _ = call("POST", "/v1/tokenize", {"model": "LFM2.5-Embedding-350M", "text": "hello world"})
check("tokenize encoder", s == 200 and "ids" in b and len(b["tokens"]) == len(b["ids"]), f"{s} {str(b)[:120]}")
s, b, _, _ = call("POST", "/v1/tokenize", {"model": "LFM2.5-8B-A1B-Q5_K_M", "text": "hello world", "context": "The system says:"})
check("tokenize adjudicator+context", s == 200 and "context" in b and "suffix_stable" in b["context"], f"{s} {str(b)[:160]}")

# opinion specs: upload a fixture spec, list, use it, delete it
spec = open("/home/jgavinray/dev/lfm2d/lfm2d/tests/fixtures/specs/email-triage-v1.json", "rb").read()
s, b, _, _ = call("POST", "/v1/opinion/specs", None)
check("spec upload requires body", s == 400, f"{s}")
req = urllib.request.Request(BASE + "/v1/opinion/specs", data=spec, method="POST")
with urllib.request.urlopen(req, timeout=60) as r:
    spec_resp = json.loads(r.read()); spec_id = spec_resp["id"] if isinstance(spec_resp, dict) else spec_resp
    s_spec = r.status
check("spec upload", s_spec in (200, 201), f"{s_spec} {str(spec_resp)[:200]}")
s, b, _, _ = call("GET", "/v1/opinion/specs")
check("spec listed", s == 200 and any((x.get("id") == spec_id) if isinstance(x, dict) else x == spec_id for x in b), f"{s} {str(b)[:200]}")

# /v1/opinion — schema-constrained read (questions are {field: str} objects)
s, b, _, _ = call("POST", "/v1/opinion", {"spec": spec_id, "state": {"input": "Server disk is 97% full on prod-db-1, alerts firing since 06:00."}, "questions": [{"field": "verdict"}]})
check("opinion answered", s == 200 and "distributions" in str(b), f"{s} {str(b)[:200]}")

# /v1/adjudicate — generative (2048-token cap default; cap at 32 tokens for the test)
s, b, _, _ = call("POST", "/v1/adjudicate", {"spec": spec_id, "input": "Server disk is 97% full on prod-db-1, alerts firing since 06:00.", "max_tokens": 32})
check("adjudicate answered", s == 200 and ("output" in b or "text" in b or "content" in b), f"{s} {str(b)[:200]}")

# /v1/probe — raw instrument
s, b, _, _ = call("POST", "/v1/probe", {"text": "The capital of France is"})
check("probe answered", s == 200, f"{s} {str(b)[:160]}")

# error paths
s, b, _, _ = call("POST", "/v1/tokenize", {"model": "no-such-model", "text": "x"})
check("tokenize 404", s == 404, f"{s}")
s, b, _, _ = call("POST", "/v1/opinion", {"spec": "no-such-spec", "state": {"input": "x"}, "questions": [{"field": "verdict"}]})
check("opinion unknown spec 4xx", s in (400, 404), f"{s}")
s, b, _, _ = call("POST", "/v1/spans", {"inputs": ""})
check("spans empty input handled", s in (200, 400), f"{s}")

print("\n== throughput ==")
bench("embed single doc", lambda: call("POST", "/embed", {"inputs": "A mutex protects shared memory from concurrent writes."}), 50)
bench("embed batch 8 doc", lambda: call("POST", "/embed", {"inputs": [f"Document number {i} about databases and indexes." for i in range(8)]}), 30)
bench("embed batch 32 query", lambda: call("POST", "/embed", {"inputs": [f"query {i}" for i in range(32)], "kind": "query"}), 30)
bench("route 3 routes", lambda: call("POST", "/v1/route", {"input": "kubectl get pods -n kube-system", "routes": ["shell", "k8s", "general conversation"]}), 50)
bench("spans single", lambda: call("POST", "/v1/spans", {"inputs": "Contact alice@example.com for the deployment review."}), 50)
bench("spans batch 8", lambda: call("POST", "/v1/spans", {"inputs": [f"User {i} email user{i}@example.com" for i in range(8)]}), 30)
bench("spans/credentials batch 8", lambda: call("POST", "/v1/spans/credentials", {"inputs": [f"password=secret{i}" for i in range(8)]}), 30)
bench("tokenize", lambda: call("POST", "/v1/tokenize", {"model": "LFM2.5-Embedding-350M", "text": "hello world"}), 100)
bench("probe 5-token gen", lambda: call("POST", "/v1/probe", {"text": "The capital of France is", "max_tokens": 5} if True else None), 10)
bench("opinion read", lambda: call("POST", "/v1/opinion", {"spec": spec_id, "state": {"input": "Server disk is 97% full on prod-db-1."}, "questions": [{"field": "verdict"}]}), 20)
bench("adjudicate gen 32tok", lambda: call("POST", "/v1/adjudicate", {"spec": spec_id, "input": "Disk 97% full on prod-db-1.", "max_tokens": 32}), 10)

# cleanup spec
req = urllib.request.Request(BASE + f"/v1/opinion/specs/{spec_id}", method="DELETE")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        check("spec delete", r.status in (200, 204), str(r.status))
except urllib.error.HTTPError as e:
    check("spec delete", e.code in (200, 204), str(e.code))

print("\n== summary ==")
print(json.dumps(RESULTS, indent=2))
print(f"\nFAILURES: {len(FAILS)}")
for f in FAILS:
    print(f"  {f}")
open("/tmp/lfm2d_exercise_results.json", "w").write(json.dumps({"results": RESULTS, "failures": FAILS}, indent=2))
