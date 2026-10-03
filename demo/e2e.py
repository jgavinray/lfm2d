"""Real binary + real weights + HTTP over UDS. Missing prerequisites FAIL.

Run after cargo build --release -p lfm2d: python3 demo/e2e.py -v
No existing daemon or production endpoint is used; the child is always stopped.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest

from lfm2 import Client, ROOT, chunks, dot, extract


class RealDaemonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = Path(os.environ.get("LFM2D_BIN", ROOT / "target/release/lfm2d")).resolve()
        model = Path(os.environ.get("LFM2_MODELS_DIR", ROOT / ".models")) / "LFM2.5-Embedding-350M"
        if not binary.is_file() or not (model / "model.safetensors").is_file():
            raise RuntimeError("build lfm2d and supply LFM2.5-Embedding-350M via LFM2_MODELS_DIR")
        temporary = tempfile.TemporaryDirectory(prefix="lfm2d-demo-")
        cls.addClassCleanup(temporary.cleanup)
        socket_path = Path(temporary.name) / "lfm2d.sock"
        cls.endpoint = f"unix://{socket_path}"
        cls.requested_device = os.environ.get("LFM2D_TEST_DEVICE", "cpu")
        cls.expected_device = os.environ.get("LFM2D_EXPECT_DEVICE", cls.requested_device)
        if cls.expected_device not in ("cpu", "rocm", "cuda", "metal", "sycl"):
            raise RuntimeError("LFM2D_EXPECT_DEVICE must name an actual backend, not auto")
        log_path = Path(temporary.name) / "server.log"
        log = log_path.open("w")
        cls.addClassCleanup(log.close)
        # Do not inherit production model selection or telemetry destinations.
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LFM2D_", "OTEL_"))}
        child = subprocess.Popen([str(binary), "--embedder-dir", str(model.resolve()),
                                  "--socket-path", str(socket_path), "--threads", "8",
                                  "--device", cls.requested_device, "--device-index",
                                  os.environ.get("LFM2D_TEST_DEVICE_INDEX", "0")],
                                 stdout=log, stderr=log, env=env)
        def stop():
            # A graceful stop is part of the device gate: on 2026-09-13 a ROCm
            # build logged a clean drain and then segfaulted in driver teardown.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)  # 10s drain cap + 5s worker wait, with slack
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise RuntimeError(f"daemon ignored SIGTERM for 30s:\n{log_path.read_text()}")
            if child.returncode != 0:
                raise RuntimeError(f"daemon exited {child.returncode} on SIGTERM, not 0:\n{log_path.read_text()}")
        cls.addClassCleanup(stop)
        deadline = time.monotonic() + 60
        while not socket_path.exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"daemon failed to start:\n{log_path.read_text()}")
            time.sleep(0.05)
        cls.client = Client(cls.endpoint)
        cls.log_path = log_path

    def test_actual_execution_backend_is_reported(self):
        # These fields are sourced from the selected engine, not host inventory.
        # tracing's fmt layer colors fields even when writing to a file:
        # `backend\x1b[0m\x1b[2m=\x1b[0mcpu`. Match the text, not the escapes.
        log = re.sub(r"\x1b\[[0-9;]*m", "", self.log_path.read_text())
        self.assertIn(f"backend={self.expected_device}", log)
        self.assertIn(f"device_type={'cpu' if self.expected_device == 'cpu' else 'gpu'}", log)
        self.assertIn("dtype=f32", log)

    def test_batch_order_prefixes_and_model_headers(self):
        client = self.client
        self.assertEqual(client.model["hidden_size"], 1024)
        texts = ["A mutex protects shared data from simultaneous writes.",
                 "A sourdough starter ferments flour and water."]
        batch = client.embed(texts, "document")
        for text, vector in zip(texts, batch):
            single = client.embed([text], "document")[0]
            self.assertLess(max(abs(a - b) for a, b in zip(vector, single)), 1e-6)
        query = client.embed([texts[0]], "query")[0]
        self.assertLess(dot(query, batch[0]), 0.98, "query prefix was lost")
        self.assertGreater(dot(query, batch[0]), dot(query, batch[1]))
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            client.request("/embed", {"inputs": []})

    def test_retrieval_quality_over_http(self):
        corpus = json.loads((ROOT / "tests/data/semantic_search_eval.json").read_text())
        documents = corpus["documents"]
        vectors = self.client.embed([d["text"] for d in documents], "document")
        queries = self.client.embed([q["text"] for q in corpus["queries"]], "query")
        hits1 = hits3 = hard_wins = hard_total = 0
        for case, query in zip(corpus["queries"], queries):
            scores = {d["id"]: dot(query, v) for d, v in zip(documents, vectors)}
            order = sorted(scores, key=scores.get, reverse=True)
            hits1 += order[0] in case["relevant"]
            hits3 += bool(set(order[:3]) & set(case["relevant"]))
            best = max(scores[d] for d in case["relevant"])
            for negative in case["hard_negatives"]:
                hard_wins += scores[negative] > best
                hard_total += 1
        count = len(queries)
        print(f"\nHTTP retrieval: R@1={hits1}/{count}, R@3={hits3}/{count}, "
              f"hard-negative wins={hard_wins}/{hard_total}", flush=True)
        self.assertGreaterEqual(hits1 / count, 0.80)
        self.assertGreaterEqual(hits3 / count, 0.95)
        self.assertLessEqual(hard_wins / hard_total, 0.10)

    def test_python_cli_and_extractive_provenance(self):
        fixture = ROOT / "demo/conversation.txt"
        result = subprocess.run([os.sys.executable, str(ROOT / "demo/lfm2.py"),
                                 "--endpoint", self.endpoint, "search", "What is still undecided?",
                                 "--file", str(fixture), "--top", "6"],
                                capture_output=True, text=True, timeout=60, check=True)
        self.assertIn("storage schema", result.stdout)
        # This CLI contract test checks coverage, including the final paragraph.
        # Ranking quality has its own full-corpus test above. The open-ended
        # question here misses the intended passage at rank 1 (see README).
        self.assertEqual(result.stdout.count(f"[{fixture}:"), 6)
        self.assertIn(self.client.model["weight_hash"], result.stderr)
        source = fixture.read_text()
        parts = chunks(source)
        vectors = self.client.embed([p.text for p in parts], "document")
        selected = extract(parts, vectors, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, sorted(selected, key=lambda p: p.start))
        for part in selected:
            self.assertEqual(source[part.start:part.end], part.text)

        # Exercise splitting over HTTP and the extract CLI with original CRLF
        # offsets. The original fixture's short paragraphs don't cover this.
        with tempfile.TemporaryDirectory(prefix="lfm2d-source-") as directory:
            long_source = ("Shared memory needs synchronization. " * 35 +
                           "\r\n\r\nFinal decision: protect writes with a mutex.\r\n")
            path = Path(directory) / "long.txt"
            path.write_bytes(long_source.encode())
            pieces = chunks(long_source)
            self.assertGreater(len(pieces), 3)
            rendered = subprocess.run([
                os.sys.executable, str(ROOT / "demo/lfm2.py"), "--endpoint", self.endpoint,
                "extract", str(path), "--count", str(len(pieces)),
            ], capture_output=True, text=True, timeout=60, check=True)
            for piece in pieces:
                self.assertIn(f"[{path}:{piece.start}-{piece.end}]\n{piece.text}", rendered.stdout)
            self.assertIn("Final decision", rendered.stdout)

    def test_keyphrases_select_only_prose_from_the_requested_block(self):
        from keyphrases import keyphrases, select_block
        blocks = json.loads((ROOT / "demo/keyphrase_blocks.json").read_text())
        block = select_block(blocks, "search-thinking")
        phrases = keyphrases(block, self.client)
        self.assertEqual(len(phrases), 3)
        for phrase in phrases:
            self.assertEqual(phrase["block_id"], block["id"])
            self.assertEqual(phrase["text"], block["content"][phrase["start"]:phrase["end"]])
            self.assertNotIn("orchestration_registry", phrase["text"])


if __name__ == "__main__":
    unittest.main()
