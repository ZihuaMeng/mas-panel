"""Check the pairing and provenance of a complete scripted panel."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mas_panel.__main__ import DemoBackend
from mas_panel.runner import inspect_request, read_jsonl, run_experiment


DATASET = Path(__file__).resolve().parents[1] / "examples" / "prefix_sum.json"


class RunnerTest(unittest.TestCase):
    def test_peer_and_private_branch_from_frozen_initials(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "panel"
            summary = run_experiment(DATASET, output, DemoBackend(), "main", seed=29,
                                     initialization="complementary")["overall"]
            calls = list(read_jsonl(output / "trace.jsonl"))
            self.assertEqual(len(calls), 16)
            self.assertEqual(len({call["call_id"] for call in calls}), 16)
            self.assertEqual(summary["private"]["last_correct_loss"], 1)
            self.assertEqual(summary["peer"]["wrong_vote_with_retained_candidate"], 1)
            for agent in range(4):
                requests = {row["request"]["stage"]: row["request"] for row in calls
                            if row["request"]["agent_id"] == agent}
                initial, private, peer = (requests[s] for s in ("initial", "private", "peer"))
                self.assertEqual(private["seed"], peer["seed"])
                self.assertEqual(private["messages"][:-1], peer["messages"][:-1])
                self.assertEqual(private["messages"][:-2], initial["messages"])
                self.assertNotIn("gold", peer)
                self.assertIn("Peer 1:", peer["messages"][-1]["content"])
                self.assertNotIn("Peer 1:", private["messages"][-1]["content"])

            source = next(call for call in calls if call["request"]["stage"] == "peer")
            replay = Path(temporary) / "replay"
            self.assertEqual(inspect_request(output / "trace.jsonl", source["call_id"], replay,
                                             DemoBackend(), [5, 9]), 2)
            replications = list(read_jsonl(replay / "trace.jsonl"))
            self.assertEqual([row["request"]["seed"] for row in replications], [5, 9])
            for row in replications:
                self.assertEqual(row["request"]["source_call_id"], source["call_id"])
                self.assertEqual(row["request"]["messages"], source["request"]["messages"])
            manifest = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["inputs"]["source_request_sha256"], source["request_sha256"])
