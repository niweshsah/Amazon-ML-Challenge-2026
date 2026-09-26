from __future__ import annotations

import csv
import tempfile
import unittest
import random
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import torch

from core import f05, metrics, serialize, tags
from prepare import prepare
from train import group_batches, restore_training_state
from evaluate import score


class PipelineTests(unittest.TestCase):
    def test_unicode_serialization(self):
        text = serialize({"business_name": "  राम\n मार्केट ", "business_address": "  १२ रोड\tदिल्ली ",
                          "country": " India "})
        self.assertEqual(text, "Name: राम मार्केट | Address: १२ रोड दिल्ली | Country: India")
        self.assertNotIn("mixed_script", tags({"business_name": "राम", "business_address": "१२ रोड", "country": "India"}))

    def test_f05_multi_match_and_singleton(self):
        self.assertAlmostEqual(f05({"a", "b"}, {"a"}), 1.25 * 1 * .5 / (.25 + .5))
        self.assertEqual(f05(set(), set()), 1.0)
        self.assertEqual(f05(set(), {"x"}), 0.0)
        queries = [{"truth": "a,b", "tags": "multi"}, {"truth": "", "tags": "singleton"}]
        result = metrics(queries, [["a", "x"], ["x"]], np.array([[.9, .2], [.1, 0]]), .5, 2)
        self.assertAlmostEqual(result["macro_f0.5"], (f05({"a", "b"}, {"a"}) + 1) / 2)
        self.assertEqual(result["singleton_accuracy"], 1)

    def test_distinct_group_batches(self):
        rows = [{"anchor_id": str(i // 3)} for i in range(60)]
        for batch in group_batches(rows, 8):
            self.assertEqual(len({rows[i]["anchor_id"] for i in batch}), 8)

    def test_faiss_exact_rank(self):
        vectors = np.array([[1, 0], [0, 1], [.8, .6]], dtype="float32")
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        queries = np.array([[1, 0], [0, 1]], dtype="float32")
        scores, ranks = index.search(queries, 3)
        exact = np.argsort(-(queries @ vectors.T), axis=1)
        np.testing.assert_array_equal(ranks, exact)
        np.testing.assert_allclose(scores, np.take_along_axis(queries @ vectors.T, exact, axis=1))

    def test_checkpoint_resume_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
            state = {"step": 1, "best_score": .4, "best_step": 1,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.get_rng_state(),
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "max_length": 96, "batch_size": 128, "max_pairs": 200000}
            path = Path(tmp) / "state.pt"
            torch.save(state, path)
            expected = torch.rand(1).item()
            other = torch.optim.AdamW(model.parameters(), lr=.01)
            other_scheduler = torch.optim.lr_scheduler.LambdaLR(other, lambda step: 1 - step / 10)
            restored = restore_training_state(path, other, other_scheduler, 96, 128, 200000)
            self.assertEqual(restored["step"], 1)
            self.assertEqual(other_scheduler.last_epoch, scheduler.last_epoch)
            self.assertEqual(torch.rand(1).item(), expected)
            with self.assertRaises(ValueError):
                restore_training_state(path, other, other_scheduler, 64, 128, 200000)

    def test_scored_tsv_passes_repository_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = [{"entity_id": f"S1-{i}", "text": "Name: Test", "truth": "S2-0" if i % 2 else "",
                        "tags": "singleton" if i % 2 == 0 else "latin"} for i in range(6)]
            targets = [{"entity_id": f"S2-{i}", "text": f"Name: Target {i}"} for i in range(21)]
            pq.write_table(pa.Table.from_pylist(queries), root / "eval_queries.parquet")
            pq.write_table(pa.Table.from_pylist(targets), root / "eval_targets.parquet")
            np.save(root / "retrieval_positions.npy", np.tile(np.arange(20), (6, 1)))
            np.save(root / "retrieval_scores.npy", np.tile(np.linspace(.9, .1, 20), (6, 1)))
            score(root, lambda *args, **kwargs: None, top_k=20, calibration_count=1)
            validator = Path(__file__).resolve().parents[2] / "student_resource/utils/validate_submission.py"
            result = subprocess.run([sys.executable, str(validator), "--matching", str(root / "matching_results.tsv"),
                                     "--candidate", str(root / "candidate_pairs.tsv"), "--test-dir",
                                     str(root / "validation_dataset"), "--check-ids"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_split_and_pool_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data, out = base / "data", base / "out"
            data.mkdir()
            source1 = []
            source2 = []
            truth = []
            for i in range(40):
                sid = f"S1-{i}"
                source1.append({"entity_id": sid, "business_name": f"Shop {i}",
                                "business_address": f"Road {i}", "country": "India"})
                matches = [] if i % 7 == 0 else [f"S2-{i * 2}", f"S2-{i * 2 + 1}"]
                for tid in matches:
                    source2.append({"entity_id": tid, "business_name": f"Shop {i}",
                                    "business_address": f"Road {i}", "country": "India"})
                truth.append({"source1_entity_id": sid, "matched_entity_ids": ",".join(matches)})
            for i in range(100, 130):
                source2.append({"entity_id": f"S2-{i}", "business_name": f"Other {i}",
                                "business_address": "Elsewhere", "country": "India"})
            def write(name, rows):
                with (data / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
            write("train_source1.tsv", source1)
            write("train_source2.tsv", source2)
            write("train_source3.tsv", [{"entity_id": "S3-999", "business_name": "Other",
                                         "business_address": "Elsewhere", "country": "India"}])
            write("train_ground_truth.tsv", truth)
            manifest = prepare(out, data, eval_count=8, dev_count=3, pair_count=10, distractor_count=10)
            train = pq.read_table(out / "train_pairs.parquet").to_pylist()
            dev = pq.read_table(out / "dev_queries.parquet").to_pylist()
            evaluation = pq.read_table(out / "eval_queries.parquet").to_pylist()
            pool = {r["entity_id"] for r in pq.read_table(out / "eval_targets.parquet").to_pylist()}
            self.assertFalse({r["anchor_id"] for r in train} &
                             {r["entity_id"] for r in dev + evaluation})
            self.assertFalse({r["entity_id"] for r in dev} & {r["entity_id"] for r in evaluation})
            self.assertEqual(len(pool), manifest["eval_pool_size"])
            self.assertTrue(all(set(filter(None, q["truth"].split(","))) <= pool for q in evaluation))
            self.assertEqual(len({(r["anchor_id"], r["positive_id"]) for r in train}), len(train))
            self.assertTrue(all(r["negative_id"] != r["positive_id"] for r in train))


if __name__ == "__main__":
    unittest.main()
