import csv
import tempfile
import unittest
from pathlib import Path

from ml_benchmark.benchmark import make_models, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_expected_models(self):
        self.assertEqual(len(make_models(42)), 4)
        self.assertIn("Logistic Regression", make_models(42))

    def test_end_to_end_outputs_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as root:
            left = run_benchmark(output=Path(root) / "a", folds=3, repeats=1)
            right = run_benchmark(output=Path(root) / "b", folds=3, repeats=1)
            self.assertEqual(left["selected_model"], right["selected_model"])
            self.assertEqual(left["cv_results"], right["cv_results"])
            self.assertEqual(left["held_out_test_metrics"], right["held_out_test_metrics"])
            for file in ("cv_results.csv", "results.json", "REPORT.md", "model_comparison.png", "confusion_matrix.png"):
                self.assertTrue((Path(root) / "a" / file).is_file(), file)
            with (Path(root) / "a" / "cv_results.csv").open(newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            for score in left["held_out_test_metrics"].values():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

    def test_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run_benchmark(output=Path(root), folds=1)
            with self.assertRaises(ValueError):
                run_benchmark(output=Path(root), test_size=0.8)


if __name__ == "__main__":
    unittest.main()
