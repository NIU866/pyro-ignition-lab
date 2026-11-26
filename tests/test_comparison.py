import csv
import tempfile
import unittest
from pathlib import Path

from pyro_ignition_lab import (
    SlabParameters, comparison_points, comparison_statistics, export_comparison,
    read_experiment, write_experiment_template,
)


class ExperimentComparisonTests(unittest.TestCase):
    def test_import_compare_export_with_missing_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.csv"
            write_experiment_template(path)
            with path.open("a", encoding="utf-8", newline="") as file:
                csv.writer(file).writerows([(0, 100, "", 21, ""), (2.5, 97, "", "", ""), (7, 91, "", 40, "")])

            case = read_experiment(path, SlabParameters())
            predictions = ((100, 0, 19, 20), (98, 1, 34, 22), (90, 2, 44, 24))
            points = comparison_points(case, predictions, 2)
            n, rmse, mae, bias = comparison_statistics(points)
            self.assertEqual((n, mae, bias), (2, 3, 1))
            self.assertAlmostEqual(rmse, 10 ** .5)
            self.assertEqual(comparison_statistics(comparison_points(case, predictions, 3)),
                             (0, None, None, None))

            output = Path(tmp) / "errors.csv"
            export_comparison(output, 2, points)
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[1], {"time_s": "2.5", "observed_front_C": "",
                                       "predicted_front_C": "34", "residual_front_C": ""})
            self.assertEqual([float(rows[i]["residual_front_C"]) for i in (0, 2)], [-2, 4])


if __name__ == "__main__":
    unittest.main()
