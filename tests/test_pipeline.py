import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate_data import make_dataset


class PipelineTests(unittest.TestCase):
    def test_dataset_shape_and_columns(self):
        df = make_dataset(100, seed=1)
        self.assertEqual(len(df), 100)
        self.assertIn('overreliance_label', df.columns)
        self.assertTrue(set(df['overreliance_label'].unique()).issubset({0, 1}))

    def test_feature_ranges(self):
        df = make_dataset(200, seed=2)
        self.assertTrue(df['revision_delta'].between(0, 1).all())
        self.assertTrue(df['self_reported_confidence'].between(0, 1).all())
        self.assertTrue(df['task_difficulty'].between(1, 5).all())


if __name__ == '__main__':
    unittest.main()
