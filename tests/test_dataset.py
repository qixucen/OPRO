# tests/test_dataset.py
import unittest
import tempfile
import json
import pandas as pd
from opro.dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.test_data = [
            {"input": "test input 1", "output": "test output 1"},
            {"input": "test input 2", "output": "test output 2"}
        ]

    def test_init_validation(self):
        """Test dataset initialization and validation."""
        dataset = Dataset(self.test_data)
        self.assertEqual(len(dataset), 2)
        
        # Test invalid data
        with self.assertRaises(ValueError):
            Dataset([{"input": "test"}])  # Missing output
        
        with self.assertRaises(ValueError):
            Dataset([])  # Empty dataset

    def test_from_jsonl(self):
        """Test loading dataset from JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in self.test_data:
                f.write(json.dumps(item) + '\n')
            temp_jsonl = f.name

        dataset = Dataset.from_jsonl(temp_jsonl)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], self.test_data[0])

    def test_from_pandas(self):
        """Test creating dataset from pandas DataFrame."""
        df = pd.DataFrame(self.test_data)
        dataset = Dataset.from_pandas(df)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], self.test_data[0])

        # Test invalid DataFrame
        invalid_df = pd.DataFrame([{"col1": "test"}])
        with self.assertRaises(ValueError):
            Dataset.from_pandas(invalid_df)

    def test_indexing(self):
        """Test dataset indexing."""
        dataset = Dataset(self.test_data)
        self.assertEqual(dataset[0], self.test_data[0])
        self.assertEqual(dataset[1], self.test_data[1])
        self.assertEqual(dataset[0:2], self.test_data[0:2])

if __name__ == '__main__':
    unittest.main()