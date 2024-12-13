# tests/test_api.py
import unittest
from unittest.mock import patch, MagicMock
from opro.config import OPROConfig
from opro.api import OPRO, OptimizationResult
from opro.dataset import Dataset

class TestOPRO(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.config = OPROConfig(
            base_url="https://oneapi.deepwisdom.ai/v1",
            api_key="sk-7kroxV4tTVYWDNFw8c1b40Cb88684d5d88E49dA9Fe413737"
        )
        self.opro = OPRO(self.config)
        self.test_dataset = Dataset([
            {"input": "test input", "output": "test output"}
        ])

    def test_initialization(self):
        """Test OPRO initialization."""
        self.assertEqual(self.opro.config.base_url, "https://oneapi.deepwisdom.ai/v1")
        self.assertEqual(self.opro.config.api_key, "sk-7kroxV4tTVYWDNFw8c1b40Cb88684d5d88E49dA9Fe413737")

    @patch('requests.post')
    def test_make_request(self, mock_post):
        """Test API request functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_post.return_value = mock_response

        response = self.opro._make_request("test prompt", "test input")
        self.assertEqual(response, "test response")

        # Test failed request
        mock_response.status_code = 400
        mock_response.text = "Error"
        with self.assertRaises(Exception):
            self.opro._make_request("test prompt", "test input")

    @patch('opro.api.OPRO._make_request')
    def test_evaluate_prompt(self, mock_make_request):
        """Test prompt evaluation."""
        mock_make_request.return_value = "test output"
        score = self.opro.evaluate_prompt(
            "test prompt",
            self.test_dataset,
            "accuracy"
        )
        self.assertEqual(score, 1.0)

        # Test invalid metric
        with self.assertRaises(ValueError):
            self.opro.evaluate_prompt(
                "test prompt",
                self.test_dataset,
                "invalid_metric"
            )

    @patch('opro.api.OPRO.evaluate_prompt')
    def test_optimize(self, mock_evaluate):
        """Test prompt optimization."""
        mock_evaluate.return_value = 0.8
        result = self.opro.optimize(
            dataset=self.test_dataset,
            metric="accuracy",
            n_trials=2
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(hasattr(result, 'best_prompt'))
        self.assertTrue(hasattr(result, 'best_score'))
        self.assertTrue(hasattr(result, 'all_prompts'))
        self.assertTrue(hasattr(result, 'all_scores'))

if __name__ == '__main__':
    unittest.main()