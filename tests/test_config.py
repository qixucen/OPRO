# tests/test_config.py
import unittest
from opro.config import OPROConfig
from pydantic import ValidationError

class TestOPROConfig(unittest.TestCase):
    def test_default_config(self):
        """Test default configuration values."""
        config = OPROConfig(api_key="test-key")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.max_tokens, 150)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.timeout, 30)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OPROConfig(
            base_url="https://custom-api.com",
            api_key="custom-key",
            model="gpt-4",
            max_tokens=200,
            temperature=0.5,
            timeout=60
        )
        self.assertEqual(config.base_url, "https://custom-api.com")
        self.assertEqual(config.api_key, "custom-key")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.max_tokens, 200)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.timeout, 60)

    def test_missing_api_key(self):
        """Test configuration validation for missing API key."""
        with self.assertRaises(ValidationError):
            OPROConfig()

if __name__ == '__main__':
    unittest.main()