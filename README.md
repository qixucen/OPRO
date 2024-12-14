# Enhanced OPRO Implementation

This is an enhanced version of the [OPRO (Large Language Models as Optimizers)](https://github.com/google-deepmind/opro) library, providing additional features and improved usability.

## Features

- **Configurable Endpoints**: Customize base URL and API key for different LLM providers
- **Enhanced Dataset Interface**: Support for multiple data formats (JSONL, JSON, Pandas DataFrame)
- **Improved Error Handling**: Robust validation and error reporting
- **Progress Tracking**: Visual progress bars for optimization
- **Flexible Configuration**: Pydantic-based configuration management
- **Extensible Metrics**: Support for custom evaluation metrics

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from opro.config import OPROConfig
from opro.api import OPRO
from opro.dataset import Dataset

# Configure OPRO
config = OPROConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key-here"
)

# Initialize OPRO
opro = OPRO(config)

# Load dataset
dataset = Dataset.from_jsonl("path/to/your/data.jsonl")

# Optimize prompts
result = opro.optimize(
    dataset=dataset,
    metric="accuracy",
    n_trials=10
)

print(f"Best prompt: {result.best_prompt}")
print(f"Best score: {result.best_score}")
```

### Dataset Formats

The enhanced OPRO supports multiple dataset formats:

1. JSONL format:
```python
dataset = Dataset.from_jsonl("data.jsonl")
```

2. JSON format:
```python
dataset = Dataset.from_json("data.json")
```

3. Pandas DataFrame:
```python
import pandas as pd
df = pd.DataFrame([{"input": "text", "output": "label"}])
dataset = Dataset.from_pandas(df)
```

## Configuration Options

- `base_url`: API endpoint URL (default: "https://api.openai.com/v1")
- `api_key`: Your API key
- `model`: Model to use (default: "gpt-3.5-turbo")
- `max_tokens`: Maximum tokens in completion (default: 150)
- `temperature`: Sampling temperature (default: 0.7)
- `timeout`: API request timeout in seconds (default: 30)

## Examples

Check the `examples/` directory for more detailed usage examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same terms as the original OPRO project.

## Usage

### Basic Example


### Dataset Formats

The enhanced OPRO supports multiple dataset formats:

1. JSONL format:

2. Pandas DataFrame:

3. JSON format:

### Configuration Options


## Examples

Check out the `examples/` directory for more detailed usage examples:
- `basic_usage.py`: Demonstrates basic functionality and different dataset formats
- More examples coming soon...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Based on the original [OPRO](https://github.com/google-deepmind/opro) implementation by Google DeepMind.
