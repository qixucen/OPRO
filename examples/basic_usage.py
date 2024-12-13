# examples/basic_usage.py
from opro.config import OPROConfig
from opro.api import OPRO
from opro.dataset import Dataset
import pandas as pd
import json
import tempfile

def demonstrate_jsonl_dataset():
    """Demonstrate using JSONL dataset."""
    # Create a temporary JSONL file
    data = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is the capital of France?", "output": "Paris"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        temp_jsonl = f.name
    
    # Load dataset from JSONL
    dataset = Dataset.from_jsonl(temp_jsonl)
    return dataset

def demonstrate_pandas_dataset():
    """Demonstrate using pandas DataFrame dataset."""
    df = pd.DataFrame([
        {"input": "Translate 'hello' to French", "output": "bonjour"},
        {"input": "Translate 'goodbye' to French", "output": "au revoir"}
    ])
    dataset = Dataset.from_pandas(df)
    return dataset

def main():
    # Configure OPRO with custom settings
    config = OPROConfig(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7,
        timeout=30
    )
    
    # Initialize OPRO
    opro = OPRO(config)
    
    # Example 1: Using JSONL dataset
    print("Example 1: Using JSONL dataset")
    dataset_jsonl = demonstrate_jsonl_dataset()
    
    # Optimize prompt with JSONL dataset
    result_jsonl = opro.optimize(
        dataset=dataset_jsonl,
        metric="accuracy",
        n_trials=5,
        initial_prompt="You are a helpful assistant. Answer the following question:"
    )
    
    print(f"Best prompt (JSONL): {result_jsonl.best_prompt}")
    print(f"Best score (JSONL): {result_jsonl.best_score}")
    
    # Example 2: Using pandas DataFrame dataset
    print("\nExample 2: Using pandas DataFrame dataset")
    dataset_pandas = demonstrate_pandas_dataset()
    
    # Optimize prompt with pandas dataset
    result_pandas = opro.optimize(
        dataset=dataset_pandas,
        metric="accuracy",
        n_trials=5,
        initial_prompt="You are a language translator. Please translate:"
    )
    
    print(f"Best prompt (Pandas): {result_pandas.best_prompt}")
    print(f"Best score (Pandas): {result_pandas.best_score}")
    
    # Example 3: Custom evaluation with different configuration
    print("\nExample 3: Custom evaluation with different configuration")
    # Create a new configuration with different settings
    custom_config = OPROConfig(
        base_url="https://custom-endpoint.com/v1",
        api_key="custom-api-key",
        model="gpt-4",
        temperature=0.5
    )
    
    custom_opro = OPRO(custom_config)
    
    # Evaluate a single prompt
    custom_prompt = "Translate the following text to French, providing only the translation:"
    score = custom_opro.evaluate_prompt(
        prompt=custom_prompt,
        dataset=dataset_pandas,
        metric="accuracy"
    )
    
    print(f"Custom prompt evaluation score: {score}")

if __name__ == "__main__":
    main()