# main.py
from opro.config import OPROConfig
from opro.api import OPRO
from opro.dataset import Dataset
from config import base_url, api_key

def main():
    # Example usage of enhanced OPRO
    config = OPROConfig(
        base_url=base_url,
        api_key=api_key
    )
    
    # Initialize OPRO with config
    opro = OPRO(config)
    
    # Example dataset
    dataset = Dataset("hotpotqa")
    
    # Run optimization
    result = opro.optimize(
        dataset=dataset.get_data(),
        metric="f1",
        n_trials=10
    )
    
    print("Optimization Results:")
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best score: {result.best_score}")

if __name__ == "__main__":
    main()