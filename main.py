# main.py
from opro.config import OPROConfig
from opro.api import OPRO
from opro.dataset import Dataset

def main():
    # Example usage of enhanced OPRO
    config = OPROConfig(
        base_url="https://oneapi.deepwisdom.ai/",
        api_key="sk-7kroxV4tTVYWDNFw8c1b40Cb88684d5d88E49dA9Fe413737"
    )
    
    # Initialize OPRO with config
    opro = OPRO(config)
    
    # Example dataset
    dataset = Dataset.from_json("dataset/hotpotqa.json")
    
    # Run optimization
    result = opro.optimize(
        dataset=dataset,
        metric="accuracy",
        n_trials=10
    )
    
    print("Optimization Results:")
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best score: {result.best_score}")

if __name__ == "__main__":
    main()