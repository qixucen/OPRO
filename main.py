# main.py
import json
import asyncio
from opro.config import OPROConfig
from opro.api import OPRO
from opro.dataset import Dataset
from config import base_url, api_key

async def main():
    # Example usage of enhanced OPRO
    config = OPROConfig(
        base_url=base_url,
        api_key=api_key,
        optimization_model="gpt-4o-mini",
        execution_model="gpt-4o-mini"
    )
    
    # Initialize OPRO with config
    opro = OPRO(config)
    
    # Example dataset
    dataset = Dataset("hotpotqa", "dataset/hotpotqa_train.jsonl")
    
    # Run optimization
    result = await opro.optimize(
        dataset=dataset.get_data(),
        metric_name="f1",
        n_trials=2,
    )
    
    print("Optimization Results:")
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best score: {result.best_score}")
    log = json.load(open("log.json", "r"))
    log.append({f"len(log)": len(log), "best_prompt": result.best_prompt, "best_score": result.best_score, "total_optimization_prompt_tokens": result.total_optimization_prompt_tokens, "total_optimization_completion_tokens": result.total_optimization_completion_tokens, "total_execution_prompt_tokens": result.total_execution_prompt_tokens, "total_execution_completion_tokens": result.total_execution_completion_tokens, "optimization_log": result.optimization_log})
    json.dump(log, open("log.json", "w"))

if __name__ == "__main__":
    asyncio.run(main())
