# opro/api.py
import requests
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from .config import OPROConfig
from .dataset import Dataset

class OptimizationResult:
    def __init__(self, best_prompt: str, best_score: float, all_prompts: List[str], all_scores: List[float]):
        self.best_prompt = best_prompt
        self.best_score = best_score
        self.all_prompts = all_prompts
        self.all_scores = all_scores

class OPRO:
    """Enhanced OPRO implementation with configurable endpoints."""
    
    def __init__(self, config: OPROConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate API configuration."""
        if not self.config.api_key:
            raise ValueError("API key is required")
    
    def _make_request(self, prompt: str, input_text: str) -> str:
        """Make API request with configurable endpoint."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.config.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def evaluate_prompt(self, prompt: str, dataset: Dataset, metric: str = "accuracy") -> float:
        """Evaluate a prompt with multiple metric options."""
        if metric == "f1":
            return self._evaluate_f1(prompt, dataset)
        elif metric == "code":
            return self._evaluate_code(prompt, dataset)
        elif metric == "number_included_accuracy":
            return self._evaluate_number_accuracy(prompt, dataset)
    
    def optimize(
        self,
        dataset: Dataset,
        metric: str = "f1",
        n_trials: int = 10,
        initial_prompt: Optional[str] = None
    ) -> OptimizationResult:
        """Optimize prompt using the dataset."""
        if initial_prompt is None:
            initial_prompt = "Solve the problem."
        
        prompts = [initial_prompt]
        scores = []
        
        best_prompt = initial_prompt
        best_score = float('-inf')
        
        for _ in tqdm(range(n_trials), desc="Optimizing prompt"):
            # Generate variations of the best prompt
            current_prompt = self._generate_prompt_variation(best_prompt)
            current_score = self.evaluate_prompt(current_prompt, dataset, metric)
            
            prompts.append(current_prompt)
            scores.append(current_score)
            
            if current_score > best_score:
                best_score = current_score
                best_prompt = current_prompt
        
        return OptimizationResult(best_prompt, best_score, prompts, scores)
    
    def _generate_prompt_variation(self, base_prompt: str) -> str:
        """Generate variations of the base prompt using more sophisticated strategies."""
        variations = [
            "Let's approach this step by step. " + base_prompt,
            "Think carefully and " + base_prompt,
            base_prompt + " Be precise and accurate.",
            "Given the context, " + base_prompt
        ]
        
        return np.random.choice(variations)