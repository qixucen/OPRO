# opro/api.py
import aiohttp
import backoff
import asyncio
import random
from typing import List, Optional
from tqdm import tqdm
from .config import OPROConfig
from .dataset import Dataset
from .utils import parse_tag_content, evaluate_f1, evaluate_code, evaluate_number


example_num = 3
class OptimizationResult:
    def __init__(
        self,
        best_prompt: str,
        best_score: float,
        all_prompts: List[str],
        all_scores: List[float],
    ):
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

    async def _llm_gen(self, prompt: str) -> str:
        """Make async API request with configurable endpoint."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError))
        async def make_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API request failed: {await response.text()}")
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

        return await make_request()
    
    async def evaluate_instruction(
        self, instruction: str, dataset: Dataset, metric: str = "accuracy"
    ) -> float:
        """Evaluate a prompt with multiple metric options."""
        answers = await asyncio.gather(*[self.generate_answer(instruction, item["input"]) for item in dataset])
        if metric == "f1":
            metric = evaluate_f1
        elif metric == "code":
            metric = evaluate_code
        elif metric == "number":
            metric = evaluate_number
        scores = [metric(answer, item["output"]) for answer, item in zip(answers, dataset)]
        score = sum(scores) / len(scores)
        
        return score
    
    def get_prompt(self, instruction: str, input_text: str):
        return f"Q: {input_text}\nA: {instruction}\n Generate the answer above. The answer should begin with <ANS> and end with </ANS>."
    
    def extract_instruction(self, response: str):
        return parse_tag_content(response, "<INS>", "</INS>")
    
    def extract_answer(self, response: str):
        return parse_tag_content(response, "<ANS>", "</ANS>")
    
    async def generate_answer(self, instruction: str, input_text: str):
        while True:
            response = await self._llm_gen(self.get_prompt(instruction, input_text))
            answer = self.extract_answer(response)
            if answer != []:
                answer = answer[0]
                break
        return answer

    async def optimize(
        self,
        dataset: Dataset,
        metric: str = "f1",
        n_trials: int = 200,
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize prompt using the dataset."""
        if initial_prompt is None:
            initial_prompt = "Solve the problem."

        prompts = [initial_prompt]
        scores = []

        best_prompt = initial_prompt
        best_score = float("-inf")
        instructions_and_scores = [(initial_prompt, await self.evaluate_instruction(initial_prompt, dataset, metric))]

        for _ in tqdm(range(n_trials), desc="Optimizing prompt"):
            # Generate variations of the best prompt
            while True:
                current_instruction = await self._llm_gen(self.generate_meta_prompt(instructions_and_scores))
                current_instruction = self.extract_instruction(current_instruction)
                if current_instruction != []:
                    current_instruction = current_instruction[0]
                    break
            current_score = await self.evaluate_instruction(current_instruction, dataset, metric)

            prompts.append(current_instruction)
            scores.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_prompt = current_instruction
            
            instructions_and_scores.append((current_instruction, current_score))

        return OptimizationResult(best_prompt, best_score, prompts, scores)

    def generate_instruction_score_pairs(
        self,
        instructions_and_scores: List[tuple], 
        score_threshold: float = 0.1,
        max_instructions: int = 1000,
    ) -> str:
        """Generate formatted string of instruction-score pairs.
        
        Args:
            instructions_and_scores: List of (instruction, score, step) tuples
            score_threshold: Minimum score threshold to include instruction
            max_instructions: Maximum number of instructions to include
            
        Returns:
            Formatted string containing instruction-score pairs
        """
        # Sort by score and take top max_instructions
        sorted_pairs = sorted(instructions_and_scores, key=lambda x: x[1])[-max_instructions:]
        
        pairs_str = ""
        for instruction, score in sorted_pairs:
            if score >= score_threshold:
                score = round(score, 3)
                pairs_str += f"\ntext:\n{instruction}\nscore:\n{score * 100}\n"
                
        return pairs_str

    def generate_meta_prompt(
        self,
        instructions_and_scores: List[tuple],
        score_threshold: float = 0.1,
        max_instructions: int = 1000,
        dataset: Optional[Dataset] = None,
    ) -> str:
        """Generate meta prompt for instruction optimization."""
        # Generate instruction-score pairs string
        instruction_pairs = self.generate_instruction_score_pairs(
            instructions_and_scores,
            score_threshold,
            max_instructions, 
        )

        # Build meta prompt
        meta_prompt = (
            "Your task is to generate the instruction <INS>. "
            "Below are some previous instructions with their scores. "
            "The score ranges from 0 to 100.\n"
        )
        
        # Add instruction-score pairs
        meta_prompt += instruction_pairs

        # Add examples
        example_indices = random.sample(range(len(dataset)), example_num)
        if dataset and example_indices:
            meta_prompt += "\nBelow are some problems.\n"
            
            for idx in example_indices:
                question = dataset[idx]["input"]
                answer = dataset[idx]["output"]
                meta_prompt += f"\ninput:\nQ: <INS>\n{question}\nA:"
                meta_prompt += f"\nGround truth answer:\n{answer}\n"

        # Add final instruction
        meta_prompt += (
            "\n\nGenerate an instruction that is different from all the "
            "instructions <INS> above, and has a higher score than all the "
            "instructions <INS> above. The instruction should begin with <INS> "
            "and end with </INS>. The instruction should be concise, effective, "
            "and generally applicable to all problems above."
        )

        return meta_prompt