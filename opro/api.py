# opro/api.py
import aiohttp
import asyncio
import random
import numpy as np
from typing import List, Optional
from tqdm import tqdm
from .config import OPROConfig
from .dataset import Dataset
from .utils import parse_tag_content, evaluate_f1, evaluate_code, evaluate_number


example_num = 3
max_retries = 3
resample_num = 8
total_optimization_prompt_tokens = 0
total_optimization_completion_tokens = 0
total_execution_prompt_tokens = 0
total_execution_completion_tokens = 0
optimization_log = []

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
        global total_optimization_prompt_tokens
        global total_optimization_completion_tokens
        global total_execution_prompt_tokens
        global total_execution_completion_tokens
        global optimization_log
        self.total_optimization_prompt_tokens = total_optimization_prompt_tokens
        self.total_optimization_completion_tokens = total_optimization_completion_tokens
        self.total_execution_prompt_tokens = total_execution_prompt_tokens
        self.total_execution_completion_tokens = total_execution_completion_tokens
        optimization_log = optimization_log

class OPRO:
    """Enhanced OPRO implementation with configurable endpoints."""

    def __init__(self, config: OPROConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate API configuration."""
        if not self.config.api_key:
            raise ValueError("API key is required")

    async def _llm_gen(self, prompt: str, mode:str) -> str:
        """Make async API request with configurable endpoint."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.config.optimization_model if mode == "optimization" else self.config.execution_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        errors = []
        MAX_RETRIES = 3
        DEFAULT_RETRY_AFTER = 1

        for retry in range(MAX_RETRIES):
            try:
                async with asyncio.timeout(30):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.config.base_url}/chat/completions",
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status != 200:
                                raise Exception(f"API request failed: {await response.text()}")
                            result = await response.json()
                            if mode == "optimization":
                                global total_optimization_prompt_tokens
                                global total_optimization_completion_tokens
                                total_optimization_prompt_tokens += result["usage"]["prompt_tokens"]
                                total_optimization_completion_tokens += result["usage"]["completion_tokens"]
                            elif mode == "execution":
                                global total_execution_prompt_tokens
                                global total_execution_completion_tokens
                                total_execution_prompt_tokens += result["usage"]["prompt_tokens"]
                                total_execution_completion_tokens += result["usage"]["completion_tokens"]
                            return result["choices"][0]["message"]["content"]
                            
            except asyncio.TimeoutError:
                errors.append("Request timeout")
            except aiohttp.ClientError as e:
                errors.append(f"Client error: {str(e)}")
            except Exception as e:
                errors.append(f"Error: {type(e).__name__}, {str(e)}")
            
            await asyncio.sleep(DEFAULT_RETRY_AFTER * (2 ** retry))
        
        print(f"Error log: {errors}")
        raise Exception("Max retries exceeded")
    
    async def evaluate_instruction(
        self, instruction: str, dataset: Dataset, metric_name: str = "f1"
    ) -> float:
        """Evaluate a prompt with multiple metric options."""
        answers = await asyncio.gather(*[self.generate_answer(instruction, item["input"]) for item in dataset])
        if metric_name == "f1":
            metric = evaluate_f1
        elif metric_name == "code":
            metric = evaluate_code
        elif metric_name == "number":
            metric = evaluate_number
        scores = [metric(answer, item["output"]) for answer, item in zip(answers, dataset)]
        score = sum(scores) / len(scores)
        examples = [(item["input"], item["output"]) for item, score in zip(dataset, scores) if score < 1]
        examples = random.sample(examples, example_num)
        
        return score, examples
    
    def get_prompt(self, instruction: str, input_text: str):
        return f"Q: {input_text}\nA: {instruction}\n Generate the answer above. The answer should begin with <ANS> and end with </ANS>."
    
    def extract_instruction(self, response: str):
        return parse_tag_content(response, "<INS>", "</INS>")
    
    def extract_answer(self, response: str):
        return parse_tag_content(response, "<ANS>", "</ANS>")
    
    async def generate_answer(self, instruction: str, input_text: str):
        for _ in range(max_retries):
            response = await self._llm_gen(self.get_prompt(instruction, input_text), "execution")
            answer = self.extract_answer(response)
            if answer != []:
                answer = answer[0]
                break
            else:
                answer = ""
        return answer

    async def optimize(
        self,
        dataset: Dataset,
        metric_name: str = "f1",
        n_trials: int = 10,
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize prompt using the dataset."""
        if initial_prompt is None:
            initial_prompt = "Solve the problem."

        prompts = [initial_prompt]
        scores = []

        best_prompt = initial_prompt
        best_score = float("-inf")
        score, examples = await self.evaluate_instruction(initial_prompt, dataset, metric_name)
        instructions_and_scores = [(initial_prompt, score)]
        max_instructions = max(n_trials // 10, 3)
        optimization_log.append({
            "instruction": initial_prompt,
            "score": score,
            "examples": examples,
        })

        for _ in tqdm(range(n_trials), desc="Optimizing prompt"):
            # Generate variations of the best prompt
            meta_prompt = self.generate_meta_prompt(instructions_and_scores, examples=examples, max_instructions=max_instructions)
            async def sample_instruction(meta_prompt):
                for _ in range(max_retries):
                    current_instructions = await self._llm_gen(meta_prompt, "optimization")
                    current_instruction = self.extract_instruction(current_instructions)
                    if current_instruction != []:
                        current_instruction = current_instruction[0]
                        break
                    else:
                        current_instruction = "Solve the problem"
                return current_instruction
            current_instructions = await asyncio.gather(*[sample_instruction(meta_prompt) for _ in range(resample_num)])
            current_results = await asyncio.gather(*[self.evaluate_instruction(instruction, dataset, metric_name) for instruction in current_instructions])
            current_scores = [score for score, _ in current_results]
            current_score = max(current_scores)
            current_instruction = current_instructions[current_scores.index(current_score)]
            # print(current_score, current_instruction)
            current_score, examples = await self.evaluate_instruction(current_instruction, dataset, metric_name)
            optimization_log.append({
                "instruction": current_instruction,
                "score": current_score,
                "examples": examples,
            })

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
        max_instructions: int = 3,  # max_instructions = n_trials / 10
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
        max_instructions: int = 3,
        examples: List[list] = None,
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
        for item in examples:
            question = item[0]
            answer = item[1]
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