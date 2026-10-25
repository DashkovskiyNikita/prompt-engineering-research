import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptVariation:
    task_id: str
    method: str
    prompt: str
    metadata: dict

class PromptBuilder:
    def __init__(self, few_shot_examples: Optional[List[Dict]] = None):
        self.few_shot_examples = few_shot_examples or []
    
    def zero_shot(self, task: Dict) -> str:
        if task.get("context"):
            return f"{task['context']}\n\n{task['input']}"
        return task["input"]
    
    def few_shot(self, task: Dict) -> str:
        examples = random.sample(self.few_shot_examples, min(3, len(self.few_shot_examples)))
        examples_text = "\n\n".join([
            f"Example {i+1}:\nQuestion: {ex['input']}\nAnswer: {ex['ground_truth']}"
            for i, ex in enumerate(examples)
        ])
        
        if task.get("context"):
            return f"{examples_text}\n\nNow solve the following task:\n\n{task['context']}\n\n{task['input']}"
        return f"{examples_text}\n\nNow solve the following task:\n\n{task['input']}"
    
    def chain_of_thought(self, task: Dict) -> str:
        instruction = "Let's think step by step. First, analyze the task, then solve it by explaining each step."
        if task.get("context"):
            return f"{instruction}\n\n{task['context']}\n\n{task['input']}"
        return f"{instruction}\n\n{task['input']}"
    
    def role_prompting(self, task: Dict) -> str:
        roles = {
            "math": "You are a mathematics expert with 20 years of experience solving complex problems.",
            "summarization": "You are a professional editor specializing in creating concise and accurate summaries.",
            "code": "You are a senior Python developer with deep knowledge of algorithms and best practices.",
            "qa": "You are an erudite expert with extensive knowledge across various fields."
        }
        role = roles.get(task["domain"], "You are an experienced assistant.")
        
        if task.get("context"):
            return f"{role}\n\n{task['context']}\n\n{task['input']}"
        return f"{role}\n\n{task['input']}"
    
    def structured(self, task: Dict) -> str:
        parts = [f"<role>{self._get_role(task['domain'])}</role>"]
        
        if task.get("context"):
            parts.append(f"<context>\n{task['context']}\n</context>")
        
        parts.append(f"<task>\n{task['input']}\n</task>")
        parts.append("<instructions>Provide a precise and complete answer, following all instructions.</instructions>")
        parts.append("<format>The answer must be clear and structured.</format>")
        
        return "\n\n".join(parts)
    
    def _get_role(self, domain: str) -> str:
        roles = {
            "math": "Mathematics Expert",
            "summarization": "Professional Editor",
            "code": "Senior Python Developer",
            "qa": "Erudite Expert"
        }
        return roles.get(domain, "Experienced Assistant")
    
    def generate_all_variations(self, task: Dict) -> List[PromptVariation]:
        variations = []
        
        methods = [
            ("zero_shot", self.zero_shot),
            ("few_shot", self.few_shot),
            ("chain_of_thought", self.chain_of_thought),
            ("role_prompting", self.role_prompting),
            ("structured", self.structured)
        ]
        
        for method_name, method_func in methods:
            prompt = method_func(task)
            variations.append(PromptVariation(
                task_id=task["id"],
                method=method_name,
                prompt=prompt,
                metadata={"domain": task["domain"], "difficulty": task["difficulty"]}
            ))
        
        return variations


def load_dataset_for_few_shot(dataset_path: str, domain: str, size: int = 10) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    domain_tasks = [t for t in data if t["domain"] == domain]
    return random.sample(domain_tasks, min(size, len(domain_tasks)))


def generate_prompts_for_dataset(dataset_path: str, output_path: str) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    few_shot_examples = {
        "math": load_dataset_for_few_shot(dataset_path, "math", 10),
        "summarization": load_dataset_for_few_shot(dataset_path, "summarization", 10),
        "code": load_dataset_for_few_shot(dataset_path, "code", 10),
        "qa": load_dataset_for_few_shot(dataset_path, "qa", 10)
    }
    
    all_variations = []
    
    for task in dataset:
        builder = PromptBuilder(few_shot_examples.get(task["domain"], []))
        variations = builder.generate_all_variations(task)
        all_variations.extend(variations)
    
    output = Path(output_path)
    output.mkdir(exist_ok=True)
    
    with open(output / "prompt_variations.json", "w", encoding="utf-8") as f:
        json.dump([
            {
                "task_id": v.task_id,
                "method": v.method,
                "prompt": v.prompt,
                "metadata": v.metadata
            }
            for v in all_variations
        ], f, ensure_ascii=False, indent=2)
    
    from collections import Counter
    methods = Counter(v.method for v in all_variations)
    print(f"Total prompt variations: {len(all_variations)}")
    print(f"By method: {dict(methods)}")
    print(f"Saved to: {output / 'prompt_variations.json'}")

if __name__ == "__main__":
    generate_prompts_for_dataset("public_dataset.json", "prompts_output")