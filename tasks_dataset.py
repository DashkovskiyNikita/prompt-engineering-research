import json
import csv
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from datasets import load_dataset

@dataclass
class Task:
    id: str
    domain: str
    difficulty: str
    input: str
    ground_truth: str
    evaluation_type: str
    context: Optional[str] = None
    metadata: Optional[dict] = None

def extract_gsm8k_answer(solution: str) -> str:
    match = re.search(r"####\s*(.+)$", solution, re.MULTILINE)
    return match.group(1).strip() if match else ""

def load_gsm8k(size: int = 100) -> List[Task]:
    ds = load_dataset("gsm8k", "main", split="test")
    tasks = []
    for i, item in enumerate(ds.select(range(min(size, len(ds))))):
        tasks.append(Task(
            id=f"gsm8k_{i:03d}",
            domain="math",
            difficulty="medium",
            input=item["question"],
            ground_truth=extract_gsm8k_answer(item["answer"]),
            evaluation_type="exact_match",
            context=None,
            metadata={"source": "gsm8k", "full_solution": item["answer"]}
        ))
    return tasks

def load_samsum(size: int = 100) -> List[Task]:
    ds = load_dataset("Samsung/samsum", split="test")
    tasks = []
    for i, item in enumerate(ds.select(range(min(size, len(ds))))):
        tasks.append(Task(
            id=f"samsum_{i:03d}",
            domain="summarization",
            difficulty="medium",
            input="Составь краткое резюме диалога в 1-2 предложениях.",
            ground_truth=item["summary"],
            evaluation_type="llm_judge",
            context=item["dialogue"],
            metadata={"source": "samsum", "criteria": ["точность", "краткость", "сохранение смысла"]}
        ))
    return tasks

def load_humaneval(size: int = 100) -> List[Task]:
    ds = load_dataset("openai_humaneval", split="test")
    tasks = []
    for i, item in enumerate(ds.select(range(min(size, len(ds))))):
        tasks.append(Task(
            id=f"humaneval_{i:03d}",
            domain="code",
            difficulty="medium",
            input=item["prompt"],
            ground_truth=item["canonical_solution"],
            evaluation_type="code_test",
            context=None,
            metadata={
                "source": "humaneval",
                "test": item["test"],
                "entry_point": item["entry_point"]
            }
        ))
    return tasks

def load_triviaqa(size: int = 100) -> List[Task]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    tasks = []
    for i, item in enumerate(ds.select(range(min(size, len(ds))))):
        answer = item["answer"]["value"]
        aliases = item["answer"].get("aliases", [])
        tasks.append(Task(
            id=f"triviaqa_{i:03d}",
            domain="qa",
            difficulty="medium",
            input=item["question"],
            ground_truth=answer,
            evaluation_type="contains",
            context=None,
            metadata={"source": "triviaqa", "aliases": aliases}
        ))
    return tasks

def load_all_datasets(size_per_domain: int = 100) -> List[Task]:
    all_tasks = []
    all_tasks.extend(load_gsm8k(size_per_domain))
    all_tasks.extend(load_samsum(size_per_domain))
    all_tasks.extend(load_humaneval(size_per_domain))
    all_tasks.extend(load_triviaqa(size_per_domain))
    return all_tasks

def save_dataset(tasks: List[Task], output_dir: str = "public_dataset") -> None:
    path = Path(output_dir)
    path.mkdir(exist_ok=True)

    json_path = path / "public_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tasks], f, ensure_ascii=False, indent=2)

    csv_path = path / "public_dataset.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "domain", "difficulty", "input", "context",
            "ground_truth", "evaluation_type", "metadata"
        ])
        writer.writeheader()
        for t in tasks:
            row = asdict(t)
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            writer.writerow(row)

    from collections import Counter
    domains = Counter(t.domain for t in tasks)
    print(f"Всего задач: {len(tasks)}")
    print(f"По доменам: {dict(domains)}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

if __name__ == "__main__":
    dataset = load_all_datasets(size_per_domain=100)
    save_dataset(dataset)