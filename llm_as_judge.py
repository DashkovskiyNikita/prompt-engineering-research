import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_completeness,
    answer_relevancy,
    faithfulness
)
from ragas.dataset_schema import SingleTurnSample

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")

DATASET_PATH = "public_dataset/public_dataset.json"
RESULTS_PATH = "deepseek_results.json"
OUTPUT_DIR = "evaluation_results"

def extract_reference(task: dict) -> str:
    domain = task["domain"]
    if domain == "math":
        return task.get("metadata", {}).get("full_solution", task["ground_truth"])
    return task["ground_truth"]

def extract_context(task: dict) -> list:
    context = task.get("context")
    if context:
        return [context]
    return []

def load_and_prepare_data():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = {item["id"]: item for item in json.load(f)}

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    samples = []
    metadata = []
    skipped = 0

    for r in results:
        if r["status"] == "failed":
            skipped += 1
            continue

        task = dataset.get(r["task_id"])
        if not task:
            skipped += 1
            continue

        samples.append(SingleTurnSample(
            user_input=task["input"],
            retrieved_contexts=extract_context(task),
            reference=extract_reference(task),
            response=r["response"]
        ))

        metadata.append({
            "task_id": r["task_id"],
            "method": r["method"],
            "domain": task["domain"],
            "difficulty": task["difficulty"],
            "attempts": r["attempts"]
        })

    print(f"Подготовлено семплов: {len(samples)}, пропущено: {skipped}")
    return samples, metadata

def evaluate_dataset(samples, metadata):
    evaluation_result = evaluate(
        dataset=samples,
        metrics=[answer_correctness, answer_completeness, answer_relevancy, faithfulness],
        llm="gpt-4o-mini"
    )

    df_metrics = pd.DataFrame(evaluation_result)
    df_meta = pd.DataFrame(metadata)
    return pd.concat([df_meta.reset_index(drop=True), df_metrics.reset_index(drop=True)], axis=1)

def plot_results(df):
    metrics_cols = ["answer_correctness", "answer_completeness", "answer_relevancy", "faithfulness"]
    df_grouped = df.groupby(["method", "domain"])[metrics_cols].mean().reset_index()

    plt.figure(figsize=(20, 10))
    sns.set_theme(style="whitegrid")

    melted_df = df_grouped.melt(
        id_vars=["method", "domain"],
        value_vars=metrics_cols,
        var_name="metric",
        value_name="score"
    )

    metric_labels = {
        "answer_correctness": "Точность",
        "answer_completeness": "Полнота",
        "answer_relevancy": "Релевантность",
        "faithfulness": "Отсутствие галлюцинаций"
    }
    melted_df["metric_ru"] = melted_df["metric"].map(metric_labels)

    g = sns.catplot(
        data=melted_df,
        x="method",
        y="score",
        hue="domain",
        col="metric_ru",
        kind="bar",
        height=5,
        aspect=1.1,
        palette="viridis",
        errorbar=None,
        col_wrap=2
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Метод промпта", "Средняя оценка (0-1)")
    g.set(ylim=(0, 1))

    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "evaluation_all_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(14, 7))
    avg_by_method = df.groupby("method")[metrics_cols].mean()
    avg_by_method.columns = [metric_labels[c] for c in avg_by_method.columns]

    sns.heatmap(avg_by_method, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Средняя оценка"})
    plt.title("Средние метрики по методам промптинга")
    plt.xlabel("Метрика")
    plt.ylabel("Метод промпта")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "heatmap_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

    return df_grouped

def save_reports(df, df_grouped):
    output = Path(OUTPUT_DIR)
    output.mkdir(exist_ok=True)

    df.to_csv(output / "full_evaluation.csv", index=False, encoding="utf-8")
    df_grouped.to_csv(output / "metrics_by_method_domain.csv", index=False, encoding="utf-8")

    overall = df.groupby("method")[["answer_correctness", "answer_completeness", "answer_relevancy", "faithfulness"]].mean()
    overall.to_csv(output / "overall_by_method.csv", encoding="utf-8")

    by_domain = df.groupby("domain")[["answer_correctness", "answer_completeness", "answer_relevancy", "faithfulness"]].mean()
    by_domain.to_csv(output / "overall_by_domain.csv", encoding="utf-8")

    print(f"\nОтчеты сохранены в {output}/")

def main():
    print("Загрузка и подготовка данных...")
    samples, metadata = load_and_prepare_data()

    if not samples:
        print("Нет данных для оценки")
        return

    print("Запуск оценки через Ragas (это может занять несколько минут)...")
    df = evaluate_dataset(samples, metadata)

    print("\nПостроение графиков...")
    df_grouped = plot_results(df)

    save_reports(df, df_grouped)

    print("\n=== Средние метрики по методам ===")
    metric_labels = {
        "answer_correctness": "Точность",
        "answer_completeness": "Полнота",
        "answer_relevancy": "Релевантность",
        "faithfulness": "Отсутствие галлюцинаций"
    }
    overall = df.groupby("method")[["answer_correctness", "answer_completeness", "answer_relevancy", "faithfulness"]].mean()
    overall.columns = [metric_labels[c] for c in overall.columns]
    print(overall.round(3).to_markdown())

    print("\n=== Детализация по методам и доменам ===")
    print(df_grouped.to_markdown(index=False))

if __name__ == "__main__":
    main()