"""RAGAS 指标对比脚本：读取 dense/hybrid 两份 CSV，输出表格与图表。"""
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS: List[str] = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较 dense / hybrid 两套 RAGAS 结果，并生成图表"
    )
    parser.add_argument("dense_csv", type=Path, help="稠密检索评估 CSV")
    parser.add_argument("hybrid_csv", type=Path, help="混合检索评估 CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="图表与摘要导出目录（默认 reports/）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RAGAS Metrics Comparison",
        help="图表标题",
    )
    return parser.parse_args()


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 {label} CSV: {path}")
    df = pd.read_csv(path)
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少指标列: {missing}")
    return df


def summarize_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {metric: float(df[metric].mean()) for metric in METRICS}


def build_summary_frame(dense_summary: Dict[str, float], hybrid_summary: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        dense_val = dense_summary.get(metric, float("nan"))
        hybrid_val = hybrid_summary.get(metric, float("nan"))
        rows.append(
            {
                "metric": metric,
                "dense": dense_val,
                "hybrid": hybrid_val,
                "delta": hybrid_val - dense_val,
            }
        )
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["dense"], width, label="Dense")
    ax.bar([i + width / 2 for i in x], df["hybrid"], width, label="Hybrid")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["metric"], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    for i, row in df.iterrows():
        ax.text(i - width / 2, row["dense"] + 0.01, f"{row['dense']:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["hybrid"] + 0.01, f"{row['hybrid']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_df = load_dataset(args.dense_csv, "dense")
    hybrid_df = load_dataset(args.hybrid_csv, "hybrid")

    dense_summary = summarize_metrics(dense_df)
    hybrid_summary = summarize_metrics(hybrid_df)
    summary_df = build_summary_frame(dense_summary, hybrid_summary)

    summary_csv = output_dir / "ragas_metric_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    chart_path = output_dir / "ragas_metric_compare.png"
    plot_summary(summary_df, args.title, chart_path)

    print("对比完成 ✅")
    print(summary_df.to_string(index=False, formatters={
        "dense": lambda v: f"{v:.3f}",
        "hybrid": lambda v: f"{v:.3f}",
        "delta": lambda v: f"{v:+.3f}",
    }))
    print(f"摘要 CSV: {summary_csv}")
    print(f"图表: {chart_path}")


if __name__ == "__main__":
    main()
