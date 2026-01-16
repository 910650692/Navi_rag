"""
对比 Dense 和 Hybrid 检索的 RAGAS 评估结果
使用方法：
    python compare_results.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DENSE_CSV = BASE_DIR / "ragas_results_dense.csv"
HYBRID_CSV = BASE_DIR / "ragas_results_hybrid.csv"

METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def load_results():
    """加载两个评估结果文件"""
    if not DENSE_CSV.exists():
        raise FileNotFoundError(f"找不到 Dense 结果: {DENSE_CSV}")
    if not HYBRID_CSV.exists():
        raise FileNotFoundError(f"找不到 Hybrid 结果: {HYBRID_CSV}")

    df_dense = pd.read_csv(DENSE_CSV, encoding='utf-8-sig')
    df_hybrid = pd.read_csv(HYBRID_CSV, encoding='utf-8-sig')

    return df_dense, df_hybrid


def print_overall_comparison(df_dense, df_hybrid):
    """打印整体指标对比"""
    print("\n" + "=" * 80)
    print("📊 整体指标对比 (平均值)")
    print("=" * 80)
    print(f"{'指标':<25} {'Dense':<15} {'Hybrid':<15} {'提升':<15}")
    print("-" * 80)

    improvements = []

    for metric in METRICS:
        if metric not in df_dense.columns or metric not in df_hybrid.columns:
            continue

        dense_mean = df_dense[metric].mean()
        hybrid_mean = df_hybrid[metric].mean()
        diff = hybrid_mean - dense_mean
        diff_pct = (diff / dense_mean * 100) if dense_mean != 0 else 0

        improvements.append({
            'metric': metric,
            'dense': dense_mean,
            'hybrid': hybrid_mean,
            'diff': diff,
            'diff_pct': diff_pct
        })

        # 格式化输出
        sign = "+" if diff >= 0 else ""
        color_sign = "🟢" if diff >= 0 else "🔴"

        print(f"{metric:<25} {dense_mean:<15.4f} {hybrid_mean:<15.4f} "
              f"{color_sign} {sign}{diff:.4f} ({sign}{diff_pct:.2f}%)")

    print("=" * 80)

    # 计算总体改进
    avg_improvement = sum(item['diff_pct'] for item in improvements) / len(improvements)
    print(f"\n📈 平均相对提升: {avg_improvement:+.2f}%")

    return improvements


def print_per_question_comparison(df_dense, df_hybrid):
    """打印每个问题的详细对比"""
    print("\n" + "=" * 80)
    print("🔍 逐问题对比")
    print("=" * 80)

    # 假设两个 DataFrame 的问题顺序一致
    if 'question' in df_dense.columns and 'question' in df_hybrid.columns:
        for idx in range(len(df_dense)):
            question = df_dense.iloc[idx]['question']
            qid = df_dense.iloc[idx].get('id', f'Q{idx+1}')

            print(f"\n[{qid}] {question[:60]}...")
            print("-" * 80)

            for metric in METRICS:
                if metric not in df_dense.columns or metric not in df_hybrid.columns:
                    continue

                dense_val = df_dense.iloc[idx][metric]
                hybrid_val = df_hybrid.iloc[idx][metric]
                diff = hybrid_val - dense_val

                # 跳过 NaN 值
                if pd.isna(dense_val) or pd.isna(hybrid_val):
                    continue

                sign = "+" if diff >= 0 else ""
                color_sign = "🟢" if diff >= 0 else "🔴"

                print(f"  {metric:<22}: Dense={dense_val:.4f}, Hybrid={hybrid_val:.4f}, "
                      f"{color_sign} {sign}{diff:.4f}")


def export_comparison_summary(improvements):
    """导出对比汇总到 CSV"""
    summary_df = pd.DataFrame(improvements)
    output_path = BASE_DIR / "comparison_summary.csv"
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 对比汇总已保存至: {output_path}")


def main():
    print("[*] 加载评估结果...")
    df_dense, df_hybrid = load_results()

    print(f"[+] Dense 样本数: {len(df_dense)}")
    print(f"[+] Hybrid 样本数: {len(df_hybrid)}")

    # 整体对比
    improvements = print_overall_comparison(df_dense, df_hybrid)

    # 导出汇总
    export_comparison_summary(improvements)

    # 询问是否查看逐问题对比
    print("\n" + "=" * 80)
    response = input("是否查看逐问题详细对比? (y/n): ").strip().lower()
    if response == 'y':
        print_per_question_comparison(df_dense, df_hybrid)

    print("\n[*] 对比完成!")


if __name__ == '__main__':
    main()
