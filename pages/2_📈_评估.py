"""
RAG评估记录和对比页面
查看历史RAGAS评估结果，对比不同配置的效果
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


def load_eval_records():
    """加载所有评估记录"""
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    eval_file = log_dir / "rag_evals.jsonl"

    if not eval_file.exists():
        return pd.DataFrame()

    rows = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                # 展平成一行
                row = {
                    "eval_id": record["eval_id"],
                    "timestamp": record["timestamp"],
                }
                # 展开config
                for k, v in record["config"].items():
                    row[f"cfg_{k}"] = v
                # 展开metrics
                row.update(record["metrics"])
                row["notes"] = record.get("notes", "")
                rows.append(row)
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def main():
    st.set_page_config(
        page_title="RAG评估",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 RAG评估记录与对比")
    st.markdown("查看历史RAGAS评估结果，对比不同检索配置的效果")

    # 加载评估记录
    df_eval = load_eval_records()

    if df_eval.empty:
        st.info("📭 还没有评估记录")
        st.markdown("""
        **如何生成评估记录？**

        1. 运行评估脚本：
        ```bash
        cd src
        python evaluate.py --mode both
        ```

        2. 评估完成后会自动保存到日志

        3. 刷新此页面查看结果
        """)
        return

    # ========== 评估记录列表 ==========
    st.header("📊 历史评估记录")

    # 格式化显示表格
    display_df = df_eval.copy()
    display_df['时间'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 选择要显示的列
    display_cols = ['eval_id', '时间']

    # 添加配置列
    config_cols = [col for col in display_df.columns if col.startswith('cfg_')]
    for col in config_cols:
        display_cols.append(col)

    # 添加指标列
    metric_cols = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    for col in metric_cols:
        if col in display_df.columns:
            display_cols.append(col)

    display_cols.append('notes')

    # 对指标列保留3位小数
    for col in metric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    st.dataframe(
        display_df[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # ========== 评估对比 ==========
    st.header("🔬 评估结果对比")

    if len(df_eval) < 2:
        st.info("至少需要2条评估记录才能对比，请运行更多评估")
        return

    col1, col2 = st.columns(2)

    # 创建选项（显示时间+配置）
    options = []
    for _, row in df_eval.iterrows():
        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        retriever = row.get('cfg_retriever', '未知')
        reranker = "✓" if row.get('cfg_use_reranker', False) else "✗"
        label = f"{row['eval_id']} | {time_str} | {retriever} | Reranker:{reranker}"
        options.append((row['eval_id'], label))

    with col1:
        left_id = st.selectbox(
            "选择左侧评估",
            options=[opt[0] for opt in options],
            format_func=lambda x: next(opt[1] for opt in options if opt[0] == x),
            index=max(0, len(options) - 2)  # 倒数第二个
        )

    with col2:
        right_id = st.selectbox(
            "选择右侧评估",
            options=[opt[0] for opt in options],
            format_func=lambda x: next(opt[1] for opt in options if opt[0] == x),
            index=len(options) - 1  # 最后一个
        )

    # 获取两条评估记录
    left = df_eval[df_eval["eval_id"] == left_id].iloc[0]
    right = df_eval[df_eval["eval_id"] == right_id].iloc[0]

    # 显示配置对比
    st.subheader("⚙️ 配置对比")

    config_compare_data = []
    for col in config_cols:
        config_compare_data.append({
            "配置项": col.replace('cfg_', ''),
            "左侧": str(left.get(col, 'N/A')),
            "右侧": str(right.get(col, 'N/A')),
        })

    st.dataframe(pd.DataFrame(config_compare_data), use_container_width=True, hide_index=True)

    # 显示指标对比
    st.subheader("📊 指标对比")

    metric_compare_data = []
    for metric in metric_cols:
        if metric in df_eval.columns:
            l_val = float(left[metric])
            r_val = float(right[metric])
            delta = r_val - l_val

            # 判断升降（绿色为升，红色为降）
            if delta > 0:
                delta_str = f"+{delta:.3f} ↑"
                delta_color = "🟢"
            elif delta < 0:
                delta_str = f"{delta:.3f} ↓"
                delta_color = "🔴"
            else:
                delta_str = "0.000 ="
                delta_color = "⚪"

            metric_compare_data.append({
                "指标": metric.replace('_', ' ').title(),
                "左侧": f"{l_val:.3f}",
                "右侧": f"{r_val:.3f}",
                "差值": delta_str,
                "": delta_color
            })

    compare_df = pd.DataFrame(metric_compare_data)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # 显示备注
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**左侧备注:** {left.get('notes', '无')}")
    with col2:
        st.caption(f"**右侧备注:** {right.get('notes', '无')}")

    # ========== 指标趋势 ==========
    st.header("📈 指标趋势")

    if len(df_eval) >= 3:
        # 按时间排序
        trend_df = df_eval.sort_values('timestamp').copy()

        # 选择要展示的指标
        selected_metrics = st.multiselect(
            "选择要查看的指标",
            metric_cols,
            default=metric_cols[:3]
        )

        if selected_metrics:
            # 准备绘图数据
            trend_data = trend_df[['timestamp'] + selected_metrics].set_index('timestamp')

            st.line_chart(trend_data)
        else:
            st.info("请选择至少一个指标")
    else:
        st.info("至少需要3条评估记录才能显示趋势图")


if __name__ == "__main__":
    main()
