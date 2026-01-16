"""
RAG系统监控页面
实时查看查询日志、性能指标和统计分析
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd


@st.cache_data(ttl=10)  # 缓存10秒，避免频繁读取
def load_logs(days: int = 7):
    """
    加载最近N天的日志文件

    Args:
        days: 加载最近几天的日志

    Returns:
        DataFrame
    """
    log_dir = Path(__file__).parent.parent / "data" / "logs"

    if not log_dir.exists():
        return pd.DataFrame()

    # 获取最近N天的日期
    today = datetime.now()
    date_range = [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]

    # 读取所有匹配的日志文件
    rows = []
    for date_str in date_range:
        log_file = log_dir / f"rag_queries_{date_str}.jsonl"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def main():
    st.set_page_config(
        page_title="RAG监控",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 RAG系统监控")
    st.markdown("实时查看查询日志、性能指标和统计分析")

    # 侧边栏：时间范围选择
    with st.sidebar:
        st.header("⚙️ 设置")
        days = st.slider("加载最近几天的日志", 1, 30, 7)
        auto_refresh = st.checkbox("自动刷新（10秒）", value=False)

        if auto_refresh:
            st.info("⏱️ 页面将每10秒自动刷新")

    # 加载日志
    df = load_logs(days=days)

    if df.empty:
        st.info("📭 当前还没有日志数据，去主页问几个问题吧！")
        return

    # ========== 概览指标 ==========
    st.header("📈 概览")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📝 总请求数", len(df))

    with col2:
        avg_latency = df['latency_ms'].mean()
        st.metric("⏱️ 平均耗时", f"{avg_latency:.0f} ms")

    with col3:
        error_count = df['error'].notna().sum()
        error_rate = (error_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("❌ 错误率", f"{error_rate:.1f}%", delta=f"{error_count} 个")

    with col4:
        unique_sessions = df['session_id'].nunique()
        st.metric("👤 独立会话", unique_sessions)

    # ========== 配置使用统计 ==========
    st.header("⚙️ 配置使用统计")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("检索方式")
        search_stats = df['use_hybrid'].value_counts()
        search_df = pd.DataFrame({
            '方式': ['Hybrid' if x else 'Dense' for x in search_stats.index],
            '次数': search_stats.values
        })
        st.bar_chart(search_df.set_index('方式'))

    with col2:
        st.subheader("Query改写")
        rewriter_stats = df['use_rewriter'].value_counts()
        rewriter_df = pd.DataFrame({
            '配置': ['启用' if x else '禁用' for x in rewriter_stats.index],
            '次数': rewriter_stats.values
        })
        st.bar_chart(rewriter_df.set_index('配置'))

    with col3:
        st.subheader("重排序")
        reranker_stats = df['use_reranker'].value_counts()
        reranker_df = pd.DataFrame({
            '配置': ['启用' if x else '禁用' for x in reranker_stats.index],
            '次数': reranker_stats.values
        })
        st.bar_chart(reranker_df.set_index('配置'))

    # ========== 性能分析 ==========
    st.header("⏱️ 性能分析")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("耗时分布")
        st.line_chart(df.set_index('timestamp')['latency_ms'])

    with col2:
        st.subheader("耗时统计")
        st.write(df[['latency_ms', 'retrieval_latency_ms', 'llm_latency_ms']].describe())

    # ========== 检索来源分析 ==========
    st.header("📚 检索来源分析")

    # 展开retrieval_docs提取所有source
    all_sources = []
    for docs_list in df['retrieval_docs']:
        if isinstance(docs_list, list):
            for doc in docs_list:
                if isinstance(doc, dict):
                    all_sources.append(doc.get('source', '未知'))

    if all_sources:
        source_counts = pd.Series(all_sources).value_counts()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Top 10 文档来源")
            st.bar_chart(source_counts.head(10))

        with col2:
            st.subheader("来源统计")
            st.dataframe(
                pd.DataFrame({
                    '文档': source_counts.index[:10],
                    '被检索次数': source_counts.values[:10]
                }),
                hide_index=True
            )

    # ========== 查询记录 ==========
    st.header("📋 最近查询记录")

    # 显示最近20条
    recent_df = df.sort_values('timestamp', ascending=False).head(20)

    # 格式化显示
    display_df = pd.DataFrame({
        '时间': recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'Session': recent_df['session_id'],
        '问题': recent_df['query'].str[:50] + '...',
        '检索': recent_df['use_hybrid'].map({True: 'Hybrid', False: 'Dense'}),
        '改写': recent_df['use_rewriter'].map({True: '✓', False: '✗'}),
        '重排': recent_df['use_reranker'].map({True: '✓', False: '✗'}),
        'Top-K': recent_df['top_k'],
        '耗时(ms)': recent_df['latency_ms'].round(0),
        '答案长度': recent_df['answer_length'],
        '状态': recent_df['error'].apply(lambda x: '❌' if pd.notna(x) else '✅')
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ========== 错误分析 ==========
    if error_count > 0:
        st.header("❌ 错误分析")

        error_df = df[df['error'].notna()][['timestamp', 'session_id', 'query', 'error_type', 'error']]
        error_df['时间'] = error_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        st.dataframe(
            error_df[['时间', 'session_id', 'query', 'error_type', 'error']],
            use_container_width=True,
            hide_index=True
        )

    # 自动刷新
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
