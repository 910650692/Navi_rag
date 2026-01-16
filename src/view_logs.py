"""
RAG日志查看工具
用于查看和分析RAG查询日志
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter


def view_logs(log_file: str = None, limit: int = 10):
    """
    查看最近的RAG查询日志

    Args:
        log_file: 日志文件路径（如果不指定，查看最新的日志文件）
        limit: 显示最近的N条记录
    """
    log_dir = Path(__file__).parent.parent / "data" / "logs"

    if log_file:
        log_path = Path(log_file)
    else:
        # 找最新的日志文件
        log_files = list(log_dir.glob("rag_queries_*.jsonl"))
        if not log_files:
            print("❌ 没有找到日志文件")
            return

        log_path = max(log_files, key=lambda p: p.stat().st_mtime)

    print("=" * 80)
    print(f"📋 日志文件: {log_path.name}")
    print("=" * 80)

    # 读取日志
    logs = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not logs:
        print("⚠️  日志文件为空")
        return

    print(f"\n总查询次数: {len(logs)}")
    print(f"显示最近 {min(limit, len(logs))} 条记录:\n")

    # 显示最近N条
    for i, log in enumerate(logs[-limit:], 1):
        print(f"\n{'='*80}")
        print(f"查询 #{len(logs) - limit + i}")
        print(f"{'='*80}")

        timestamp = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        print(f"时间: {timestamp}")
        print(f"Session: {log['session_id']}")
        print(f"\n原始问题: {log['query']}")

        if log.get('rewritten_query'):
            print(f"改写问题: {log['rewritten_query']}")

        print(f"\n配置:")
        print(f"  - Query改写: {'✓' if log['use_rewriter'] else '✗'}")
        print(f"  - 重排序: {'✓' if log['use_reranker'] else '✗'}")
        print(f"  - Top-K: {log['top_k']}")

        print(f"\n检索结果: {len(log['retrieval_docs'])} 个文档")
        for doc in log['retrieval_docs'][:3]:  # 只显示前3个
            source = doc.get('source', '未知')
            rank = doc.get('rank', '?')
            print(f"  [{rank}] {source}", end="")

            if 'section' in doc:
                print(f" - {doc['section']}", end="")
            elif 'row_number' in doc:
                print(f" - 行{doc['row_number']}", end="")
            elif 'page' in doc:
                print(f" - 页{doc['page']}", end="")

            print()

        if len(log['retrieval_docs']) > 3:
            print(f"  ... 还有 {len(log['retrieval_docs']) - 3} 个文档")

        print(f"\n答案长度: {log['answer_length']} 字符")
        print(f"答案预览: {log['answer'][:100]}...")

        print(f"\n性能:")
        print(f"  - 总耗时: {log['latency_ms']:.0f} ms")
        if log.get('retrieval_latency_ms'):
            print(f"  - 检索耗时: {log['retrieval_latency_ms']:.0f} ms")
        if log.get('llm_latency_ms'):
            print(f"  - LLM耗时: {log['llm_latency_ms']:.0f} ms")

        if log.get('error'):
            print(f"\n❌ 错误: [{log.get('error_type')}] {log['error']}")


def analyze_logs(log_file: str = None):
    """
    分析RAG查询日志，生成统计信息

    Args:
        log_file: 日志文件路径（如果不指定，查看最新的日志文件）
    """
    log_dir = Path(__file__).parent.parent / "data" / "logs"

    if log_file:
        log_path = Path(log_file)
    else:
        # 找最新的日志文件
        log_files = list(log_dir.glob("rag_queries_*.jsonl"))
        if not log_files:
            print("❌ 没有找到日志文件")
            return

        log_path = max(log_files, key=lambda p: p.stat().st_mtime)

    print("=" * 80)
    print(f"📊 日志分析: {log_path.name}")
    print("=" * 80)

    # 读取日志
    logs = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not logs:
        print("⚠️  日志文件为空")
        return

    print(f"\n📈 基本统计:")
    print(f"  总查询次数: {len(logs)}")

    # 配置统计
    rewriter_count = sum(1 for log in logs if log['use_rewriter'])
    reranker_count = sum(1 for log in logs if log['use_reranker'])
    print(f"\n⚙️  配置使用:")
    print(f"  Query改写: {rewriter_count}/{len(logs)} ({rewriter_count/len(logs)*100:.1f}%)")
    print(f"  重排序: {reranker_count}/{len(logs)} ({reranker_count/len(logs)*100:.1f}%)")

    # 性能统计
    latencies = [log['latency_ms'] for log in logs if log['latency_ms']]
    if latencies:
        print(f"\n⏱️  性能统计:")
        print(f"  平均耗时: {sum(latencies)/len(latencies):.0f} ms")
        print(f"  最快: {min(latencies):.0f} ms")
        print(f"  最慢: {max(latencies):.0f} ms")

    # 文档来源统计
    all_sources = []
    for log in logs:
        for doc in log['retrieval_docs']:
            all_sources.append(doc.get('source', '未知'))

    source_counter = Counter(all_sources)
    print(f"\n📚 检索来源统计 (Top 5):")
    for source, count in source_counter.most_common(5):
        print(f"  {source}: {count} 次")

    # 错误统计
    errors = [log for log in logs if log.get('error')]
    if errors:
        print(f"\n❌ 错误统计:")
        print(f"  错误次数: {len(errors)}/{len(logs)} ({len(errors)/len(logs)*100:.1f}%)")

        error_types = Counter(log.get('error_type') for log in errors)
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count} 次")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_logs()
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        view_logs(limit=limit)
