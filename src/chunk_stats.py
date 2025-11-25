from __future__ import annotations

from collections import OrderedDict
from statistics import mean, median
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document


def summarize_chunk_lengths(
    chunks: Iterable[Document],
    bucket_boundaries: Sequence[int] | None = None,
) -> Tuple[List[int], OrderedDict[str, int]]:
    """
    统计 chunk 长度并按区间划分分布。

    Args:
        chunks: 文档分块列表。
        bucket_boundaries: 递增的分界值列表，例如 [0, 150, 300, 600, 1000]。
            默认为 [0, 100, 250, 500, 750, 1000, 1500]，最后额外追加 inf。

    Returns:
        (lengths, distribution)
        lengths: 所有 chunk 的字符长度列表。
        distribution: OrderedDict，key 为区间字符串，value 为数量。
    """

    lengths: List[int] = [len(doc.page_content or "") for doc in chunks]

    if bucket_boundaries is None:
        bucket_boundaries = [0, 100, 250, 500, 750, 1000, 1500]

    if not bucket_boundaries:
        raise ValueError("bucket_boundaries 不能为空。")

    sorted_bounds = sorted(bucket_boundaries)
    buckets = OrderedDict()
    for left, right in zip(sorted_bounds, sorted_bounds[1:]):
        buckets[f"[{left}, {right})"] = 0
    buckets[f"[{sorted_bounds[-1]}, +inf)"] = 0

    for length in lengths:
        placed = False
        for left, right in zip(sorted_bounds, sorted_bounds[1:]):
            if left <= length < right:
                buckets[f"[{left}, {right})"] += 1
                placed = True
                break
        if not placed:
            buckets[f"[{sorted_bounds[-1]}, +inf)"] += 1

    return lengths, buckets


def format_length_stats(lengths: Sequence[int]) -> str:
    """
    简单格式化均值/中位数/P95 统计信息，辅助打印。
    """
    if not lengths:
        return "无 chunk"

    sorted_lengths = sorted(lengths)
    p95_index = max(0, int(len(sorted_lengths) * 0.95) - 1)
    p95 = sorted_lengths[p95_index]
    stats_lines = [
        f"总数：{len(sorted_lengths)}",
        f"最小值：{sorted_lengths[0]}",
        f"最大值：{sorted_lengths[-1]}",
        f"均值：{mean(sorted_lengths):.1f}",
        f"中位数：{median(sorted_lengths)}",
        f"P95：{p95}",
    ]
    return " | ".join(stats_lines)
