import random
from pathlib import Path

from src.loaders import load_single_document
from src.splitters import hierarchical_split_documents
from src.chunk_stats import summarize_chunk_lengths, format_length_stats


DOC_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "data"
    / "documents"
    / "PIS-2116_Location Based Service_A-V0.0.2.3.docx"
)


def test_split():
    docs = load_single_document(str(DOC_PATH))
    chunks = hierarchical_split_documents(docs)
    total = len(chunks)
    print(f"分块数量：{total}")

    lengths, distribution = summarize_chunk_lengths(chunks)
    print("长度统计：", format_length_stats(lengths))
    print("区间分布：")
    for bucket, count in distribution.items():
        ratio = (count / total * 100) if total else 0
        print(f"  {bucket}: {count} ({ratio:.1f}%)")

    sample_size = min(5, total)
    sample_chunks = random.sample(chunks, sample_size) if total else []

    for idx, chunk in enumerate(sample_chunks, 1):
        snippet = chunk.page_content[:200].replace("\n", " ")
        metadata = chunk.metadata or {}
        print("-" * 50)
        print(f"样本 {idx}")
        print(f"长度：{len(chunk.page_content)}")
        print(f"来源：{metadata.get('source')}")
        print(f"层级：{metadata.get('section', '未知')}")
        print(f"页码：{metadata.get('page', '无页码')}")
        print(f"内容片段：{snippet}{'...' if len(chunk.page_content) > 200 else ''}")

    print("测试完成！")


if __name__ == "__main__":
    test_split()
