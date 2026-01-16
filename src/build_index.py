from pathlib import Path
from itertools import groupby

from langchain_community.vectorstores import FAISS

from loaders import load_documents
from splitters import split_by_doc_type
from embeddings import get_embeddings

INDEX_DIR = "../data/index"
INDEX_NAME = "nav_faiss"  # 最终会生成 data/index/nav_faiss 目录


def build_index():
    print("🔍 开始加载文档...")
    docs = load_documents("../data/documents")
    print(f"✅ 加载完成，共 {len(docs)} 个原始chunk")

    print("\n✂️  开始自适应切分文档...")
    print("=" * 60)

    # 按文档分组（按source分组，保证同一文档的chunks一起处理）
    docs_sorted = sorted(docs, key=lambda d: d.metadata.get('source', ''))
    all_splits = []

    for source, group in groupby(docs_sorted, key=lambda d: d.metadata['source']):
        group_docs = list(group)
        doc_type = group_docs[0].metadata.get('doc_type', 'doc_generic')
        file_type = group_docs[0].metadata.get('file_type', '.pdf')

        print(f"\n📄 {source}")
        print(f"  类型: {doc_type} | 格式: {file_type} | 原始chunks: {len(group_docs)}")

        # Excel文件已经在加载时按行切分，跳过二次切分
        if file_type in ['.xlsx', '.xls']:
            splits = group_docs
            print(f"  📊 Excel文件，跳过切分（已按行切分）")
        else:
            # 其他文件类型进行自适应切分
            splits = split_by_doc_type(group_docs)
            print(f"  ✅ 切分后: {len(splits)} chunks")

        all_splits.extend(splits)

    print("\n" + "=" * 60)
    print(f"✂️  总切分结果: {len(all_splits)} chunks")

    # 统计不同doc_type和切分策略的分布
    print("\n📊 切分统计:")
    from collections import Counter
    doc_types = Counter(d.metadata.get('doc_type', 'unknown') for d in all_splits)
    for dtype, count in doc_types.most_common():
        print(f"  - {dtype}: {count} chunks")

    print("\n🧠 准备 embedding 模型...")
    embeddings = get_embeddings()

    print("📦 正在构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    save_path = Path(INDEX_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(save_path / INDEX_NAME))
    print(f"✅ 向量库已保存到: {save_path / INDEX_NAME}")


if __name__ == "__main__":
    build_index()
