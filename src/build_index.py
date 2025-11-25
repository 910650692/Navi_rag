from pathlib import Path

from langchain_community.vectorstores import FAISS

from loaders import load_documents
from splitters import split_documents,hierarchical_split_documents
from embeddings import get_embeddings

INDEX_DIR = "../data/index"
INDEX_NAME = "nav_faiss"  # 最终会生成 data/index/nav_faiss 目录


def build_index():
    print("🔍 开始加载文档...")
    docs = load_documents("../data/documents")

    print("✂️  开始切分文档...")
    splits = split_documents(docs)

    print("🧠 准备 embedding 模型...")
    embeddings = get_embeddings()

    print("📦 正在构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    save_path = Path(INDEX_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(save_path / INDEX_NAME))
    print(f"✅ 向量库已保存到: {save_path / INDEX_NAME}")


if __name__ == "__main__":
    build_index()
