from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS

from loaders import load_single_document
from splitters import hierarchical_split_documents, split_documents
from embeddings import get_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOC = BASE_DIR / "data" / "documents" / "PIS-2116_Location Based Service_A-V0.0.2.3.docx"
DEFAULT_INDEX_DIR = BASE_DIR / "data" / "index"
DEFAULT_INDEX_NAME = "pis2116_single"


def build_single_index(
    doc_path: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    hierarchical: bool = True,
) -> Path:
    if not doc_path.exists():
        raise FileNotFoundError(f"{doc_path} does not exist.")

    print(f"📄 文档路径: {doc_path}")
    docs = load_single_document(str(doc_path))
    print(f"✅ 原始 chunk 数量: {len(docs)}")

    splitter = hierarchical_split_documents if hierarchical else split_documents
    print("✂️  开始切分文档...")
    splits = splitter(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"✂️  切分后 chunk 数量: {len(splits)}")

    print("🧠 准备 embedding 模型...")
    embeddings = get_embeddings()

    print("📦 构建 FAISS 索引...")
    vectorstore = FAISS.from_documents(splits, embeddings)

    target_dir = Path(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / index_name
    vectorstore.save_local(str(save_path))
    print(f"✅ 向量库已保存到: {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单文档构建 FAISS 索引，便于快速测试 RAG 效果。",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_DOC,
        help=f"文档路径 (默认: {DEFAULT_DOC})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"索引保存目录 (默认: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help=f"索引名称 (默认: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="递归切分 chunk_size 参数。",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="递归切分 chunk_overlap 参数。",
    )
    parser.add_argument(
        "--basic-splitter",
        action="store_true",
        help="默认使用层级切分，若设置该参数则退回基础递归切分。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_single_index(
        doc_path=args.doc,
        index_dir=args.index_dir,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        hierarchical=not args.basic_splitter,
    )


if __name__ == "__main__":
    main()
