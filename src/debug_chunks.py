# debug_chunks.py
from src.loaders import load_documents
from src.splitters import split_documents

def main():
    docs = load_documents("../data/documents")
    print(f"原始 document 块数: {len(docs)}")

    splits = split_documents(docs)
    print(f"切分后 chunk 数: {len(splits)}")

    # 看几条样本
    for i, d in enumerate(splits[:5], 1):
        print("\n" + "=" * 40)
        print(f"[Chunk {i}]")
        print("source:", d.metadata.get("source"), "page:", d.metadata.get("page"))
        print(d.page_content[:400].replace("\n", " "))
        if len(d.page_content) > 400:
            print("...")

if __name__ == "__main__":
    main()
