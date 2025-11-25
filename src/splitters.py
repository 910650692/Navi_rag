from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DEFAULT_HIERARCHY_HEADERS: Sequence[Tuple[str, str]] = (
    ("#", "section"),
    ("##", "subsection"),
    ("###", "subsubsection"),
)


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """基础版递归切分，适合结构一般的文档。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", "。", "，", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    print(f"切分后 chunk 数量：{len(splits)}")
    return splits


def hierarchical_split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    headers: Sequence[Tuple[str, str]] = DEFAULT_HIERARCHY_HEADERS,
) -> List[Document]:
    """
    先按标题层级拆分，再在每个小节内递归分块，适合 Markdown / 标题明确的文档。
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=list(headers),
        strip_headers=False,
    )
    inner_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    hierarchical_chunks: List[Document] = []
    for doc in documents:
        header_docs = header_splitter.split_text(doc.page_content)
        if not header_docs:
            header_docs = [Document(page_content=doc.page_content, metadata={})]

        enriched_docs = []
        for h_doc in header_docs:
            text = h_doc.page_content.strip()
            if not text:
                continue
            metadata = {**doc.metadata, **h_doc.metadata}
            enriched_docs.append(Document(page_content=text, metadata=metadata))

        if not enriched_docs:
            continue

        hierarchical_chunks.extend(inner_splitter.split_documents(enriched_docs))

    print(f"层级切分 chunk 数量：{len(hierarchical_chunks)}")
    return hierarchical_chunks
