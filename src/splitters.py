from typing import List, Sequence, Tuple, Dict, Any

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


# ==================== 自适应切分配置 ====================
SPLIT_CONFIG: Dict[str, Dict[str, Any]] = {
    'nav_spec': {
        'hierarchical': {'chunk_size': 1200, 'chunk_overlap': 200},
        'basic': {'chunk_size': 1000, 'chunk_overlap': 150},
    },
    'api_spec': {
        'hierarchical': {'chunk_size': 1000, 'chunk_overlap': 150},
        'basic': {'chunk_size': 800, 'chunk_overlap': 120},
    },
    'process_spec': {
        'hierarchical': {'chunk_size': 600, 'chunk_overlap': 100},
        'basic': {'chunk_size': 600, 'chunk_overlap': 100},
    },
    'metrics_spec': {
        'hierarchical': {'chunk_size': 800, 'chunk_overlap': 120},
        'basic': {'chunk_size': 800, 'chunk_overlap': 100},
    },
    'doc_generic': {
        'hierarchical': {'chunk_size': 800, 'chunk_overlap': 100},
        'basic': {'chunk_size': 800, 'chunk_overlap': 100},
    },
}


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


def split_by_doc_type(
    documents: List[Document],
    doc_type: str = None,
    file_type: str = None,
) -> List[Document]:
    """
    根据文档类型和是否有层级结构自适应切分

    核心逻辑：
    1. 检查文档是否有 section 元数据（无论是 DOCX 还是 PDF）
       - 有 section → 使用 hierarchical_split（保留语义完整性）
       - 无 section → 使用基础 split_documents（纯文本递归切分）
    2. 根据 doc_type 选择对应的 chunk_size 和 overlap 参数

    Args:
        documents: 待切分的文档列表
        doc_type: 文档类型 (nav_spec/api_spec/process_spec等)，如果为None则从metadata中读取
        file_type: 文件类型 (.docx/.pdf)，如果为None则从metadata中读取

    Returns:
        切分后的文档列表
    """
    if not documents:
        return []

    # 从第一个文档的metadata推断类型（如果未指定）
    first_doc = documents[0]
    if doc_type is None:
        doc_type = first_doc.metadata.get('doc_type', 'doc_generic')
    if file_type is None:
        file_type = first_doc.metadata.get('file_type', '.pdf')

    # 判断是否有层级结构（检查是否有 section 元数据）
    has_section = any('section' in d.metadata for d in documents)

    if has_section:
        has_hierarchy = True
    else:
        has_hierarchy = False
        if file_type in ['.docx', '.pdf']:
            print(f"  ⚠️  {file_type}文档但未提取到section信息，退回基础切分")

    # 获取切分参数
    config = SPLIT_CONFIG.get(doc_type, SPLIT_CONFIG['doc_generic'])
    strategy = 'hierarchical' if has_hierarchy else 'basic'
    params = config[strategy]

    # 选择切分器
    if has_hierarchy:
        print(f"  📐 使用层级切分 | doc_type={doc_type} | chunk_size={params['chunk_size']}")
        return hierarchical_split_documents(
            documents,
            chunk_size=params['chunk_size'],
            chunk_overlap=params['chunk_overlap'],
        )
    else:
        print(f"  📐 使用基础切分 | doc_type={doc_type} | chunk_size={params['chunk_size']}")
        return split_documents(
            documents,
            chunk_size=params['chunk_size'],
            chunk_overlap=params['chunk_overlap'],
        )
