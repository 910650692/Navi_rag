from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document


@dataclass
class _SectionNode:
    title: str
    level: int
    contents: List[str] = field(default_factory=list)
    children: List["_SectionNode"] = field(default_factory=list)


def _parse_heading_level(style_name: Optional[str]) -> Optional[int]:
    if not style_name:
        return None
    name = style_name.lower()
    match = re.search(r'(\d+)', name)
    if match and ('heading' in name or '标题' in style_name):
        return int(match.group(1))
    return None


def _build_loader(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        return PyPDFLoader(str(path))
    if suffix == '.docx':
        return Docx2txtLoader(str(path))
    raise ValueError(f'Unsupported file type: {path.suffix}')


def _load_docx_with_headings(path: Path) -> List[Document]:
    try:
        from docx import Document as WordDocument
    except ImportError:
        return _load_fallback_docx(path)

    word_doc = WordDocument(str(path))
    root = _SectionNode(title='root', level=0)
    stack = [root]

    for para in word_doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        level = _parse_heading_level(getattr(para.style, 'name', None))
        if level is None:
            stack[-1].contents.append(text)
            continue
        while stack and stack[-1].level >= level:
            stack.pop()
        node = _SectionNode(title=text, level=level)
        stack[-1].children.append(node)
        stack.append(node)

    docs: List[Document] = []

    def _flatten(node: _SectionNode, breadcrumb: List[str]):
        if node.contents:
            metadata = {'source': path.name}
            if breadcrumb:
                metadata['section'] = ' > '.join(breadcrumb)
            docs.append(
                Document(
                    page_content='\n'.join(node.contents),
                    metadata=metadata,
                )
            )
        for child in node.children:
            _flatten(child, breadcrumb + [child.title])

    if root.contents:
        docs.append(
            Document(
                page_content='\n'.join(root.contents),
                metadata={'source': path.name, 'section': 'preface'},
            )
        )

    for child in root.children:
        _flatten(child, [child.title])

    if docs:
        return docs
    return _load_fallback_docx(path)


def _load_fallback_docx(path: Path) -> List[Document]:
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault('source', path.name)
    return docs


def _load_with_metadata(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == '.docx':
        return _load_docx_with_headings(path)

    loader = _build_loader(path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault('source', path.name)
    return docs


def load_documents(docs_dir: str = 'data/documents') -> List[Document]:
    base_path = Path(docs_dir)
    if not base_path.exists():
        raise FileNotFoundError(f'{docs_dir} does not exist.')

    all_docs: List[Document] = []
    for path in base_path.rglob('*'):
        if path.is_dir():
            continue
        try:
            docs = _load_with_metadata(path)
        except ValueError:
            continue
        all_docs.extend(docs)

    print(f'Loaded document chunks: {len(all_docs)}')
    return all_docs


def load_single_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'{file_path} does not exist.')

    docs = _load_with_metadata(path)
    print(f'{path.name} chunks: {len(docs)}')
    return docs
