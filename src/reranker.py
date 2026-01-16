# src/reranker.py
from typing import List, Optional

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker


class CrossEncoderReranker:
    """
    使用 BAAI/bge-reranker-base 对检索到的文档进行重排序。

    使用方式：
        reranker = CrossEncoderReranker()
        reranked_docs = reranker.rerank(query, docs, top_k=4)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
    ) -> None:
        # 这里模型会在第一次实例化时下载，建议整个进程里只建一个实例
        self._reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        对 docs 按 query 相关性重排，并返回前 top_k 条（不传 top_k 就全返回）。
        """
        if not docs:
            return docs

        # FlagReranker 接受 [query, doc] 的 pair 列表
        pairs = [[query, doc.page_content] for doc in docs]

        # 返回的是一个与 pairs 对应的 score list
        scores = self._reranker.compute_score(pairs, normalize=True)

        # 带分数一起排序
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        reranked_docs = [d for d, _ in scored_docs]
        return reranked_docs
