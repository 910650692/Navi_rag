"""
RAG请求日志记录模块
记录每次查询的详细信息，用于后续分析和优化
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class RAGLogEntry:
    """RAG请求日志条目"""
    timestamp: str  # ISO格式时间戳
    session_id: str  # 会话ID
    query: str  # 用户原始问题
    rewritten_query: Optional[str]  # 改写后的问题

    # 配置信息
    use_rewriter: bool  # 是否使用query改写
    use_reranker: bool  # 是否使用重排序
    use_hybrid: bool  # 是否使用混合检索
    top_k: int  # 返回文档数

    # 检索结果
    retrieval_docs: List[Dict[str, Any]]  # 检索到的文档列表

    # 回答
    answer: str  # LLM生成的答案（可截断）
    answer_length: int  # 完整答案长度

    # 性能指标
    latency_ms: float  # 总耗时（毫秒）
    retrieval_latency_ms: Optional[float]  # 检索耗时
    llm_latency_ms: Optional[float]  # LLM耗时

    # 错误信息
    error: Optional[str] = None  # 错误信息
    error_type: Optional[str] = None  # 错误类型


class RAGLogger:
    """RAG日志记录器"""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # 使用项目根目录下的data/logs
            base_dir = Path(__file__).parent.parent
            log_dir = base_dir / "data" / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 按日期分文件
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"rag_queries_{today}.jsonl"

    def log_query(
        self,
        session_id: str,
        query: str,
        rewritten_query: Optional[str],
        use_rewriter: bool,
        use_reranker: bool,
        use_hybrid: bool,
        top_k: int,
        retrieval_docs: List,
        answer: str,
        latency_ms: float,
        retrieval_latency_ms: Optional[float] = None,
        llm_latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        **extra_fields  # 新增：接受额外字段（如query_classification, adaptive_strategy）
    ):
        """
        记录一次RAG查询

        Args:
            session_id: 会话ID
            query: 原始问题
            rewritten_query: 改写后的问题
            use_rewriter: 是否使用query改写
            use_reranker: 是否使用重排序
            use_hybrid: 是否使用混合检索
            top_k: 返回文档数
            retrieval_docs: 检索到的文档列表（Document对象）
            answer: LLM答案
            latency_ms: 总耗时
            retrieval_latency_ms: 检索耗时
            llm_latency_ms: LLM耗时
            error: 错误信息
            error_type: 错误类型
        """
        # 构建检索文档信息
        docs_info = []
        for i, doc in enumerate(retrieval_docs[:top_k]):  # 只记录实际返回的文档
            doc_info = {
                "rank": i + 1,
                "source": doc.metadata.get("source", "未知"),
                "doc_type": doc.metadata.get("doc_type", "未知"),
                "file_type": doc.metadata.get("file_type", "未知"),
            }

            # 添加section或row_number（如果有）
            if "section" in doc.metadata:
                doc_info["section"] = doc.metadata["section"]
            elif "row_number" in doc.metadata:
                doc_info["row_number"] = doc.metadata["row_number"]
            elif "page" in doc.metadata:
                doc_info["page"] = doc.metadata["page"]

            # 添加内容预览（前100字符）
            doc_info["content_preview"] = doc.page_content[:100].replace('\n', ' ')

            docs_info.append(doc_info)

        # 构建日志条目
        log_entry = RAGLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query=query,
            rewritten_query=rewritten_query,
            use_rewriter=use_rewriter,
            use_reranker=use_reranker,
            use_hybrid=use_hybrid,
            top_k=top_k,
            retrieval_docs=docs_info,
            answer=answer[:500] if answer else "",  # 只保存前500字符
            answer_length=len(answer) if answer else 0,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            error=error,
            error_type=error_type,
        )

        # 写入日志文件（JSONL格式，每行一个JSON对象）
        try:
            # 将dataclass转为字典
            log_dict = asdict(log_entry)

            # 添加额外字段（如Adaptive RAG的分类和策略信息）
            log_dict.update(extra_fields)

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + '\n')
            print(f"✅ 日志已写入: {self.log_file}")
        except Exception as e:
            print(f"❌ 日志写入失败: {e}")

    def get_logs_path(self) -> str:
        """获取当前日志文件路径"""
        return str(self.log_file)


# 全局单例
_logger: Optional[RAGLogger] = None


def get_rag_logger() -> RAGLogger:
    """获取RAG日志记录器（单例）"""
    global _logger
    if _logger is None:
        _logger = RAGLogger()
    return _logger
