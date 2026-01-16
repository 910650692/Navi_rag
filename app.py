# app.py
import os
import time
import uuid
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 注意这里：从 src.embeddings 导入你之前写好的 get_embeddings()
from src.embeddings import get_embeddings
from src.reranker import CrossEncoderReranker
from src.rag_logger import get_rag_logger
from src.query_classifier import classify_and_get_strategy

# 读取 .env（deepseek 的 key / base_url）
load_dotenv(override=True)

# 尝试导入 BM25 (用于混合检索)
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank_bm25 或 jieba 未安装，混合检索不可用。安装: pip install rank-bm25 jieba")


# ========== 缓存一些重资源：向量库、模型 ==========

@st.cache_resource
def load_vectorstore():
    """
    加载已经构建好的 FAISS 向量库，并提取所有文档（用于BM25）
    返回: (vectorstore, all_docs)
    """
    base_dir = Path(__file__).parent
    index_path = base_dir / "data" / "index" / "nav_faiss"

    if not index_path.exists():
        raise FileNotFoundError(f"找不到向量库目录: {index_path}")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # 本地环境 OK
    )

    # 提取所有文档（用于BM25混合检索）
    all_docs = []
    if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
        all_docs = list(vectorstore.docstore._dict.values())

    # 获取索引统计信息
    total_chunks = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else len(all_docs)
    print(f"📚 加载了 FAISS 向量库，共 {total_chunks} 个chunks")

    return vectorstore, all_docs


@st.cache_resource
def get_llms():
    """返回用于改写和回答的两个 LLM 实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    # 用于 query rewrite（非流式、温度 0，更稳定）
    rewrite_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        streaming=False,
    )

    # 用于最终回答（流式）
    answer_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
        streaming=True,
    )

    return rewrite_llm, answer_llm


@st.cache_resource
def get_reranker():
    """加载 CrossEncoder 重排模型（只加载一次）"""
    try:
        return CrossEncoderReranker()
    except Exception as e:
        st.warning(f"⚠️ Reranker 加载失败: {e}，将不使用重排")
        return None


# ========== Parent 内容扩展 ==========
def expand_with_parent_content(docs: List[Document], all_docs: List[Document]) -> List[Document]:
    """
    对检索结果进行父节点内容扩展

    Args:
        docs: 检索到的文档列表
        all_docs: 全部文档（用于查找父节点）

    Returns:
        扩展后的文档列表（去重后可能变少）
    """
    if not docs or not all_docs:
        return docs

    expanded = []
    seen_parents = set()  # 防止同一父节点被重复添加

    for doc in docs:
        parent_path = doc.metadata.get("parent_section", "")

        # 如果没有父节点（已是根节点），直接保留
        if not parent_path:
            expanded.append(doc)
            continue

        # 如果这个父节点已经处理过，跳过（避免重复）
        if parent_path in seen_parents:
            continue

        seen_parents.add(parent_path)

        # 找到父节点的所有 chunks（根据 breadcrumb 匹配）
        parent_chunks = [
            d for d in all_docs
            if d.metadata.get("breadcrumb") == parent_path
        ]

        if not parent_chunks:
            # 找不到父节点，保留原 chunk
            expanded.append(doc)
            continue

        # 按 global_chunk_index 排序（保证顺序）
        parent_chunks.sort(key=lambda x: x.metadata.get("global_chunk_index", 0))

        # 合并父节点的所有内容
        merged_content = "\n\n".join([p.page_content for p in parent_chunks])

        # 创建扩展后的文档（metadata 保留原始检索 chunk 的信息）
        expanded_doc = Document(
            page_content=merged_content,
            metadata={
                **doc.metadata,
                "expanded_from": doc.metadata.get("breadcrumb", ""),  # 记录原始检索来源
                "expansion_type": "parent_merge",
                "original_content_length": len(doc.page_content),
                "expanded_content_length": len(merged_content),
            }
        )

        expanded.append(expanded_doc)

    return expanded


# ========== 混合检索 ==========
def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    all_docs: List[Document] = None,
    top_k: int = 10,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Document]:
    """
    混合检索：Dense (FAISS) + BM25

    Args:
        query: 检索query
        vectorstore: FAISS向量库
        all_docs: 全部文档列表（用于BM25，如果为None则只用Dense）
        top_k: 返回前k条
        dense_weight: Dense权重
        bm25_weight: BM25权重

    Returns:
        融合后的Top K文档
    """
    # Dense 检索
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 3},  # 多召回一些作为候选
    )
    dense_docs = dense_retriever.invoke(query)

    # 如果没有 BM25 或没有文档列表，只返回Dense结果
    if not BM25_AVAILABLE or not all_docs:
        return dense_docs[:top_k]

    # BM25 检索
    import jieba

    # 分词
    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
    tokenized_query = list(jieba.cut(query))

    # 构建BM25索引
    bm25 = BM25Okapi(tokenized_corpus)

    # 计算BM25分数
    bm25_scores = bm25.get_scores(tokenized_query)

    # 获取Top N的索引
    top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 3]
    bm25_docs = [all_docs[i] for i in top_n_indices]

    # 融合：计算加权分数
    doc_scores = {}

    # Dense 分数（用排名的倒数作为分数）
    for rank, doc in enumerate(dense_docs, 1):
        # 用内容作为唯一标识（更可靠）
        doc_key = doc.page_content[:100]
        doc_scores[doc_key] = {
            'doc': doc,
            'score': dense_weight * (1.0 / rank),
        }

    # BM25 分数
    for rank, doc in enumerate(bm25_docs, 1):
        doc_key = doc.page_content[:100]
        if doc_key in doc_scores:
            doc_scores[doc_key]['score'] += bm25_weight * (1.0 / rank)
        else:
            doc_scores[doc_key] = {
                'doc': doc,
                'score': bm25_weight * (1.0 / rank),
            }

    # 按分数排序
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)

    # 返回Top K
    return [item['doc'] for item in sorted_docs[:top_k]]


# ========== Query Rewrite ==========
def rewrite_query(question: str, rewrite_llm: ChatOpenAI) -> str:
    """
    使用 LLM 把用户问题改写成适合检索的关键词 query。
    """
    prompt = ChatPromptTemplate.from_template(
        "你是一个检索助手，请将下面的用户问题改写成适合在技术文档中检索的简短查询语句，"
        "保留关键信息，可以适当加入可能的同义词或专业术语，不要客套话，直接输出改写结果：\n\n"
        "用户问题：{question}"
    )
    chain = prompt | rewrite_llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()


# ========== RAG Pipeline ==========
def build_rag_chain(answer_llm: ChatOpenAI):
    """
    构建 RAG LCEL 管道：
    输入: {"question": 原始问题, "context": [Document, ...]}
    输出: 答案字符串（通过 .stream() 流式生成）
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是车企智能座舱导航团队的内部知识库助手，"
                "请严格根据提供的上下文回答问题；"
                "如果上下文里没有答案，就直说不知道，不要瞎编。\n\n"
                "上下文：\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["context"]),  # 把 docs 转成字符串给 prompt
        }
        | prompt
        | answer_llm
        | StrOutputParser()
    )

    return rag_chain


def format_docs(docs: List[Document]) -> str:
    """把多个 Document 拼成一个大字符串喂给 LLM"""
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "未知文件")
        page = d.metadata.get("page", None)
        section = d.metadata.get("section", None)

        header = f"[{i}] {source}"
        if section:
            header += f" - {section}"
        elif page is not None:
            header += f" - 页码 {page}"

        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)


# ========== Streamlit UI ==========

def main():
    st.set_page_config(
        page_title="导航知识库助手（RAG）",
        page_icon="🧭",
        layout="wide",
    )

    st.title("🧭 导航知识库助手（RAG v0.1）")
    st.markdown(
        "基于团队内部文档（PDF / Word）构建的本地 RAG 问答系统，用于支持智能座舱导航业务知识查询。"
    )

    # 初始化session state
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "session_id" not in st.session_state:
        # 生成唯一的session ID（用于日志追踪）
        st.session_state["session_id"] = str(uuid.uuid4())[:8]

    # 左右布局：左侧问答，右侧显示来源
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("💬 提问")
        question = st.text_area(
            "请输入你的问题：",
            placeholder="例如：高德地图推包流程是什么？ / 代码提交流程是怎样的？",
            height=100,
        )

        # 检索设置
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            search_type = st.selectbox(
                "🔍 检索方式",
                ["Dense", "Hybrid"] if BM25_AVAILABLE else ["Dense"],
                help="Dense: 纯向量检索 | Hybrid: 向量+BM25混合"
            )
        with col2:
            use_rewrite = st.checkbox("🔄 Query改写", value=True, help="使用LLM改写问题为检索关键词")
        with col3:
            use_reranker = st.checkbox("✨ 重排序", value=True, help="使用CrossEncoder重排序")
        with col4:
            top_k = st.number_input("返回文档数", min_value=3, max_value=10, value=6, help="最终返回的文档数量")

        if st.button("发送", type="primary"):
            # 初始化变量
            start_time = time.time()
            retrieval_start = None
            retrieval_latency = None
            llm_start = None
            llm_latency = None
            error_info = None
            error_type = None
            rewritten_query = None
            docs = []
            full_answer = ""
            strategy = None  # 保存策略信息

            try:
                if not question.strip():
                    st.warning("请先输入问题。")
                    return

                # 0. Query 分类与策略选择（Adaptive RAG）
                with st.spinner("🤖 分析问题类型..."):
                    strategy = classify_and_get_strategy(question)

                # 显示分类结果
                classification = strategy.get("classification", {})
                category = classification.get("category", "unknown")
                confidence = classification.get("confidence", 0)
                reasoning = classification.get("reasoning", "")

                st.info(f"""
**🧠 Query分类结果**
- **类型**: `{category}`
- **置信度**: {confidence:.2f}
- **理由**: {reasoning}
                """)

                # 根据策略决定是否跳过检索
                if strategy.get("skip_retrieval", False):
                    # no_retrieval: 直接LLM回答，不检索
                    st.caption("💬 此问题无需检索知识库，直接回答")

                    # 加载LLM
                    with st.spinner("加载模型中..."):
                        _, answer_llm = get_llms()

                    # 直接生成答案
                    st.subheader("🧠 回答")
                    answer_placeholder = st.empty()
                    full_answer = ""

                    # 使用简单的对话prompt
                    from langchain_core.prompts import ChatPromptTemplate
                    simple_prompt = ChatPromptTemplate.from_messages([
                        ("system", "你是一个友好的助手。"),
                        ("human", "{question}")
                    ])

                    simple_chain = simple_prompt | answer_llm | StrOutputParser()

                    llm_start = time.time()
                    for chunk in simple_chain.stream({"question": question}):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer)

                    llm_latency = (time.time() - llm_start) * 1000

                    # 保存到对话历史
                    st.session_state["history"].append(
                        {"question": question, "answer": full_answer, "sources": []}
                    )

                    # 跳过后续检索流程
                    return

                # 如果需要检索，继续原有流程
                # 加载资源
                with st.spinner("加载向量库和模型中..."):
                    vectorstore, all_docs = load_vectorstore()
                    rewrite_llm, answer_llm = get_llms()

                # 根据策略覆盖部分用户设置
                adaptive_top_k = strategy.get("top_k", top_k)
                adaptive_use_reranker = strategy.get("use_reranker", use_reranker)
                adaptive_retrieval_mode = strategy.get("retrieval_mode", "dense")

                st.caption(f"""
**🎯 Adaptive RAG 策略**
检索方式: {adaptive_retrieval_mode.upper()} | Top-K: {adaptive_top_k} | Query改写: {'✓' if use_rewrite else '✗'} (用户设置) | Reranker: {'✓' if adaptive_use_reranker else '✗'}
                """)

                # 1. Query Rewrite (由用户勾选决定)
                if use_rewrite:
                    with st.spinner("正在改写检索 Query..."):
                        rewritten_query = rewrite_query(question, rewrite_llm)
                    st.write(f"✏️ **检索用改写：** `{rewritten_query}`")
                else:
                    rewritten_query = question
                    st.write(f"✏️ **使用原问题检索**")

                # 2. 检索文档（根据策略选择检索方式）
                retrieval_start = time.time()
                candidate_k = strategy.get("candidate_k", 20)

                with st.spinner("正在检索相关文档..."):
                    if adaptive_retrieval_mode == "hybrid" and BM25_AVAILABLE:
                        # 混合检索：Dense + BM25
                        candidate_docs = hybrid_retrieve(
                            query=rewritten_query,
                            vectorstore=vectorstore,
                            all_docs=all_docs,
                            top_k=candidate_k,
                            dense_weight=0.6,
                            bm25_weight=0.4
                        )
                        st.caption(f"🔍 使用 Hybrid 检索（召回 {len(candidate_docs)} 条候选）")
                    else:
                        # Dense检索
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": candidate_k},
                        )
                        candidate_docs = retriever.invoke(rewritten_query)
                        st.caption(f"🔍 使用 Dense 检索（召回 {len(candidate_docs)} 条候选）")

                if not candidate_docs:
                    st.error("没有检索到相关文档，可能语料里还没有相关内容。")
                    return

                # 3. CrossEncoder 重排序 (根据策略决定)
                if adaptive_use_reranker:
                    reranker = get_reranker()
                    if reranker:
                        with st.spinner("正在重排序文档..."):
                            docs = reranker.rerank(rewritten_query, candidate_docs, top_k=adaptive_top_k)
                        st.caption(f"✨ CrossEncoder 精排，返回 Top {len(docs)}")
                    else:
                        docs = candidate_docs[:adaptive_top_k]
                        st.caption(f"⚠️ Reranker未加载，直接返回 Top {adaptive_top_k}")
                else:
                    docs = candidate_docs[:adaptive_top_k]
                    st.caption(f"📋 直接返回 Top {adaptive_top_k}")

                # 4. Parent 内容扩展 (根据策略决定)
                if strategy.get("expand_context", False):
                    with st.spinner("正在扩展父节点上下文..."):
                        original_count = len(docs)
                        docs = expand_with_parent_content(docs, all_docs)
                        st.caption(f"🌳 Parent 扩展：{original_count} → {len(docs)} 个文档（去重后）")

                retrieval_latency = (time.time() - retrieval_start) * 1000  # 转为毫秒

                # 5. 构建 RAG 链，流式生成答案
                st.subheader("🧠 回答")

                answer_placeholder = st.empty()
                full_answer = ""

                rag_chain = build_rag_chain(answer_llm)

                llm_start = time.time()
                with st.spinner("正在生成回答..."):
                    for chunk in rag_chain.stream({"question": question, "context": docs}):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer)

                llm_latency = (time.time() - llm_start) * 1000  # 转为毫秒

                # 保存到对话历史
                st.session_state["history"].append(
                    {"question": question, "answer": full_answer, "sources": docs}
                )

            except Exception as e:
                error_info = str(e)
                error_type = type(e).__name__
                st.error(f"❌ 发生错误: {error_info}")

            finally:
                # 记录日志（包含分类和策略信息）
                total_latency = (time.time() - start_time) * 1000
                logger = get_rag_logger()

                # 构建日志数据
                log_data = {
                    "session_id": st.session_state["session_id"],
                    "query": question,
                    "rewritten_query": rewritten_query,
                    "use_rewriter": use_rewrite,
                    "use_reranker": use_reranker,
                    "use_hybrid": (search_type == "Hybrid") if 'search_type' in locals() else False,
                    "top_k": top_k,
                    "retrieval_docs": docs,
                    "answer": full_answer,
                    "latency_ms": total_latency,
                    "retrieval_latency_ms": retrieval_latency,
                    "llm_latency_ms": llm_latency,
                    "error": error_info,
                    "error_type": error_type,
                }

                # 添加Adaptive RAG信息（如果有）
                if strategy:
                    log_data["query_classification"] = strategy.get("classification", {})
                    log_data["adaptive_strategy"] = {
                        "category": strategy.get("classification", {}).get("category"),
                        "skip_retrieval": strategy.get("skip_retrieval", False),
                        "retrieval_mode": strategy.get("retrieval_mode"),
                        "top_k": strategy.get("top_k"),
                        "use_reranker": strategy.get("use_reranker"),
                    }

                logger.log_query(**log_data)

                # 在UI上显示日志路径（仅开发时）
                if error_info is None:
                    st.caption(f"📝 查询已记录到日志 | Session: {st.session_state['session_id']} | 耗时: {total_latency:.0f}ms")

        # 显示历史对话（在按钮外部，避免点击其他组件时消失）
        if "history" in st.session_state and st.session_state["history"]:
            with st.expander("📝 最近一次问答", expanded=False):
                last_turn = st.session_state["history"][-1]

                st.markdown("**问题：**")
                st.info(last_turn["question"])

                st.markdown("**回答：**")
                st.markdown(last_turn["answer"])

    # 右侧：显示来源
    with right_col:
        st.subheader("📘 本次检索到的文档片段")

        if "history" in st.session_state and st.session_state["history"]:
            last_turn = st.session_state["history"][-1]
            docs = last_turn["sources"]

            for i, d in enumerate(docs, 1):
                source = d.metadata.get("source", "未知文件")
                page = d.metadata.get("page", None)
                section = d.metadata.get("section", None)
                doc_type = d.metadata.get("doc_type", "")

                # 优先显示section，其次是page
                location_str = ""
                if section:
                    location_str = f" - {section}"
                elif page is not None:
                    location_str = f" - 页码 {page}"

                snippet = d.page_content[:200].replace("\n", " ")

                with st.expander(f"[{i}] {source}{location_str}", expanded=(i == 1)):
                    if doc_type:
                        st.caption(f"类型: {doc_type}")
                    st.write(snippet + ("..." if len(d.page_content) > 200 else ""))
        else:
            st.info("提交问题后，这里会显示相关文档来源。")


if __name__ == "__main__":
    main()
