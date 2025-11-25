# app.py
import os
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

# 读取 .env（deepseek 的 key / base_url）
load_dotenv(override=True)


# ========== 缓存一些重资源：向量库、模型 ==========

@st.cache_resource
def load_vectorstore():
    """加载已经构建好的 FAISS 向量库（只加载一次，后面复用）"""
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
    return vectorstore


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
        header = f"[{i}] {source}"
        if page is not None:
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

    # 左右布局：左侧问答，右侧显示来源
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("💬 提问")
        question = st.text_area(
            "请输入你的问题：",
            placeholder="例如：高德地图推包流程是什么？ / 代码提交流程是怎样的？",
            height=100,
        )

        if "history" not in st.session_state:
            st.session_state["history"] = []

        if st.button("发送", type="primary"):

            if not question.strip():
                st.warning("请先输入问题。")
                return

            # 加载资源
            with st.spinner("加载向量库和模型中..."):
                vectorstore = load_vectorstore()
                rewrite_llm, answer_llm = get_llms()

            # 1. Query Rewrite
            with st.spinner("正在改写检索 Query..."):
                rewritten_query = rewrite_query(question, rewrite_llm)
            st.write(f"✏️ **检索用改写：** `{rewritten_query}`")

            # 2. MMR 检索
            with st.spinner("正在检索相关文档..."):
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 20},
                )
                docs = retriever.invoke(rewritten_query)

            if not docs:
                st.error("没有检索到相关文档，可能语料里还没有相关内容。")
                return

            # 3. 构建 RAG 链，流式生成答案
            st.subheader("🧠 回答")

            answer_placeholder = st.empty()
            full_answer = ""

            rag_chain = build_rag_chain(answer_llm)

            with st.spinner("正在生成回答..."):
                for chunk in rag_chain.stream({"question": question, "context": docs}):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer)

            # 保存到对话历史
            st.session_state["history"].append(
                {"question": question, "answer": full_answer, "sources": docs}
            )

    # 右侧：显示来源
    with right_col:
        st.subheader("📘 本次检索到的文档片段")

        if "history" in st.session_state and st.session_state["history"]:
            last_turn = st.session_state["history"][-1]
            docs = last_turn["sources"]

            for i, d in enumerate(docs, 1):
                source = d.metadata.get("source", "未知文件")
                page = d.metadata.get("page", None)
                page_str = f" - 页码 {page}" if page is not None else ""
                snippet = d.page_content[:200].replace("\n", " ")

                with st.expander(f"[{i}] {source}{page_str}", expanded=(i == 1)):
                    st.write(snippet + ("..." if len(d.page_content) > 200 else ""))
        else:
            st.info("提交问题后，这里会显示相关文档来源。")


if __name__ == "__main__":
    main()
