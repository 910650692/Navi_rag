import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from embeddings import get_embeddings
from src.reranker import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



load_dotenv(override=True)

DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

FIXED_INDEX_PATH = Path(__file__).parent.parent / "data" / "index" / "pis2116_single"

def load_vectorstore(index_path: Path):
    """加载已经构建好的 FAISS 向量库"""
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} does not exist.")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True)
    return vectorstore


def build_rag_chain(retriever, model):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是车企智能座舱导航团队的内部知识库助手，"
                "请严格根据提供的上下文回答问题；"
                "如果上下文里没有答案，就直说不知道，不要瞎编。\n\n"
                "上下文：\n{context}"
            ),
            ("human", "{input}"),
        ]
    )
    rag_chain = (
        {
            "input": RunnablePassthrough(),
            "context": retriever,
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


def rewrite_query(question: str) -> str:
    rewrite_model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.0,
    )
    rewrite_prompt = ChatPromptTemplate.from_template(
        "请将下面的用户问题改写成适合在技术文档中检索的简短查询语句，"
        "保留关键信息，偏向关键词，不要客套话，直接输出改写结果：\n\n"
        "用户问题：{question}"
    )

    chain = rewrite_prompt | rewrite_model | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单文档RAG测试脚本")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="直接传入问题文本，避免交互输入",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="跳过 query rewrite，直接使用原问题检索",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="检索返回的文档片段数量（默认：4）",
    )
    parser.add_argument(
        "question_parts",
        nargs="*",
        help="命令行直接输入的问题内容（可含空格）",
    )
    return parser


def _collect_question(args: argparse.Namespace) -> str:
    if args.question:
        return args.question.strip()
    if args.question_parts:
        return " ".join(args.question_parts).strip()
    return input("请输入问题：").strip()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    question = _collect_question(args)
    if not question:
        print("输入为空，退出")
        return

    index_path = FIXED_INDEX_PATH

    if args.no_rewrite:
        rewritten_query = question
        print("\n跳过 query 改写，直接使用原问题检索。")
    else:
        print("\n正在改写检索query...")
        rewritten_query = rewrite_query(question)
        print(f"原始问题：{question}")
        print(f"检索用改写：{rewritten_query}")

    print(f"\n📦 正在加载向量库: {index_path}")
    vectorstore = load_vectorstore(index_path)

    # 1. 从向量库里把所有 Document 拿出来，给 BM25 用
    all_docs = list(vectorstore.docstore._dict.values())
    print(f"向量库中共有文档块: {len(all_docs)}")

    # 2. 构建 BM25 稀疏检索器（基于倒排索引，内存里建即可）
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = args.top_k * 3  # 稀疏这边先多取一点候选

    # 3. 构建原来的稠密检索器（FAISS）
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": args.top_k * 3},  # 和 BM25 统一一下数量
    )
    # 4. 用 EnsembleRetriever 把 BM25 + Dense 融合成一个混合检索器
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.7, 0.3],  # 语义 0.7 + 关键词 0.3，后面可以再调
    )

    retriever = hybrid_retriever  # 后面统一用 retriever 这个变量
    # 检索并显示文档片段
    print(f"\n🔍 正在检索相关文档 (top-{args.top_k * 3})...\n")
    # docs = retriever.invoke(rewritten_query)
    docs = retriever.invoke(question)
    reranker = CrossEncoderReranker()
    docs = reranker.rerank(question, docs,top_k=args.top_k)

    if not docs:
        print("没有找到相关文档，请重新提问。")
        return

    print("检索到的文档片段：")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "未知文件")
        page = d.metadata.get("page", "无页码")
        snippet = d.page_content[:300].replace("\n", " ")

        page_str = f"页码：{page}" if page and page != "无页码" else ""
        section = d.metadata.get("section", "")
        print(f"\n{i}. 来源：{source} {page_str}")
        if section:
            print(f"   位置：{section}")
        print(f"   内容：{d.page_content}...")
        print("-" * 80)

    print("\n正在生成回答(流式输出)...\n")

    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.3,
        streaming=True,
    )
    rag_chain = build_rag_chain(retriever, model)

    print(f"❓ 问题：{question}\n")
    print("💬 回答：", end="", flush=True)

    # 流式输出答案
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
