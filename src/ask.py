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

load_dotenv(override=True)

DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
DEFAULT_INDEX_NAME = "nav_faiss"


def _resolve_index_path(
    index_path: str | Path | None = None,
    index_dir: str | Path | None = None,
    index_name: str | None = None,
) -> Path:
    if index_path:
        return Path(index_path)
    base_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    target_name = index_name or DEFAULT_INDEX_NAME
    return base_dir / target_name


def load_vectorstore(index_path: Path):
    """加载已经构建好的 FAISS 向量库"""
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} does not exist.")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization= True)
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
    parser = argparse.ArgumentParser(description="RAG 问答脚本")
    parser.add_argument(
        "--index-path",
        type=Path,
        help="显式指定向量库目录，若填写则忽略 index-dir / index-name",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help=f"向量库目录（默认：{DEFAULT_INDEX_DIR}）",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help=f"向量库名称（默认：{DEFAULT_INDEX_NAME}）",
    )
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

    index_path = _resolve_index_path(
        index_path=args.index_path,
        index_dir=args.index_dir,
        index_name=args.index_name,
    )

    if args.no_rewrite:
        rewritten_query = question
        print("\n跳过 query 改写，直接使用原问题检索。")
    else:
        print("\n 正在改写检索query...")
        rewritten_query = rewrite_query(question)
        print(f"原始问题：{question}")
        print(f"检索用改写：{rewritten_query}")

    print(f"📦 正在加载向量库: {index_path}")
    vectorstore = load_vectorstore(index_path)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4,"fetch_k": 20})

    # 3. 先检索一遍，打印来源
    print("\n🔍 正在检索相关文档...\n")
    docs = retriever.invoke(rewritten_query)

    if not docs:
        print("没有找到相关文档，请重新提问。")
        return
    print("检索到的文档片段（基于改写后的 query）：")
    for i,d in enumerate(docs,1):
        source = d.metadata.get("source","未知文件")
        page = d.metadata.get("page", "无页码")
        snippet = d.page_content[:200].replace("\n", " ")

        page_str = f"页码：{page}" if page else ""
        print(f"{i}. 来源：{source} {page_str}\n{snippet}\n")
        print("-" * 50)

    print("\n 正在生成回答(流式输出) ...\n")

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

    # 5. 流式输出答案
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

    print("\n")

if __name__ == "__main__":
    main()
