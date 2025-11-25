# evaluate.py
"""
导航知识库助手 - RAG 评估脚本 (RAGAS)
使用方法：
    python evaluate.py

先安装依赖：
    pip install ragas datasets
"""

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from datasets import Dataset

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

# 你自己的 embedding 封装
from src.embeddings import get_embeddings

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent
EVAL_FILE = BASE_DIR / "data" / "nav_rag_eval_set_v1.jsonl"
INDEX_PATH = BASE_DIR / "data" / "index" / "nav_faiss"


# =============== 加载向量库 & 模型 ===============

def load_vectorstore():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"找不到向量库目录: {INDEX_PATH}")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_llms():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    # 用于 query rewrite
    rewrite_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        streaming=False,
    )

    # 用于最终回答（评估时不需要流式）
    answer_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
        streaming=False,
    )

    return rewrite_llm, answer_llm


# =============== Query Rewrite & RAG ===============

def rewrite_query(question: str, rewrite_llm: ChatOpenAI) -> str:
    prompt = ChatPromptTemplate.from_template(
        "你是一个检索助手，请将下面的用户问题改写成适合在技术文档中检索的简短查询语句，"
        "保留关键信息，可以适当加入可能的同义词或专业术语，不要客套话，直接输出改写结果：\n\n"
        "用户问题：{question}"
    )
    chain = prompt | rewrite_llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()


def format_docs_for_prompt(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "未知文件")
        page = d.metadata.get("page", None)
        header = f"[{i}] {source}"
        if page is not None:
            header += f" - 页码 {page}"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)


def build_rag_chain(answer_llm: ChatOpenAI):
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
            "question": RunnablePassthrough(),
            "context": lambda x: x["context"],
        }
        | prompt
        | answer_llm
        | StrOutputParser()
    )

    return rag_chain


# =============== 读取 Eval Set ===============

def load_eval_items(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到评估集文件: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# =============== 主流程：跑 RAG + RAGAS ===============

def main():
    print(f"📄 正在加载评估集: {EVAL_FILE}")
    eval_items = load_eval_items(EVAL_FILE)
    print(f"✅ 样本数量: {len(eval_items)}\n")

    print("📦 加载向量库和模型...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )
    rewrite_llm, answer_llm = get_llms()
    rag_chain = build_rag_chain(answer_llm)

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    difficulties = []
    ids = []
    source_hints = []

    for idx, item in enumerate(eval_items, start=1):
        qid = item.get("id", f"Q{idx}")
        question = item["question"]
        gt = item["ground_truth"]
        difficulty = item.get("difficulty", "unknown")
        source_hint = item.get("source_hint", "")

        print(f"\n===== [{idx}/{len(eval_items)}] {qid} =====")
        print(f"❓ 问题：{question}")
        print(f"🎯 难度：{difficulty} | 来源提示：{source_hint}")

        # 1. 改写 Query
        rewritten = rewrite_query(question, rewrite_llm)
        print(f"✏️ 改写后的检索 Query：{rewritten}")

        # 2. 检索
        docs = retriever.invoke(rewritten)
        if not docs:
            print("⚠️ 未检索到任何文档，上下文留空。")
            ctx_texts = []
            prompt_context = ""
        else:
            ctx_texts = [d.page_content for d in docs]
            prompt_context = format_docs_for_prompt(docs)
            print(f"📘 检索到 {len(docs)} 个文档片段。")

        # 3. 生成回答
        answer = rag_chain.invoke(
            {
                "question": question,
                "context": prompt_context,
            }
        )
        print(f"💬 回答：{answer[:200]}{'...' if len(answer) > 200 else ''}")

        # 4. 收集数据
        ids.append(qid)
        questions.append(question)
        answers.append(answer)
        contexts_list.append(ctx_texts)          # list[str]
        ground_truths.append(gt)                # ragas 需要 ground truth 字符串
        difficulties.append(difficulty)
        source_hints.append(source_hint)

    # 构建 HF Dataset
    print("\n📊 构建 RAGAS 数据集...")
    ds = Dataset.from_dict(
        {
            "id": ids,
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
            "difficulty": difficulties,
            "source_hint": source_hints,
        }
    )

    print("✅ 数据集构建完成，开始调用 RAGAS 评估...\n")

    # 注意：这里直接把 answer_llm 和 embeddings 传给 ragas 做打分
    embeddings = get_embeddings()

    results = evaluate(
        ds,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=answer_llm,
        embeddings=embeddings,
    )

    print("📈 评估结果：")
    print(results)

    # 导出为 csv 方便查看
    out_path = BASE_DIR / "ragas_results.csv"
    results.to_pandas().to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 详细评估结果已导出到: {out_path}")


if __name__ == "__main__":
    main()
