"""
导航知识库助手 - 混合检索 RAG 评估脚本 (仅 hybrid 模式)
使用方法：
    python evaluate_hybrid.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from datasets import Dataset

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

from embeddings import get_embeddings
from reranker import CrossEncoderReranker

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent
EVAL_FILE = BASE_DIR / "data" / "nav_rag_eval_set_v1.jsonl"
INDEX_PATH = BASE_DIR / "data" / "index" / "pis2116_single"

# 混合检索参数
TOP_K = 4
CANDIDATE_MULTIPLIER = 3
DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
USE_RERANKER = True
SKIP_QUERY_REWRITE = False

RAGAS_METRIC_COLUMNS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


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

    rewrite_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        streaming=False,
    )

    answer_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        streaming=False,
    )

    return rewrite_llm, answer_llm


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
            ("system",
                "你是车企智能座舱导航团队的内部知识库助手。\n"
                "【非常重要】\n"
                "1. 只能依据“上下文”中的内容回答，不要加入任何上下文之外的推测。\n"
                "2. 必须紧扣用户问题作答，不要输出与问题无关的解释或背景。\n"
                "3. 如果问题在上下文中有明确的枚举或列表，请尽量完整列出，不要随意省略。\n"
                "4. 如果上下文没有足够信息，请回答“根据当前文档信息无法确定”。\n\n"
                "下面是检索到的上下文：\n{context}"),
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


def build_hybrid_retriever(vectorstore: FAISS, fetch_k: int):
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k},
    )

    all_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
    if not all_docs:
        raise ValueError("向量库 docstore 为空，无法构建 BM25 检索器。")

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = fetch_k

    weights = [DENSE_WEIGHT, BM25_WEIGHT]
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=weights
    )


def load_eval_items(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到评估集文件: {path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_hybrid_evaluation(
    eval_items: List[dict],
    vectorstore: FAISS,
    rewrite_llm: ChatOpenAI,
    rag_chain,
    answer_llm: ChatOpenAI,
    ragas_embeddings,
):
    fetch_k = max(TOP_K, TOP_K * CANDIDATE_MULTIPLIER)
    retriever = build_hybrid_retriever(vectorstore, fetch_k)
    reranker = CrossEncoderReranker() if USE_RERANKER else None
    rewrite_cache: Dict[str, str] = {}

    print(f"\n{'=' * 15} 开始评估模式：HYBRID {'=' * 15}")
    print(f"🎯 候选数量: {fetch_k} | 最终 top-k: {TOP_K} | reranker: {'开启' if reranker else '关闭'}")
    print(f"⚖️ Dense权重: {DENSE_WEIGHT} | BM25权重: {BM25_WEIGHT}")

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []
    difficulties: List[str] = []
    ids: List[str] = []
    source_hints: List[str] = []

    for idx, item in enumerate(eval_items, start=1):
        qid = item.get('id', f"Q{idx}")
        question = item['question']
        gt = item['ground_truth']
        difficulty = item.get('difficulty', 'unknown')
        source_hint = item.get('source_hint', '')

        print(f"\n----- [{idx}/{len(eval_items)}] {qid} (hybrid) -----")
        print(f"❓ 问题：{question}")
        print(f"🎯 难度：{difficulty} | 来源提示：{source_hint}")

        # Query Rewrite
        if SKIP_QUERY_REWRITE:
            rewritten = question
            print("✏️ 已禁用 Query Rewrite，直接使用原问题。")
        else:
            if question in rewrite_cache:
                rewritten = rewrite_cache[question]
                print(f"♻️ 使用缓存改写：{rewritten}")
            else:
                rewritten = rewrite_query(question, rewrite_llm)
                rewrite_cache[question] = rewritten
                print(f"✏️ 新改写：{rewritten}")

        # 检索
        docs = retriever.invoke(rewritten)
        if not docs:
            print("⚠️ 未检索到任何文档，上下文留空。")
            filtered_docs: List[Document] = []
        else:
            if reranker:
                filtered_docs = reranker.rerank(question, docs, top_k=TOP_K)
            else:
                filtered_docs = docs[:TOP_K]
            print(f"🔍 检索候选 {len(docs)} -> 选取 {len(filtered_docs)} 条用于回答。")

        ctx_texts = [d.page_content for d in filtered_docs]
        prompt_context = format_docs_for_prompt(filtered_docs) if filtered_docs else ''

        # 生成答案
        answer = rag_chain.invoke(
            {
                'question': question,
                'context': prompt_context,
            }
        )
        print(f"💬 回答：{answer[:200]}{'...' if len(answer) > 200 else ''}")

        ids.append(qid)
        questions.append(question)
        answers.append(answer)
        contexts_list.append(ctx_texts)
        ground_truths.append(gt[0] if isinstance(gt, list) else gt)
        difficulties.append(difficulty)
        source_hints.append(source_hint)

    print("\n📊 构建 RAGAS 数据集...")
    ds = Dataset.from_dict(
        {
            'id': ids,
            'question': questions,
            'answer': answers,
            'contexts': contexts_list,
            'ground_truth': ground_truths,
            'difficulty': difficulties,
            'source_hint': source_hints,
        }
    )

    print("✅ 数据集构建完成，调用 RAGAS 评估...\n")

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
        embeddings=ragas_embeddings,
    )

    out_path = BASE_DIR / "ragas_results_hybrid.csv"
    df = results.to_pandas()
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"📈 混合检索评估完成，结果写入 {out_path}\n")

    # 打印汇总指标
    print("\n====== 汇总指标 ======")
    for col in RAGAS_METRIC_COLUMNS:
        if col in df.columns:
            mean_val = df[col].mean()
            print(f"{col}: {mean_val:.3f}")

    return df, out_path


def main():
    print(f"📄 正在加载评估集: {EVAL_FILE}")
    eval_items = load_eval_items(EVAL_FILE)
    print(f"✅ 样本数量: {len(eval_items)}")

    print("📦 加载向量库和模型...")
    vectorstore = load_vectorstore()
    rewrite_llm, answer_llm = get_llms()
    rag_chain = build_rag_chain(answer_llm)
    ragas_embeddings = get_embeddings()

    df, out_path = run_hybrid_evaluation(
        eval_items=eval_items,
        vectorstore=vectorstore,
        rewrite_llm=rewrite_llm,
        rag_chain=rag_chain,
        answer_llm=answer_llm,
        ragas_embeddings=ragas_embeddings,
    )

    print(f"\n✨ 完成！结果已保存至: {out_path}")


if __name__ == '__main__':
    main()
