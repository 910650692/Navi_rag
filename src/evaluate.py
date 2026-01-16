"""
导航知识库助手 - RAG 评估脚本 (RAGAS)
使用方法：
    python evaluate.py

先安装依赖：
    pip install ragas datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

# 你自己的 embedding 封装
from embeddings import get_embeddings
from reranker import CrossEncoderReranker
from eval_utils import get_eval_recorder

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent

# 评估集文件：使用刚才生成的 QA jsonl
EVAL_FILE = BASE_DIR / "data" / "nav_rag_eval_set_v1.jsonl"

# 向量库路径：你可以根据自己的项目调整
INDEX_PATH = BASE_DIR / "data" / "index" / "pis2116_single"


RAGAS_METRIC_COLUMNS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对照评估 Dense / Hybrid 检索链路的 RAG 表现（基于 RAGAS）。"
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "both"],
        default="both",
        help="选择评估哪一种检索模式；both 会依次输出 dense/hybrid 两份 CSV。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="最终喂给生成模型的上下文数量（默认 4）。",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="检索候选数量 = top_k * 倍数，用于 rerank/混合融合（至少 1）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="评估结果 CSV 存放目录（默认项目根目录）。",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.7,
        help="Hybrid 模式中稠密检索得分权重。",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.3,
        help="Hybrid 模式中 BM25 检索得分权重。",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="关闭 CrossEncoder reranker（默认开启）。",
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="跳过 Query Rewrite，直接使用原始问题检索。",
    )
    return parser.parse_args()



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
        temperature=0.1,
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
                "你是车企智能座舱导航团队的内部知识库助手。\n"
                "【非常重要】\n"
                "1. 只能依据“上下文”中的内容回答，不要加入任何上下文之外的推测。\n"
                "2. 必须紧扣用户问题作答，不要输出与问题无关的解释或背景。\n"
                "3. 如果问题在上下文中有明确的枚举或列表，请尽量完整列出，不要随意省略。\n"
                "4. 如果上下文没有足够信息，请回答“根据当前文档信息无法确定”。\n\n"
                "下面是检索到的上下文：\n{context}"
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


def _resolve_modes(mode_arg: str) -> List[str]:
    return ["dense", "hybrid"] if mode_arg == "both" else [mode_arg]


def _compute_fetch_k(top_k: int, multiplier: int) -> int:
    multiplier = max(1, multiplier)
    return max(top_k, top_k * multiplier)


def build_retriever(
    vectorstore: FAISS,
    mode: str,
    fetch_k: int,
    dense_weight: float,
    bm25_weight: float,
):
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": fetch_k},
    )

    if mode == "dense":
        return dense_retriever

    all_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
    if not all_docs:
        raise ValueError("向量库 docstore 为空，无法构建 BM25 检索器。")

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = fetch_k
    weights = [dense_weight, bm25_weight]
    return EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=weights)


def _get_rewritten_query(
    question: str,
    rewrite_llm: ChatOpenAI,
    cache: Dict[str, str],
    skip_rewrite: bool,
) -> tuple[str, bool]:
    if skip_rewrite:
        return question, False
    if question in cache:
        return cache[question], False
    rewritten = rewrite_query(question, rewrite_llm)
    cache[question] = rewritten
    return rewritten, True


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


def run_mode_evaluation(
    mode: str,
    eval_items: Sequence[dict],
    vectorstore: FAISS,
    rewrite_llm: ChatOpenAI,
    rag_chain,
    answer_llm: ChatOpenAI,
    ragas_embeddings,
    args: argparse.Namespace,
    rewrite_cache: Dict[str, str],
    use_reranker: bool,
) -> tuple[Any, Path]:
    fetch_k = _compute_fetch_k(args.top_k, args.candidate_multiplier)
    retriever = build_retriever(
        vectorstore=vectorstore,
        mode=mode,
        fetch_k=fetch_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )
    reranker = CrossEncoderReranker() if use_reranker else None

    print(f"\n{'=' * 15} 开始评估模式：{mode.upper()} {'=' * 15}")
    print(f"🎯 候选数量: {fetch_k} | 最终 top-k: {args.top_k} | reranker: {'开启' if reranker else '关闭'}")

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

        print(f"\n----- [{idx}/{len(eval_items)}] {qid} ({mode}) -----")
        print(f"❓ 问题：{question}")
        print(f"🎯 难度：{difficulty} | 来源提示：{source_hint}")

        rewritten, freshly_computed = _get_rewritten_query(
            question, rewrite_llm, rewrite_cache, args.no_query_rewrite
        )
        if args.no_query_rewrite:
            print("✏️ 已禁用 Query Rewrite，直接使用原问题。")
        else:
            prefix = "✏️ 新改写" if freshly_computed else "♻️ 使用缓存改写"
            print(f"{prefix}：{rewritten}")

        docs = retriever.invoke(rewritten)
        if not docs:
            print("⚠️ 未检索到任何文档，上下文留空。")
            filtered_docs: List[Document] = []
        else:
            if reranker:
                filtered_docs = reranker.rerank(question, docs, top_k=args.top_k)
            else:
                filtered_docs = docs[: args.top_k]
            print(f"🔍 检索候选 {len(docs)} -> 选取 {len(filtered_docs)} 条用于回答。")

        ctx_texts = [d.page_content for d in filtered_docs]
        prompt_context = format_docs_for_prompt(filtered_docs) if filtered_docs else ''

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

    out_path = args.output_dir / f"ragas_results_{mode}.csv"
    df = results.to_pandas()
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"📈 模式 {mode} 评估完成，结果写入 {out_path}\n")
    return df, out_path


def summarize_metrics(df) -> Dict[str, float]:
    summary = {}
    for col in RAGAS_METRIC_COLUMNS:
        if col in df.columns:
            summary[col] = float(df[col].mean())
    return summary


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_reranker = not args.disable_reranker

    print(f"📄 正在加载评估集: {EVAL_FILE}")
    eval_items = load_eval_items(EVAL_FILE)
    print(f"✅ 样本数量: {len(eval_items)}")

    print("📦 加载向量库和模型...")
    vectorstore = load_vectorstore()
    rewrite_llm, answer_llm = get_llms()
    rag_chain = build_rag_chain(answer_llm)
    ragas_embeddings = get_embeddings()

    rewrite_cache: Dict[str, str] = {}
    modes = _resolve_modes(args.mode)
    summary_rows = []

    for mode in modes:
        df, out_path = run_mode_evaluation(
            mode=mode,
            eval_items=eval_items,
            vectorstore=vectorstore,
            rewrite_llm=rewrite_llm,
            rag_chain=rag_chain,
            answer_llm=answer_llm,
            ragas_embeddings=ragas_embeddings,
            args=args,
            rewrite_cache=rewrite_cache,
            use_reranker=use_reranker,
        )
        summary = summarize_metrics(df)
        summary_rows.append((mode, summary, out_path))

    print("\n====== 汇总对比 ======")
    recorder = get_eval_recorder()

    for mode, summary, path in summary_rows:
        metric_parts = [
            f"{metric}: {summary.get(metric, float('nan')):.3f}"
            for metric in RAGAS_METRIC_COLUMNS
            if metric in summary
        ]
        print(f"{mode.upper()} -> {', '.join(metric_parts)} | CSV: {path}")

        # 保存评估结果到JSONL
        config = {
            "retriever": mode,
            "use_reranker": use_reranker,
            "top_k": args.top_k,
            "candidate_k": _compute_fetch_k(args.top_k, args.candidate_multiplier),
            "dense_weight": args.dense_weight if mode == "hybrid" else None,
            "bm25_weight": args.bm25_weight if mode == "hybrid" else None,
            "query_rewrite": not args.no_query_rewrite,
        }

        notes = f"评估集: {EVAL_FILE.name}, 样本数: {len(eval_items)}"

        eval_id = recorder.save_eval_result(config=config, metrics=summary, notes=notes)
        print(f"   ✅ 评估结果已保存，ID: {eval_id}\n")


if __name__ == '__main__':
    main()

