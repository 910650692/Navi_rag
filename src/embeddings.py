# src/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """返回一个中文友好的 embedding 模型实例。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},            # 如果有显卡可以改成 "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )

