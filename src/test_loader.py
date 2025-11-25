# test_loaders.py
from src.loaders import load_documents,load_single_document

def test_load_docs():
    docs = load_single_document("../data/documents/PIS-2116_Location Based Service_A-V0.0.2.3.docx")

    print(f"加载到 document 数量：{len(docs)}")

    # 打印前 3 个文档的简要信息（防止打印爆屏）
    for idx, d in enumerate(docs[:5], 1):
        snippet = d.page_content[:150].replace("\n", " ")
        section = d.metadata.get("section", "⚠️ 无 section（可能是 fallback loader）")
        print(f"{idx}. {section}")
        print(f"   source={d.metadata.get('source')} page={d.metadata.get('page', '-')}")
        print(f"   {snippet}\n")
    print("测试完成！")


if __name__ == "__main__":
    test_load_docs()
