"""
Debug Excel加载，查看原始数据结构和生成的chunks
"""

from pathlib import Path
import pandas as pd
from loaders import _load_excel_as_rows

def debug_excel_file(excel_path):
    print("=" * 80)
    print(f"📊 Debug Excel文件: {excel_path.name}")
    print("=" * 80)

    # 1. 读取原始Excel，查看列结构
    print("\n【1】原始Excel列结构:")
    print("-" * 80)
    df = pd.read_excel(excel_path, engine='openpyxl')
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}\n")

    print("列名列表:")
    for i, col in enumerate(df.columns):
        print(f"  列{i}: {col}")

    # 2. 显示前5行原始数据
    print("\n【2】前5行原始数据:")
    print("-" * 80)
    print(df.head(5).to_string())

    # 3. 加载为chunks
    print("\n【3】加载为chunks:")
    print("-" * 80)
    docs = _load_excel_as_rows(excel_path)
    print(f"生成chunks数量: {len(docs)}\n")

    # 4. 显示前10个chunk的详细内容
    print("【4】前10个chunk内容:")
    print("=" * 80)
    for i, doc in enumerate(docs[:10], 1):
        print(f"\n{'='*80}")
        print(f"Chunk {i}:")
        print(f"{'='*80}")
        print(f"来源: {doc.metadata.get('source')}")
        print(f"行号: {doc.metadata.get('row_number')}")
        print(f"\n内容:\n{'-'*80}")
        print(doc.page_content)
        print(f"{'-'*80}")

    # 5. 检查是否有信息丢失
    print("\n【5】信息完整性检查:")
    print("=" * 80)

    # 检查第一行数据在chunk中的表现
    first_row = df.iloc[0]
    first_chunk = docs[0] if docs else None

    if first_chunk:
        print("\n原始第一行数据:")
        for col in df.columns:
            val = first_row[col]
            if pd.notna(val) and str(val).strip():
                print(f"  {col}: {val}")

        print("\n生成的第一个chunk内容:")
        print(first_chunk.page_content)

        print("\n⚠️  检查哪些列可能丢失:")
        chunk_content = first_chunk.page_content
        for col in df.columns:
            val = first_row[col]
            if pd.notna(val) and str(val).strip():
                if str(val) not in chunk_content:
                    print(f"  ❌ 列 '{col}' 的值 '{val}' 未出现在chunk中")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / "data" / "documents"

    # 查找埋点需求Excel文件
    excel_files = list(docs_dir.glob("*埋点*.xlsx")) + list(docs_dir.glob("*埋点*.xls"))

    if not excel_files:
        print("❌ 没有找到埋点相关的Excel文件")
    else:
        for excel_file in excel_files:
            debug_excel_file(excel_file)
            print("\n\n")
