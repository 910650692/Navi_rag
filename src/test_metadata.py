"""
测试脚本：验证增强的层级metadata是否正确生成

用法:
python test_metadata.py <文档路径> [--random] [--count N]

参数:
  --random: 随机查看chunks（默认查看前5个）
  --count N: 指定查看的chunk数量（默认5个）

示例:
python test_metadata.py "../data/documents/PIS-2116.docx"
python test_metadata.py "../data/documents/PIS-2116.docx" --random
python test_metadata.py "../data/documents/PIS-2116.docx" --random --count 10
"""

import sys
from pathlib import Path
from loaders import load_single_document
import json
import random


def test_document_metadata(file_path: str, use_random: bool = False, count: int = 5):
    """测试单个文档的metadata生成"""
    print(f"\n{'=' * 80}")
    print(f"📄 测试文档: {file_path}")
    print(f"{'=' * 80}\n")

    try:
        docs = load_single_document(file_path)
        print(f"✅ 成功加载，共 {len(docs)} 个chunks\n")

        # 选择要显示的chunks
        if use_random:
            sample_docs = random.sample(docs, min(count, len(docs)))
            print(f"🎲 随机抽取 {len(sample_docs)} 个chunks进行展示\n")
        else:
            sample_docs = docs[:count]
            print(f"📋 显示前 {len(sample_docs)} 个chunks\n")

        # 显示选中的chunks的metadata
        print(f"{'=' * 80}")
        print(f"📊 Chunks的metadata详情:")
        print(f"{'=' * 80}\n")

        for i, doc in enumerate(sample_docs, 1):
            # 显示chunk在原列表中的位置
            original_index = docs.index(doc) if use_random else i - 1
            print(f"\n--- Chunk #{original_index + 1} (共{len(docs)}个) ---")
            print(f"内容预览: {doc.page_content[:150]}...")
            print(f"\n🏷️  Metadata:")

            # 按类别分组显示metadata
            metadata = doc.metadata

            # 基础信息
            print(f"\n  [基础信息]")
            for key in ['source', 'doc_type', 'file_type']:
                if key in metadata:
                    print(f"    {key}: {metadata[key]}")

            # 层级信息
            print(f"\n  [层级信息]")
            for key in ['section', 'breadcrumb', 'section_level', 'section_number', 'section_title']:
                if key in metadata:
                    print(f"    {key}: {metadata[key]}")

            # 关系信息
            print(f"\n  [关系信息]")
            for key in ['root_section', 'parent_section', 'global_chunk_index']:
                if key in metadata:
                    print(f"    {key}: {metadata[key]}")

            # Excel特有字段
            excel_keys = ['row_number', 'level1', 'level2', 'level3', 'level4', 'hierarchy_path']
            excel_metadata = {k: v for k, v in metadata.items() if k in excel_keys}
            if excel_metadata:
                print(f"\n  [Excel层级]")
                for key, value in excel_metadata.items():
                    if value is not None:
                        print(f"    {key}: {value}")

        # 统计信息
        print(f"\n\n{'=' * 80}")
        print(f"📈 统计信息:")
        print(f"{'=' * 80}")

        # 统计有section的chunks
        chunks_with_section = sum(1 for d in docs if 'section' in d.metadata)
        print(f"  带section的chunks: {chunks_with_section}/{len(docs)}")

        # 统计有section_number的chunks
        chunks_with_number = sum(1 for d in docs if 'section_number' in d.metadata)
        print(f"  带section_number的chunks: {chunks_with_number}/{len(docs)}")

        # 显示一些有section_number的chunk示例
        if chunks_with_number > 0:
            print(f"\n  📌 Section_number 示例 (随机3个):")
            docs_with_number = [d for d in docs if 'section_number' in d.metadata]
            for doc in random.sample(docs_with_number, min(3, len(docs_with_number))):
                print(f"    {doc.metadata['section_number']} → {doc.metadata.get('section_title', 'N/A')}")

        # 统计section_level分布
        from collections import Counter
        level_dist = Counter(d.metadata.get('section_level') for d in docs if 'section_level' in d.metadata)
        if level_dist:
            print(f"\n  Section层级分布:")
            for level, count in sorted(level_dist.items()):
                print(f"    Level {level}: {count} chunks")

        # 统计root_section分布
        root_dist = Counter(d.metadata.get('root_section') for d in docs if 'root_section' in d.metadata)
        if root_dist:
            print(f"\n  根节点分布 (Top 5):")
            for root, count in root_dist.most_common(5):
                print(f"    {root}: {count} chunks")

        # Excel特有统计
        if any('level1' in d.metadata for d in docs):
            print(f"\n  Excel层级统计:")
            level1_dist = Counter(d.metadata.get('level1') for d in docs if d.metadata.get('level1'))
            print(f"    Level1分类数: {len(level1_dist)}")
            for level1, count in level1_dist.most_common(3):
                print(f"      {level1}: {count} 行")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_metadata.py <文档路径> [--random] [--count N]")
        print("\n示例:")
        print('  python test_metadata.py "../data/documents/PIS-2116.docx"')
        print('  python test_metadata.py "../data/documents/PIS-2116.docx" --random')
        print('  python test_metadata.py "../data/documents/PIS-2116.docx" --random --count 10')
        print('  python test_metadata.py "../data/documents/高德地图埋点需求.xlsx" --random')
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    # 解析参数
    use_random = '--random' in sys.argv
    count = 5

    if '--count' in sys.argv:
        try:
            count_idx = sys.argv.index('--count')
            count = int(sys.argv[count_idx + 1])
        except (IndexError, ValueError):
            print("❌ --count 参数格式错误，使用默认值 5")
            count = 5

    test_document_metadata(file_path, use_random=use_random, count=count)
