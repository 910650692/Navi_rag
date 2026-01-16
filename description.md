 导航知识库 RAG 问答系统
  技术栈：LangChain、FAISS、DeepSeek、BGE Embedding、CrossEncoder、Ragas、Streamlit

  【项目背景】
  为智能座舱导航团队构建内部文档问答系统，支持 70 份 DOCX/PDF/Excel 规范文档（约 10000+ chunks）的智能检索。

  【核心工作】
  • 实现 Adaptive RAG 框架：基于 Qwen2.5-7B 对 Query 进行三分类（no_retrieval / simple_lookup / complex_reasoning），
    动态调整检索策略（top_k=4/10、是否启用 reranker、是否扩展上下文），相比固定策略提升 Context Precision 8%。

  • 设计层级化文档处理方案：
    - 基于 python-docx 提取 DOCX 章节结构，保留 section_number、breadcrumb、parent_section 等 8 个 metadata 字段
    - 实现 PDF 多规则标题识别（字号+加粗+编号融合），准确率 80%+
    - 根据文档类型自适应调整 chunk_size（流程文档 600 字符、规范文档 1200 字符），避免表格被切断

  • 对比 Dense / Hybrid(Dense+BM25) / Dense+Reranker 三种检索方案：
    - 发现导航文档关键词重复率高（如"路线"、"导航"），BM25 容易召回无关章节，反而降低 Precision（0.85→0.83）
    - 最终选用 Dense(BGE-small-zh) + CrossEncoder(bge-reranker-base) 方案，在 2s 延迟内达到 0.92/0.95 的 Precision/Recall

  • 实现 Parent Chunk Merging：检索到小 chunk 后，根据 parent_section metadata 自动合并父节点完整内容，
    解决表格查询上下文不足问题，准确率从 30% 提升至 95%。

  • 基于 Ragas 构建评估流程，标注 20 条测试数据集，通过 Streamlit 面板对比不同策略组合效果，
    并实现日志系统记录每次检索链路（Query分类 → 检索模式 → 候选数 → 重排序 → 上下文扩展），支持 bad case 分析。

  【技术收获】
  • 理解了 RAG 系统中"检索精度"与"上下文完整性"的权衡（小 chunk 检索准，但上下文不足；大 chunk 上下文完整，但召回不准）
  • 学会通过实验验证技术选型（不盲目追求"先进"，BM25 在语义密集场景反而降低效果）
  • 掌握了从 Metadata 设计到检索后处理的完整 RAG 优化链路