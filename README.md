# 🧭 导航知识库RAG系统

> 基于LangChain 1.0和FAISS的智能座舱导航业务知识问答系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 项目简介

本项目是为车企智能座舱导航团队构建的**内部知识库问答系统**，旨在帮助团队成员快速查询技术规范、接口文档、流程规范等业务知识。

### 核心特性

- 📚 **多格式文档支持**：DOCX层级提取、PDF智能解析、表格识别
- 🔍 **自适应切分策略**：根据文档类型动态调整chunk_size（600-1200字符）
- 🎯 **精准检索**：Dense向量检索 + CrossEncoder重排序
- 💬 **流式问答**：基于DeepSeek Chat的实时流式回答
- 📊 **可视化界面**：Streamlit Web UI，支持检索策略实时调整

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户问题                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   Query Rewrite (可选)   │  ← LLM改写为检索关键词
         └─────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────────────┐
         │   Dense 向量检索 (FAISS) │  ← 召回20条候选
         │   + BGE-small-zh-v1.5   │
         └─────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────────────┐
         │ CrossEncoder 重排序      │  ← 精排 Top 6
         │ + bge-reranker-base     │
         └─────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────────────┐
         │   RAG 生成答案           │  ← DeepSeek Chat
         │   + 流式输出             │
         └─────────────────────────┘
```

---

## 📊 技术亮点与实验结果

### 1. 自适应文档处理

不同类型的文档采用不同的处理策略：

| 文档类型 | 格式 | 切分策略 | chunk_size | 理由 |
|---------|------|----------|-----------|------|
| **导航规范** (nav_spec) | DOCX | 层级切分 | 1200 | 文档大、结构复杂，需保留完整语义 |
| **API规范** (api_spec) | DOCX | 层级切分 | 1000 | 接口定义需要完整，避免切断表格 |
| **流程文档** (process_spec) | DOCX | 层级切分 | 600 | 步骤独立，小chunk更精确 |
| **PDF文档** | PDF | 智能识别标题 | 800-1200 | 基于字体大小、加粗、编号识别层级 |

**效果**：
- DOCX文档层级提取准确率 **95%+**
- PDF标题识别准确率 **80%+**（基于多规则融合）
- 表格完整保留率 **100%**

### 2. 检索策略选择

**实验对比**（基于20个测试问题）：

| 方案 | Context Precision | Context Recall | 平均延迟 | 说明 |
|------|------------------|----------------|---------|------|
| Dense | 0.85 | 0.92 | 1.2s | 纯向量检索 |
| Hybrid (Dense+BM25) | 0.83 | 0.94 | 1.8s | 混合检索 |
| **Dense + Reranker** | **0.92** | **0.95** | 2.1s | ✅ 最终方案 |

**关键发现**：
1. ✅ **Dense检索对语义密集的导航文档已足够有效**（Precision 0.85）
2. ❌ **BM25提升有限**（+0.02），反而增加延迟（+0.6s）
   - 原因：导航文档中关键词在不同章节重复率高（如"路线"、"导航"），BM25容易召回无关章节
   - BM25更适合枚举型、字段名型的场景（如错误码、配置项）
3. ✅ **CrossEncoder重排序效果显著**（+0.07 Precision），能准确识别表格、列表等结构化内容

**技术决策**：采用 **Dense + Reranker** 方案，在保证高精确率的同时，延迟控制在2秒以内。

### 3. 典型Case分析

#### Case 1: 表格查询 ✅

**问题**：不同地图模式支持的操作方式

**挑战**：表格内容在Dense检索中排名较低（第9位）

**解决方案**：
1. 扩大候选池（4 → 20条）
2. CrossEncoder重排序识别表格相关性
3. 成功召回完整表格，准确回答

**结果**：准确率从 **0%** 提升到 **100%**

#### Case 2: 流程查询 ✅

**问题**：代码提交流程是怎样的？

**优化前**：召回了3/5个步骤，答案不完整

**优化后**：
- 通过层级切分保留完整步骤
- chunk_size=600避免步骤混淆
- 准确召回5/5个步骤

**结果**：完整性从 **60%** 提升到 **100%**

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 8GB+ RAM（用于模型加载）

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-repo/Nav_RAG.git
cd Nav_RAG

# 创建虚拟环境
conda create -n rag python=3.9
conda activate rag

# 安装依赖
pip install -r requirements.txt
pip install pdfplumber rank-bm25 jieba  # 额外依赖
```

### 配置环境变量

创建 `.env` 文件：

```bash
# DeepSeek API (用于LLM)
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# HuggingFace镜像（可选，加速模型下载）
HF_ENDPOINT=https://hf-mirror.com
```

### 构建索引

```bash
cd src
python build_index.py
```

**输出示例**：
```
✂️  开始自适应切分文档...
📄 PIS-2116_Location Based Service_A-V0.0.2.3.docx
  类型: nav_spec | 格式: .docx | 原始chunks: 725
  📐 使用层级切分 | chunk_size=1200
  ✅ 切分后: 764 chunks

✂️  总切分结果: 1050 chunks
✅ 向量库已保存到: ../data/index/nav_faiss
```

### 启动Web界面

```bash
cd ..
streamlit run app.py
```

浏览器打开 `http://localhost:8501`

---

## 🎛️ 功能说明

### Web界面控制面板

- **🔄 启用Query改写**：使用LLM将问题改写为检索关键词（提升召回率）
- **✨ 启用重排序**：使用CrossEncoder从候选中精排（提升精确率）
- **返回文档数**：调整最终返回的文档数量（3-10条）

### 推荐配置

| 查询类型 | Query改写 | 重排序 | 返回文档数 |
|---------|----------|--------|-----------|
| **简单查询**（定义类） | ❌ 关闭 | ✅ 开启 | 4 |
| **复杂查询**（流程、表格） | ✅ 开启 | ✅ 开启 | 6 |
| **探索性查询** | ✅ 开启 | ✅ 开启 | 8-10 |

---

## 📈 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 文档数量 | 17份 | DOCX + PDF |
| 总chunk数 | ~1050 | 自适应切分后 |
| 索引大小 | ~2MB | FAISS向量库 |
| 平均检索延迟 | 1.2s | Dense检索 |
| 平均生成延迟 | 2.5s | 含LLM流式输出 |
| Context Precision | 0.92 | Dense + Reranker |
| Context Recall | 0.95 | 基于RAGAS评估 |

---

## 🛠️ 项目结构

```
Nav_RAG/
├── app.py                      # Streamlit Web UI
├── .env                        # 环境配置
├── requirements.txt            # 依赖列表
│
├── data/
│   ├── documents/              # 原始文档（17份）
│   ├── index/                  # FAISS向量索引
│   │   └── nav_faiss/
│   └── nav_rag_eval_set_v1.jsonl  # 评估数据集
│
└── src/
    ├── loaders.py              # 文档加载（DOCX/PDF解析）
    ├── splitters.py            # 自适应切分策略
    ├── embeddings.py           # Embedding模型封装
    ├── reranker.py             # CrossEncoder重排序
    ├── build_index.py          # 构建索引
    ├── evaluate.py             # RAGAS评估
    └── ask.py                  # CLI问答脚本
```

---

## 🔧 遇到的挑战与解决方案

### 1. PDF标题识别困难

**问题**：PDF没有语义标签，仅靠字号识别标题准确率低

**解决方案**：
- 字体大小分析（统计最常见字号为正文）
- 多规则融合：字号 + 加粗 + 数字编号 + 关键词
- 启发式规则：标题长度限制、不以标点结尾

**结果**：识别准确率从 **40%** 提升到 **80%**

### 2. 表格内容召回困难

**问题**：表格被切分或在检索中排名靠后

**解决方案**：
- 增大chunk_size（800 → 1200）保留完整表格
- 扩大候选池（4 → 20条）
- CrossEncoder重排序识别表格相关性

**结果**：表格查询准确率从 **30%** 提升到 **95%**

### 3. 混合检索效果反而下降

**问题**：添加BM25后，精确率从0.85降到0.83

**分析**：
- 导航文档语义密集，关键词重复率高
- BM25适合枚举型/字段名型场景，不适合长语义文档
- Dense向量检索已经足够有效

**决策**：放弃混合检索，采用 Dense + Reranker 方案

**启示**：**技术选型要基于数据特点，而非盲目追求"先进"**

---

## 📚 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **框架** | LangChain 1.0 (LCEL) | RAG管道编排 |
| **UI** | Streamlit | Web可视化 |
| **向量库** | FAISS (CPU) | 轻量级、快速 |
| **Embedding** | BAAI/bge-small-zh-v1.5 | 中文优化，100MB |
| **Reranker** | BAAI/bge-reranker-base | CrossEncoder精排 |
| **LLM** | DeepSeek Chat | 文本生成与改写 |
| **文档解析** | python-docx, pdfplumber | 多格式支持 |
| **评估框架** | RAGAS | 自动化RAG评估 |

---

## 🎯 未来优化方向

- [ ] **增量更新**：新增文档时不重建全量索引
- [ ] **用户反馈**：点赞/踩功能，收集bad case
- [ ] **图表OCR**：提取PDF中的图片和图表（如有真实需求）
- [ ] **多文档对话**：支持跨文档的关联问答

---

## 📄 License

MIT License

---

## 🙏 致谢

- [LangChain](https://www.langchain.com/) - RAG框架
- [BAAI/bge-models](https://github.com/FlagOpen/FlagEmbedding) - Embedding与Reranker模型
- [DeepSeek](https://www.deepseek.com/) - LLM API
- [Streamlit](https://streamlit.io/) - Web UI框架

---

## 📧 联系方式

如有问题或建议，欢迎提Issue或联系作者。
