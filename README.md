# RAG Project — 多策略检索增强生成框架

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🤖-LangChain-blue)](https://www.langchain.com/)
[![RAGAS](https://img.shields.io/badge/📊-RAGAS-green)](https://docs.ragas.io/)

一个基于 LangChain 的多策略 **RAG（Retrieval-Augmented Generation）** 实验框架，集成了 **Naive RAG**、**RAG-Fusion** 和 **HyDE** 三种检索增强生成策略，并提供基于 **RAGAS** 的自动化评估能力。

## 特性

- **三种 RAG 策略**
  - **Naive RAG** — 标准检索 + 生成流程，支持多轮对话历史
  - **RAG-Fusion** — 查询扩展 + 互惠排名融合（Reciprocal Rank Fusion），提升召回率
  - **HyDE** — 先由 LLM 生成假设性文档，再用其检索真实文档，改善语义匹配

- **多领域知识库** — 内置 FiQA（金融）、Amnesty QA（人权）等领域数据集，支持热切换

- **自动化评估** — 集成 RAGAS 框架，评估指标包括：
  - `faithfulness`（忠实度）
  - `answer_relevancy`（答案相关性）
  - `answer_correctness`（答案正确性，需标准答案）
  - `context_recall`（上下文召回率，需标准答案）

- **多文档格式支持** — 支持加载 PDF、TXT、Markdown、DOCX 及网页内容

- **Prompt 注入防护** — 内置输入验证和注入检测

- **持久化向量库** — 基于 Chroma DB 的向量存储，支持增量构建和重复使用

## 架构

```
用户输入 → [输入验证] → [检索策略] → [向量检索] → [LLM 生成] → [输出]
                        ├─ Naive: 直接检索
                        ├─ Fusion: 查询扩展 → 多路检索 → RRF 融合
                        └─ HyDE: 假设性文档生成 → 语义检索
```

```
├── config.py           # 配置文件（模型路径、API 密钥、分块参数）
├── document_loader.py  # 文档加载（PDF/TXT/MD/DOCX/URL/文本）
├── indexing.py         # 向量库构建与持久化（Chroma + HuggingFace Embeddings）
├── llm_setup.py        # LLM 初始化（DeepSeek API + Ollama）
├── prompts.py          # 系统提示词模板（含安全规则）
├── chain.py            # Naive RAG 链（带对话历史）
├── RAG_naive.py        # Naive RAG 策略实现
├── RAG_fusion.py       # RAG-Fusion 策略实现
├── RAG_HyDE.py         # HyDE 策略实现
├── evaluation.py       # RAGAS 评估框架（多数据集对比）
├── history.py          # 对话历史管理（限长 FIFO）
├── validation.py       # 输入验证与清洗（Prompt 注入检测）
├── download_datasets.py # 评估数据集下载工具
└── main.py             # 主程序入口（交互式 CLI）
```

## 快速开始

### 环境要求

- Python 3.11+
- 一个本地 embedding 模型（默认使用 [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)）
- DeepSeek API Key（或其他 OpenAI 兼容 API）
- （可选）Ollama + Qwen2.5:7b，用于离线 RAGAS 评估

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/RAG_project.git
cd RAG_project

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

编辑 `config.py` 或设置环境变量：

```bash
# DeepSeek API（必须）
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# 智谱 GLM API（可选，用于 RAGAS 评估）
export ZHIPU_API_KEY="your-zhipu-api-key"

# 本地 Embedding 模型路径（在 config.py 中修改）
# EMBEDDING_MODEL_PATH = "path/to/your/multilingual-e5-large"
```

### 运行

```bash
python main.py
```

启动后，交互式 CLI 支持以下命令：

| 命令 | 功能 |
|------|------|
| `模式 naive/fusion/hyde` | 切换 RAG 策略 |
| `评估` / `eval` | 运行 RAGAS 自动评估 |
| `exit` / `quit` | 退出程序 |

### 下载评估数据集

```bash
python download_datasets.py
```

## RAG 策略说明

### Naive RAG
标准流程：用户问题 → 向量检索 → 拼接上下文 → LLM 生成回答。支持多轮对话历史（保留最近 5 轮）。

### RAG-Fusion
1. 将用户问题扩展为 3-5 个不同角度的查询
2. 分别检索后使用 **Reciprocal Rank Fusion（RRF）** 融合结果
3. 用融合后的上下文生成最终答案

### HyDE（Hypothetical Document Embeddings）
1. 让 LLM 根据问题先生成一段"假设性答案文档"
2. 用该假设文档进行向量检索（而非直接用问题）
3. 基于检索到的真实文档生成最终回答

## 评估

框架内置基于 [RAGAS](https://docs.ragas.io/) 的评估系统：

- **单数据集评估** — 对特定数据集运行完整评估流程
- **多数据集对比** — 在不同领域数据集上运行评估，生成对比报告
- **自定义评估** — 支持加载 CSV/JSON/JSONL 格式的自定义评估数据集

## 依赖

- `langchain` / `langchain-core` / `langchain-community`
- `langchain-chroma` / `langchain-huggingface`
- `langchain-openai` / `langchain-ollama`
- `ragas`
- `datasets`
- `chromadb`
- `sentence-transformers`

## 许可

[MIT](LICENSE)
