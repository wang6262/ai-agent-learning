# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Agent 从零开始学习项目。通过 12 个递进式 step 文件，从调用 LLM 到完整的多智能体协作系统。每个 step 文件自包含、可直接运行。使用阿里云 DashScope API（Qwen 模型，OpenAI 兼容接口）+ ChromaDB 向量数据库。

## Environment Setup

```bash
pip install openai python-dotenv requests chromadb duckduckgo_search beautifulsoup4 pymupdf
```

在项目根目录创建 `.env` 文件（需自行获取 API Key）：
```
DASHSCOPE_API_KEY="sk-xxx"
```

DashScope 获取地址：https://dashscope.aliyun.com

## Learning Path（按顺序运行）

| Step | 文件 | 学什么 | 运行 |
|------|------|--------|------|
| 01 | `step01_hello_qwen.py` | LLM 基础调用 | `python step01_hello_qwen.py` |
| 02 | `step02_tool_calling.py` | Function Calling | `python step02_tool_calling.py` |
| 03 | `step03_react_agent.py` | ReAct 循环 | `python step03_react_agent.py` |
| 04 | `step04_agent_with_memory.py` | 对话记忆 | `python step04_agent_with_memory.py` |
| 05 | `step05_robust_agent.py` | 错误处理/健壮性 | `python step05_robust_agent.py` |
| 06 | `step06_reflection_agent.py` | 反思/质量检查 | `python step06_reflection_agent.py` |
| 07 | `step07_complete_agent.py` | 完整 Agent 框架（含 Memory/ToolKit/Reflector） | `python step07_complete_agent.py` |
| 08 | `step08_rag_basic.py` | RAG 基础流水线 | `python step08_rag_basic.py` |
| 09 | `step09_rag_agent.py` | RAG 集成到 Agent | `python step09_rag_agent.py` |
| 10 | `step10_document_ingest.py` | 多格式文档摄入（PDF/MD/TXT→知识库） | `python step10_document_ingest.py` |
| 11 | `step11_web_search.py` | 联网搜索 Agent | `python step11_web_search.py` |
| 12 | `step12_multi_agent.py` | 多智能体协作（3 种模式） | `python step12_multi_agent.py` |

## 核心架构模式

### Agent 类（两个版本）

- **完整版**（`step07_complete_agent.py`）：`Agent` 类包含 `Memory`（对话记忆）、`ToolKit`（工具管理）、`Reflector`（反思/质量检查），通过 `add_tool()` 注册工具，`run()` 执行 ReAct 循环
- **简化版**（`step09/step11/step12`）：`Agent` 类 ~70 行，去掉 Memory/ToolKit/Reflector，聚焦演示特定概念。接口与完整版一致（`register_tool()` / `run()` / `reset()`），可互换

### LLM 客户端（OpenAI 兼容）

```python
from openai import OpenAI
llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# Chat: llm_client.chat.completions.create(model="qwen-plus", messages=..., tools=...)
# Embedding: llm_client.embeddings.create(model="text-embedding-v2", input=texts)
```

### 工具注册（OpenAI Function Calling 标准格式）

```python
agent.register_tool(
    "tool_name", func, "工具描述",
    {"param": {"type": "string", "description": "参数描述"}}
)
```

### Embedding 获取

直接用 `llm_client.embeddings.create(model="text-embedding-v2", input=texts)` 而非 `requests.post` 直调 REST API。代码更简洁，与 LLM 调用共享同一客户端。

### ChromaDB

- 存储路径：`chroma_db/` 目录（项目根目录下）
- 查看工具：`python inspect_chromadb.py`
- 配置：`PersistentClient` + `hnsw:space: cosine`
- 详细文档：`chromadb_guide.md`（14 章，零基础到企业级）

### Agent-as-Tool（多智能体核心技巧）

```python
func, schema = agent_as_tool(
    lambda: Agent(name="专家", system_prompt="..."),
    "ask_expert", "向专家咨询"
)
manager.register_tool("ask_expert", func, ...)
```

### 多智能体三种通信拓扑

| 模式 | 拓扑 | 适用场景 |
|------|------|----------|
| 顺序流水线 | A → B → C 链式 | 有先后顺序的任务 |
| 经理-工人 | Manager ←→ Workers 星型 | 多领域专家协作 |
| 辩论 | A ↔ B → Judge 对等+裁判 | 无标准答案的决策 |

## 代码风格（精简版）

项目定位是**零基础可学的进阶代码**，核心原则：

1. **分层注释**：基础逻辑大白话 + 进阶语法原理注解。简单代码通俗讲，进阶代码拆解讲
2. **文件头部必写**：文件名、基础功能、核心知识点、使用方法、进阶说明
3. **函数文档字符串**：基础功能 + 学习知识点 + 调用示例 + 同场景替代函数
4. **允许进阶写法**（需注释讲透）：推导式、三元表达式、装饰器、异常嵌套、算法逻辑
5. **禁止**：无注释的黑魔法代码、无意义嵌套简写、晦涩命名
6. **配置集中**：常量/参数统一放代码顶部
7. **每次修改后**：在 `log.md` 末尾追加变更日志

完整代码规范见 `CLAUDE11111.md`。

## 其他文件

- `chromadb_guide.md`：ChromaDB 完整学习文档
- `inspect_chromadb.py`：ChromaDB 内容查看工具（`python inspect_chromadb.py [--export]`）
- `data_loader.py`：ChromaDB 通用资源加载器（本地文件 + HTTP URL）
- `test.py`：ChromaDB 多模态实验（文本 + 图片 Embedding）
- `main.py`：PyCharm 默认生成的模板，无实际用途