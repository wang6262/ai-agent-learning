"""
Step 9: RAG 集成到 Agent — 知识库成为工具

学习目标:
  1. 理解 RAG 作为 Agent 工具的架构优势
  2. Agent 如何自主判断何时需要查知识库
  3. 知识库 + 其他工具（时间、天气、计算）的协同使用
  4. 区分 Agent 的两种知识来源: 模型内置知识 vs 外部知识库

为什么把 RAG 封装为工具而不是直接拼到 System Prompt？
  - 不是每个问题都需要查知识库 → 节省 token
  - Agent 可以同时用知识库 + 其他工具（如"查最新产品定价 AND 获取当前时间"）
  - 用户可见 Agent 是否调用了知识库，回答更可信
  - 和 step07 的 Agent 框架无缝集成

运行: python step09_rag_agent.py
前置: pip install chromadb requests
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

import chromadb
import requests

# 修复 Windows GBK 终端无法输出 emoji
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 自动加载 .env
load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("❌ 未找到 API Key！请检查 .env 文件中的 DASHSCOPE_API_KEY")
    sys.exit(1)

llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3

# ============================================================
# 1. 知识库文档 — 虚构的公司信息（LLM 不知道）
# ============================================================

KNOWLEDGE_DOCS = [
    {
        "title": "星辰科技公司简介",
        "content": """
星辰科技（StarTech）成立于 2024 年 3 月，总部位于杭州未来科技城。
公司专注于 AI 智能办公解决方案。

核心产品线:
1. StarDocs — 智能文档协作平台
2. StarMeet — AI 会议纪要助手
3. StarFlow — 自动化工作流引擎

管理团队:
- CEO: 张明远（前阿里巴巴高级总监）
- CTO: 李思涵（前腾讯 AI Lab 研究员，NLP 博士）
- COO: 王雨婷（前字节跳动产品副总裁）

公司规模: 约 120 人，研发团队占比 65%。
""",
    },
    {
        "title": "产品定价（2025年更新）",
        "content": """
StarDocs 定价:
- 个人版: ￥29/月，10GB 存储，基础 AI 功能
- 团队版: ￥99/人/月，100GB 存储，高级 AI + 权限管理
- 企业版: ￥299/人/月，1TB 存储，私有部署 + 定制 AI

StarMeet 定价:
- 基础版: 免费（每月 20 小时）
- 专业版: ￥49/月（不限时）
- 企业版: ￥99/人/月（私有化部署）

StarFlow 定价:
- 标准版: ￥199/月（10 个流程）
- 企业版: ￥999/月（无限流程 + API 接入）
""",
    },
    {
        "title": "融资历史",
        "content": """
星辰科技融资历程:
- 2024 年 3 月: 天使轮 ￥2000 万，红杉中国种子基金
- 2024 年 9 月: A 轮 ￥1.5 亿，高瓴资本领投，估值 ￥8 亿
- 2025 年 6 月: B 轮 ￥5 亿，软银愿景基金领投，估值 ￥50 亿

总融资额: ￥6.7 亿
预计 C 轮: 2026 年 Q1，目标估值 ￥150 亿
""",
    },
    {
        "title": "技术架构",
        "content": """
星辰科技技术栈:
- 后端: Go + Rust，微服务架构，Kubernetes 部署
- 前端: React 18 + TypeScript + WebAssembly (协作引擎)
- AI: 自研 NLP 模型（基于 Qwen 微调） + 向量数据库（Milvus）
- 数据: PostgreSQL + ClickHouse + Redis + Kafka
- 云: 阿里云为主，AWS 为辅（海外业务）

系统可用性: 99.95%，P99 延迟 < 200ms。
日活跃用户: 80 万+（2025 年 6 月数据）。
""",
    },
]


# ============================================================
# 2. 文档分块
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """递归字符分割: 段落 -> 句子 -> 定长"""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para:
                chunks.extend(chunk_text(para, chunk_size, overlap))
        return chunks

    import re
    sentences = re.split(r'(?<=[。！？!?\n])', text)
    if len(sentences) > 1:
        chunks = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) <= chunk_size:
                current += sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================================================
# 3. Embedding API
# ============================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """调用 DashScope text-embedding-v2"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    all_embeddings = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i + 25]
        payload = {"model": EMBEDDING_MODEL, "input": {"texts": batch}}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embeddings = sorted(data["output"]["embeddings"], key=lambda x: x["text_index"])
        all_embeddings.extend([e["embedding"] for e in embeddings])

    return all_embeddings


# ============================================================
# 4. 向量存储 — ChromaDB
# ============================================================

class KnowledgeBase:
    """知识库: 管理文档的向量存储和检索"""

    def __init__(self):
        persist_dir = str(Path(__file__).parent / "chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 每次启动重新创建，确保数据一致
        try:
            self.client.delete_collection("agent_kb")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="agent_kb",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, docs: list[dict]):
        """摄入文档到知识库"""
        all_chunks = []
        all_metadatas = []
        for doc in docs:
            for chunk in chunk_text(doc["content"]):
                all_chunks.append(chunk)
                all_metadatas.append({"title": doc.get("title", ""), "chunk_size": len(chunk)})

        if not all_chunks:
            return

        print(f"  📥 知识库: {len(docs)} 篇文档 → {len(all_chunks)} 个块，向量化中...")
        embeddings = get_embeddings(all_chunks)
        ids = [f"kb_{i}" for i in range(len(all_chunks))]
        self.collection.add(ids=ids, documents=all_chunks, embeddings=embeddings, metadatas=all_metadatas)
        print(f"  ✅ 知识库就绪: {self.collection.count()} 条向量记录")

    def search(self, query: str, top_k: int = TOP_K) -> str:
        """检索并返回格式化的结果文本"""
        query_emb = get_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return "知识库中未找到相关信息。"

        lines = []
        for i in range(len(results["ids"][0])):
            title = results["metadatas"][0][i].get("title", "未知")
            sim = 1 - results["distances"][0][i]
            content = results["documents"][0][i][:500]
            lines.append(f"[来源: {title}] (相关度: {sim:.1%})\n{content}")

        return "\n\n---\n\n".join(lines)


# ============================================================
# 5. Agent 类 — 知识库作为一种工具
# ============================================================

class Agent:
    """
    Agent = LLM + 工具集（含知识库检索） + ReAct 循环 + 记忆

    和 step07 的架构一致，但这里重点展示:
    - search_knowledge_base 作为工具被注册
    - Agent 自动判断"需要查知识库"还是"我知道"
    - 知识库可以和其它工具组合使用
    """

    def __init__(self, name: str, system_prompt: str, kb: KnowledgeBase = None):
        self.name = name
        self.system_prompt = system_prompt
        self.kb = kb
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self.messages: list[dict] = []
        self.max_turns = 10

    def register_tool(self, name: str, func: Callable, description: str, parameters: dict = None):
        """注册工具"""
        self._functions[name] = func
        self._schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters or {},
                    "required": list(parameters.keys()) if parameters else [],
                },
            },
        })

    def _build_context(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _execute_tool(self, name: str, args: dict) -> str:
        func = self._functions.get(name)
        if not func:
            return f"未知工具: {name}"
        try:
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"

    def run(self, user_input: str, verbose: bool = True) -> str:
        """执行 Agent 的 ReAct 循环"""
        self.messages.append({"role": "user", "content": user_input})

        if verbose:
            print(f"\n{'─'*40}")
            print(f"🤖 {self.name} 正在处理...")

        for turn in range(1, self.max_turns + 1):
            context = self._build_context()
            tools = self._schemas if self._schemas else None

            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=context,
                tools=tools,
            )
            msg = response.choices[0].message

            # 直接回复 → 完成
            if msg.content and not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            # 工具调用
            if msg.tool_calls:
                # 记录 assistant 的 tool_calls 消息
                tool_call_records = []
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        arg_str = json.dumps(args, ensure_ascii=False)
                        print(f"  🔧 [{turn}] {name}({arg_str})")

                    result = self._execute_tool(name, args)

                    if verbose:
                        result_preview = result[:150].replace("\n", " ")
                        print(f"  📋 [{turn}] {result_preview}...")

                    tool_call_records.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": name, "arguments": tc.function.arguments},
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # 添加 assistant 消息（含 tool_calls 标记）
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_call_records,
                })
                continue

        # 超时
        self.messages.append({"role": "user", "content": "请基于已有信息给出最终回答。"})
        response = llm_client.chat.completions.create(model=LLM_MODEL, messages=self._build_context())
        final = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self):
        self.messages = []


# ============================================================
# 6. 构建 Agent
# ============================================================

def build_agent(kb: KnowledgeBase) -> Agent:
    """构建带有知识库工具的 Agent"""

    # 普通工具函数
    def get_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

    def get_weather(city: str) -> str:
        data = {
            "北京": {"temp": 25, "condition": "晴"},
            "上海": {"temp": 28, "condition": "多云"},
            "杭州": {"temp": 26, "condition": "小雨"},
        }
        w = data.get(city, {"temp": 22, "condition": "未知"})
        return json.dumps(w, ensure_ascii=False)

    def calculate(expression: str) -> str:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误: 包含不允许的字符"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"错误: {e}"

    # 关键: 知识库搜索作为工具
    def search_knowledge_base(query: str) -> str:
        """搜索内部知识库"""
        return kb.search(query)

    agent = Agent(
        name="智能助手 (带知识库)",
        system_prompt="""你是一个拥有内部知识库的智能助手。

你的能力:
- 查询内部知识库（公司信息、产品定价、技术文档等）—— 这是你最重要的能力
- 获取实时时间、天气
- 执行数学计算
- 一般知识可以直接回答

工作原则:
- 当用户询问公司、产品、定价、技术、融资等内部信息时，必须优先使用 search_knowledge_base 工具
- 不要凭记忆编造公司相关信息，一切以知识库返回为准
- 知识库未覆盖的信息，明确告知用户
- 可以和其它工具组合使用（如"StarDocs 团队版每年多少钱？"需要查知识库+计算）""",
        kb=kb,
    )

    # 注册工具 — 知识库搜索放在首位，让 LLM 优先考虑
    agent.register_tool(
        "search_knowledge_base",
        search_knowledge_base,
        "搜索内部知识库。当用户询问公司、产品、定价、融资、技术等内部信息时必须调用此工具。"
        "参数: query（搜索查询，使用关键词，如'StarDocs定价'、'融资历史'）",
        {"query": {"type": "string", "description": "搜索查询关键词"}},
    )
    agent.register_tool("get_time", get_time, "获取当前日期和时间")
    agent.register_tool(
        "get_weather", get_weather, "获取指定城市天气",
        {"city": {"type": "string", "description": "城市名称"}},
    )
    agent.register_tool(
        "calculate", calculate, "执行数学计算",
        {"expression": {"type": "string", "description": "数学表达式"}},
    )

    return agent


# ============================================================
# 7. 交互式运行
# ============================================================

def main():
    print("=" * 60)
    print("🤖 RAG + Agent — 知识库驱动的智能助手")
    print("=" * 60)
    print(f"LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL}")
    print(f"工具: search_knowledge_base, get_time, get_weather, calculate")
    print()

    # 初始化知识库
    print("🏗️ 初始化知识库...")
    kb = KnowledgeBase()
    kb.ingest(KNOWLEDGE_DOCS)
    print()

    # 构建 Agent
    agent = build_agent(kb)
    print("✅ Agent 就绪！")
    print()
    print("试试这些问题（注意观察 Agent 何时调用知识库）:")
    print("  'StarDocs 团队版多少钱？'")
    print("  '星辰科技融资了几轮？总融资额多少？'")
    print("  'StarMeet 专业版用一年要花多少？' — 需要 知识库 + 计算")
    print("  '星辰科技 CTO 是谁？'")
    print("  '现在几点了？' — 不需要知识库")
    print("输入 'quit' 退出, 'reset' 重置对话, 'sources' 查看知识库信息\n")

    while True:
        try:
            user_input = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄 对话已重置")
            continue
        if user_input.lower() == "sources":
            print(f"\n📚 知识库信息:")
            print(f"   文档数: {len(KNOWLEDGE_DOCS)}")
            for doc in KNOWLEDGE_DOCS:
                print(f"   - {doc['title']}")
            print()
            continue

        answer = agent.run(user_input, verbose=True)
        print(f"\n🤖 {answer}")
        print("\n" + "-" * 40)


# ============================================================
# 8. 概念总结
# ============================================================

def print_summary():
    print("""
╔══════════════════════════════════════════════════════════╗
║           RAG + Agent 架构核心要点                        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. RAG 作为工具 vs 拼入 System Prompt                    ║
║     工具方式: Agent 自主决定何时检索，节省 token          ║
║     拼入方式: 每次请求都带上下文，浪费且不灵活            ║
║                                                          ║
║  2. 知识库是 Agent 的"外挂记忆"                          ║
║     模型内置记忆: 通用知识（截止训练日）                  ║
║     知识库记忆: 私有、实时、可更新                        ║
║                                                          ║
║  3. 多工具协同                                            ║
║     知识库 + 计算器 = 定价查询 + 费用计算                 ║
║     知识库 + 时间 = 查看截止日期是否到期                  ║
║                                                          ║
║  4. 向量数据库的作用                                      ║
║     传统搜索: 关键词匹配（"团队版"找不到"Team 版"）       ║
║     语义搜索: 向量相似度（自动匹配近义表达）              ║
║                                                          ║
║  完整学习路径回顾:                                        ║
║     step01-02 → LLM 调用 + 工具                          ║
║     step03    → ReAct 循环                               ║
║     step04    → 对话记忆                                 ║
║     step05    → 健壮性 + 错误处理                        ║
║     step06    → 反思 + 质量检查                          ║
║     step07    → 完整 Agent 框架                          ║
║     step08    → RAG 基础（向量库 + 检索）                ║
║     step09    → RAG + Agent 集成 ← 你在这里              ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
    print_summary()
