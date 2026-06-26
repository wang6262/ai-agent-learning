# ==============================================
# 文件名：step13_long_term_memory.py
# 基础功能：为 Agent 打造长期记忆系统，跨会话记住用户信息、偏好和历史对话
# 核心学习知识点：
#   1. 三种记忆类型（认知科学来源）：画像记忆 / 事实记忆 / 片段记忆
#   2. 混合存储架构：SQLite（结构化精确查询）+ ChromaDB（语义模糊搜索）
#   3. 记忆生命周期：提取 → 存储 → 检索 → 整合 → 遗忘
#   4. LLM 自动记忆提取：用 prompt 让 LLM 判断"什么值得记住"
#   5. 记忆冲突检测：同一事实不同版本时的整合策略
# 适用场景：个性化助手、长期陪伴型 Agent、需要记住用户偏好的客服系统
# 使用方法：
#   1. python step13_long_term_memory.py
#   2. 按提示输入个人信息，Agent 会自动记住
#   3. 关闭程序重新运行，验证记忆是否持久化
# 进阶说明：
#   - 生产环境可换 PostgreSQL + pgvector 替代 SQLite + ChromaDB
#   - 记忆压缩：旧记忆可定期交给 LLM 总结成摘要（减少存储量）
#   - 记忆图谱：用 Neo4j 等图数据库存储记忆间的关系
# 常用配套函数：
#   1. sqlite3.connect()：Python 内置 SQLite 连接
#   2. chromadb.PersistentClient.get_or_create_collection()：持久化向量集合
#   3. llm_client.chat.completions.create()：用 LLM 做记忆提取和整合
# ==============================================
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

import chromadb

# ---- Windows 终端 UTF-8 兼容 ----
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置区
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("❌ 未找到 API Key！请检查 .env 文件")
    sys.exit(1)

llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = "qwen-plus"

# 持久化路径
MEMORY_DB_PATH = Path(__file__).parent / "memory" / "long_term.db"
CHROMA_PATH = Path(__file__).parent / "memory" / "chroma"

# 确保目录存在
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 第 1 部分：Agent 基类（复用 step12 简化版，去掉了工具注册）
# ============================================================

class Agent:
    """
    【基础功能】带对话历史的智能体
    注意：这里的 self.messages 是**会话内短期记忆**（重启丢失）。
    长期记忆由 LongTermMemory 类实现，二者互补。
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self.messages: list[dict] = []
        self.max_turns = 10

    def register_tool(self, name, func, description, parameters=None):
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
        return self

    def _build_context(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _execute_tool(self, name, args):
        func = self._functions.get(name)
        if not func:
            return f"错误：未找到工具 '{name}'"
        try:
            return str(func(**args))
        except Exception as e:
            return f"工具执行失败：{e}"

    def run(self, user_input, verbose=True):
        self.messages.append({"role": "user", "content": user_input})

        if verbose:
            print(f"\n{'─'*35}")
            print(f"🤖 {self.name} 思考中...")

        for turn in range(1, self.max_turns + 1):
            context = self._build_context()
            tools = self._schemas if self._schemas else None

            response = llm_client.chat.completions.create(
                model=LLM_MODEL, messages=context, tools=tools
            )
            msg = response.choices[0].message

            if msg.content and not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            if msg.tool_calls:
                tool_call_records = []
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  🔧 [{turn}] {name}({json.dumps(args, ensure_ascii=False)})")

                    result = self._execute_tool(name, args)

                    if verbose:
                        preview = str(result)[:200].replace("\n", " ")
                        print(f"  📋 [{turn}] {preview}")

                    tool_call_records.append({
                        "id": tc.id, "type": "function",
                        "function": {"name": name, "arguments": tc.function.arguments},
                    })
                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": str(result),
                    })

                self.messages.append({
                    "role": "assistant", "content": msg.content,
                    "tool_calls": tool_call_records,
                })
                continue

        self.messages.append({"role": "user", "content": "请基于已有信息给出最终回答"})
        response = llm_client.chat.completions.create(
            model=LLM_MODEL, messages=self._build_context()
        )
        final = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self):
        self.messages = []


# ============================================================
# 第 2 部分：长期记忆系统（核心）
# ============================================================

class LongTermMemory:
    """
    【基础功能】让 Agent 拥有跨会话的长期记忆

    【学习知识点】
        1. 三种记忆类型的认知科学来源：
           - 画像记忆(Profile)：你是谁 — 姓名、职业、偏好等稳定信息
           - 事实记忆(Facts)：关于你的事 — "你正在学Rust"、"上次聊过XX"
           - 片段记忆(Episodes)：过去的对话 — "2025-01-15你问过如何部署"
        2. 混合存储架构：
           - SQLite 负责结构化数据（精确查询："用户叫什么名字？"）
           - ChromaDB 负责语义数据（模糊搜索："用户喜欢什么编辑器？"）
        3. 记忆生命周期：提取(Extract) → 存储(Store) → 检索(Retrieve) → 整合(Consolidate) → 遗忘(Forget)

    【进阶说明】
        为什么不全用 ChromaDB？
        - ChromaDB 适合"语义相似"搜索，但做不了精确 key-value 查询
        - SQLite 适合"取用户的editor字段"这种精确查询
        - 混合架构 = 精确查询用 SQLite + 模糊搜索用 ChromaDB

    关键区分：
        - step08~10 的知识库 = 外部文档（公司手册、产品说明）
        - step13 的长期记忆 = 关于用户的个人信息（偏好、历史、身份）

    调用示例：
        ltm = LongTermMemory()
        ltm.remember("用户叫小明，后端开发", category="identity")
        results = ltm.recall("用户的职业是什么？")
        ltm.update_profile("editor", "VS Code")
    """

    def __init__(self):
        # ---- SQLite 初始化 ----
        self.sqlite_path = str(MEMORY_DB_PATH)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row  # 结果可以按列名访问
        self._init_sqlite()

        # ---- ChromaDB 初始化 ----
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._init_chroma()

    # ========== SQLite 建表 ==========

    def _init_sqlite(self):
        """
        【基础功能】创建两张核心表：profile（用户画像）+ facts（事实记忆）
        【学习知识点】
            CREATE TABLE IF NOT EXISTS：只在表不存在时创建（幂等操作，重复运行不报错）
            UNIQUE 约束：profile 表中每个 key 只有一条记录
        """
        cur = self._conn.cursor()

        # 用户画像表：key-value 结构，存储稳定信息
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,       -- 属性名，如 "name"、"editor"
                value TEXT NOT NULL,            -- 属性值，如 "小明"、"VS Code"
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 事实记忆表：存储离散的事实片段
        cur.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,           -- 事实内容
                category TEXT DEFAULT 'general', -- 分类：identity/preference/plan/knowledge
                chroma_id TEXT,                  -- 对应 ChromaDB 中的向量 ID（关联查询用）
                importance INTEGER DEFAULT 5,    -- 重要性 1-10（越高越不容易被遗忘）
                created_at TEXT DEFAULT (datetime('now')),
                accessed_at TEXT DEFAULT (datetime('now'))
            )
        """)

        self._conn.commit()

    # ========== ChromaDB 建集合 ==========

    def _init_chroma(self):
        """
        【基础功能】创建两个向量集合：事实向量 + 片段记忆
        【学习知识点】
            get_or_create_collection vs create_collection：
            - create_collection 重复创建会报错
            - get_or_create_collection 不管重复多少次都安全（生产环境推荐）
        """
        # 事实记忆向量集合（与 SQLite facts 表配合使用）
        self.fact_collection = self.chroma_client.get_or_create_collection(
            name="fact_memory",
            metadata={"hnsw:space": "cosine"},
        )
        # 对话片段集合（存储对话摘要，用于回忆"上次聊过什么"）
        self.episode_collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"},
        )

    # ========== 记忆存储 ==========

    def remember(self, fact: str, category: str = "general", importance: int = 5) -> str:
        """
        【基础功能】记住一个事实，同时存入 SQLite + ChromaDB
        【学习知识点】
            1. 双写策略：SQLite 存结构化信息（精确查询）+ ChromaDB 存向量（语义搜索）
            2. 用当前时间戳作为 ChromaDB 的 ID 前缀，保证唯一性
            3. importance 字段：记忆优先级，高优先级的记忆更不容易被遗忘

        参数：
            fact: 事实内容，如 "用户喜欢 VS Code"
            category: 分类 — identity(身份) / preference(偏好) / plan(计划) / knowledge(知识)
            importance: 重要性 1-10

        调用示例：
            ltm.remember("用户叫小明，后端开发", category="identity", importance=8)
            ltm.remember("用户正在学 Rust", category="plan", importance=7)
        """
        # 步骤1：写入 SQLite（结构化存储）
        cur = self._conn.cursor()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        chroma_id = f"fact_{timestamp}"

        cur.execute(
            "INSERT INTO facts (content, category, chroma_id, importance) VALUES (?, ?, ?, ?)",
            (fact, category, chroma_id, importance),
        )
        self._conn.commit()

        # 步骤2：写入 ChromaDB（向量存储，支持语义搜索）
        try:
            # 用 DashScope Embedding 生成向量
            embedding_resp = llm_client.embeddings.create(
                model="text-embedding-v2", input=[fact]
            )
            embedding = embedding_resp.data[0].embedding

            self.fact_collection.add(
                ids=[chroma_id],
                documents=[fact],
                embeddings=[embedding],
                metadatas=[{"category": category, "importance": importance}],
            )
        except Exception as e:
            # ChromaDB 写入失败不影响 SQLite（降级容错）
            print(f"  ⚠️ ChromaDB 写入警告: {e}")

        return f"✅ 已记住: [{category}] {fact}"

    # ========== 用户画像操作 ==========

    def update_profile(self, key: str, value: str) -> str:
        """
        【基础功能】更新用户画像中的某个属性
        【学习知识点】
            INSERT ... ON CONFLICT ... DO UPDATE（UPSERT 操作）：
            - 如果 key 不存在 → 插入新行
            - 如果 key 已存在 → 更新 value + updated_at
            - 这是 SQLite 3.24+ 的语法，比先查后写更优雅
        """
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO profile (key, value, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               updated_at = datetime('now')""",
            (key, value),
        )
        self._conn.commit()
        return f"✅ 已更新画像: {key} = {value}"

    def get_profile(self) -> dict:
        """获取完整的用户画像"""
        cur = self._conn.cursor()
        cur.execute("SELECT key, value, updated_at FROM profile ORDER BY key")
        rows = cur.fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ========== 记忆检索 ==========

    def recall(self, query: str, top_k: int = 5) -> str:
        """
        【基础功能】检索相关记忆：先语义搜索 ChromaDB，再精确查询 SQLite
        【学习知识点】
            1. 混合检索策略：
               - ChromaDB 做第一轮"海选"（语义相似度）
               - SQLite 做补充查询（精确匹配）
            2. 返回结果的排序逻辑：优先返回语义相关度高 + 重要性高的记忆

        调用示例：
            results = ltm.recall("用户喜欢什么编辑器？")
            print(results)
        """
        results = []

        # 步骤1：ChromaDB 语义搜索
        try:
            query_emb = llm_client.embeddings.create(
                model="text-embedding-v2", input=[query]
            )
            embedding = query_emb.data[0].embedding

            chroma_results = self.fact_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            if chroma_results["ids"] and chroma_results["ids"][0]:
                for i, doc_id in enumerate(chroma_results["ids"][0]):
                    distance = chroma_results["distances"][0][i]
                    similarity = round(1 - distance, 3)
                    content = chroma_results["documents"][0][i]
                    meta = chroma_results["metadatas"][0][i]
                    # 只返回相似度 > 0.3 的结果（过滤噪音）
                    if similarity > 0.3:
                        results.append({
                            "content": content,
                            "category": meta.get("category", "general"),
                            "similarity": similarity,
                            "source": "语义匹配",
                        })
        except Exception as e:
            results.append({"content": f"语义搜索异常: {e}", "category": "error", "similarity": 0, "source": "error"})

        # 步骤2：SQLite 精确补充（查用户画像）
        cur = self._conn.cursor()
        cur.execute("SELECT key, value FROM profile ORDER BY key")
        profile_rows = cur.fetchall()
        if profile_rows:
            profile_text = " | ".join([f"{r['key']}: {r['value']}" for r in profile_rows])
            results.insert(0, {
                "content": f"[用户画像] {profile_text}",
                "category": "profile",
                "similarity": 1.0,
                "source": "精确查询",
            })

        if not results:
            return "没有找到相关记忆。"

        # 格式化输出
        lines = ["📚 相关记忆："]
        for i, r in enumerate(results):
            lines.append(
                f"  [{i+1}] [{r['category']}] ({r['source']}, 相关度: {r['similarity']:.2f})\n"
                f"      {r['content']}"
            )
        return "\n".join(lines)

    # ========== 对话片段记忆 ==========

    def remember_episode(self, user_input: str, agent_response: str) -> str:
        """
        【基础功能】记录一次对话片段（存入 episodic_memory）
        【学习知识点】
            对话摘要存储 vs 原文存储：
            - 原文存储：信息完整但占用大、噪音多
            - 摘要存储：信息精炼但可能丢失细节
            本实现采用"原文+元数据"折中方案
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        episode_id = f"ep_{timestamp}"

        # 用 LLM 生成一句话摘要（比存原文更精炼）
        try:
            summary_resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"用一句话（不超过30字）总结这段对话的核心内容：\n用户: {user_input}\n助手: {agent_response[:200]}",
                }],
                max_tokens=60,
            )
            summary = summary_resp.choices[0].message.content.strip()
        except Exception:
            summary = user_input[:50]

        # 存入 ChromaDB
        episode_text = f"[{datetime.now().strftime('%m-%d %H:%M')}] 用户: {user_input} | 摘要: {summary}"

        try:
            embedding_resp = llm_client.embeddings.create(
                model="text-embedding-v2", input=[episode_text]
            )
            embedding = embedding_resp.data[0].embedding

            self.episode_collection.add(
                ids=[episode_id],
                documents=[episode_text],
                embeddings=[embedding],
                metadatas=[{
                    "user_input": user_input[:200],
                    "summary": summary,
                    "timestamp": datetime.now().isoformat(),
                }],
            )
            return f"✅ 已记录对话片段: {summary}"
        except Exception as e:
            return f"⚠️ 片段记录失败: {e}"

    def recall_episodes(self, query: str, top_k: int = 3) -> list[dict]:
        """搜索历史对话片段"""
        try:
            query_emb = llm_client.embeddings.create(
                model="text-embedding-v2", input=[query]
            )
            embedding = query_emb.data[0].embedding

            results = self.episode_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            episodes = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    similarity = round(1 - results["distances"][0][i], 3)
                    if similarity > 0.3:
                        episodes.append({
                            "content": results["documents"][0][i],
                            "summary": results["metadatas"][0][i].get("summary", ""),
                            "similarity": similarity,
                        })
            return episodes
        except Exception:
            return []

    # ========== 记忆遗忘 ==========

    def forget(self, search_query: str) -> str:
        """
        【基础功能】遗忘匹配的记忆（删除 SQLite 记录 + 标记 ChromaDB 低重要性）
        【学习知识点】
            真正的"遗忘"有三种策略：
            1. 硬删除：直接从数据库删除（本实现）
            2. 软遗忘：降低重要性，检索时被过滤
            3. 衰减遗忘：根据时间自动降低权重（进阶实现）
        """
        # 从 SQLite 搜索并删除
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, content FROM facts WHERE content LIKE ?",
            (f"%{search_query}%",),
        )
        rows = cur.fetchall()

        if not rows:
            return f"未找到匹配 '{search_query}' 的记忆"

        deleted = []
        for row in rows:
            cur.execute("DELETE FROM facts WHERE id = ?", (row["id"],))
            deleted.append(row["content"])

        self._conn.commit()
        return f"✅ 已遗忘 {len(deleted)} 条记忆:\n" + "\n".join(f"  - {d}" for d in deleted)

    # ========== 记忆整合 ==========

    def consolidate(self) -> str:
        """
        【基础功能】整合记忆：检测重复/矛盾的事实，让 LLM 判断如何合并
        【学习知识点】
            记忆整合(Memory Consolidation)的工程意义：
            - 用户可能在不同时间说同一件事，累积了多个版本
            - 整合 = 用 LLM 把同一主题的多条记忆合并为最准确的一条
            - 这是从"记住一切"到"记住重要的"的关键一步
        """
        cur = self._conn.cursor()
        cur.execute("SELECT id, content, category FROM facts ORDER BY created_at")
        rows = cur.fetchall()

        if len(rows) < 2:
            return "记忆数量不足（需至少2条），无需整合"

        # 列出所有记忆让 LLM 判断
        facts_text = "\n".join([f"[{r['id']}] ({r['category']}) {r['content']}" for r in rows])

        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""分析以下记忆列表，找出重复或矛盾的记忆。

对每组重复/矛盾的记忆，给出整合建议。按 JSON 格式回复：
{{"actions": [
    {{"action": "merge", "ids": [1, 3], "new_content": "合并后的记忆"}},
    {{"action": "delete", "ids": [5], "reason": "已过时的信息"}}
]}}

如果没有需要整合的记忆，返回 {{"actions": []}}

记忆列表:
{facts_text}""",
                }],
                temperature=0.2,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content
            # 提取 JSON
            if "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            actions = json.loads(result_text).get("actions", [])

            if not actions:
                return "✅ 记忆检查完毕，无需整合"

            # 执行整合操作
            report = []
            for action in actions:
                if action["action"] == "merge":
                    for fid in action["ids"]:
                        cur.execute("DELETE FROM facts WHERE id = ?", (fid,))
                    self.remember(action["new_content"], category="general")
                    report.append(f"合并 {action['ids']} → {action['new_content']}")
                elif action["action"] == "delete":
                    for fid in action["ids"]:
                        cur.execute("DELETE FROM facts WHERE id = ?", (fid,))
                    report.append(f"删除 {action['ids']}: {action.get('reason', '')}")

            self._conn.commit()
            return "✅ 记忆整合完成:\n" + "\n".join(f"  - {r}" for r in report)

        except Exception as e:
            return f"⚠️ 整合异常: {e}"

    # ========== 自动提取（核心功能）==========

    def auto_extract(self, user_input: str, agent_response: str) -> list[str]:
        """
        【基础功能】用 LLM 自动从对话中提取"值得记住的内容"
        【学习知识点】
            这是长期记忆系统最核心的技巧——不是用户说什么都记，而是让 LLM 判断：
            1. 是否包含新信息？（不是重复已知内容）
            2. 是否有长期价值？（"我喜欢Python" vs "今天天气不错"）
            3. 属于哪个类别？（identity / preference / plan / knowledge）

            Prompt 设计技巧：
            - 用 JSON 格式约束输出（方便解析）
            - 明确"不要记什么"（过滤噪音更有效）
            - 要求给出重要性评分（方便后续遗忘策略）

        调用示例：
            facts = ltm.auto_extract("我叫小明，喜欢Python", "你好小明！")
            for f in facts:
                ltm.remember(f["content"], f["category"], f["importance"])
        """
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""分析以下对话，提取值得存入长期记忆的信息。

对话:
  用户: {user_input}
  助手: {agent_response[:300]}

提取规则:
- 仅提取用户相关的信息（姓名、职业、偏好、计划、技能、背景等）
- 不要提取常识性内容（如"Python是编程语言"）
- 不要提取临时/一次性信息（如"帮我查一下天气"）
- 每条信息给出重要性评分 1-10
- 每条信息给出分类: identity/preference/plan/knowledge

严格按 JSON 格式回复（不要有其他文字）:
{{"facts": [
    {{"content": "用户叫小明", "category": "identity", "importance": 8}},
    {{"content": "用户偏好Python", "category": "preference", "importance": 6}}
]}}

如果没有任何值得记住的信息，返回: {{"facts": []}}""",
                }],
                temperature=0.3,
                max_tokens=400,
            )

            result_text = response.choices[0].message.content

            # 提取 JSON（兼容 markdown 代码块）
            if "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            # 去掉可能的 "json" 标记
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()

            data = json.loads(result_text)
            return data.get("facts", [])

        except Exception as e:
            print(f"  ⚠️ 自动提取失败: {e}")
            return []

    # ========== 统计信息 ==========

    def stats(self) -> dict:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) as n FROM facts")
        fact_count = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) as n FROM profile")
        profile_count = cur.fetchone()["n"]
        return {
            "facts": fact_count,
            "profile_fields": profile_count,
            "fact_vectors": self.fact_collection.count(),
            "episodes": self.episode_collection.count(),
            "db_path": self.sqlite_path,
        }

    def close(self):
        self._conn.close()


# ============================================================
# 第 3 部分：构建带长期记忆的 Agent
# ============================================================

def build_agent(memory: LongTermMemory, enable_auto_extract: bool = True) -> Agent:
    """
    【基础功能】创建带长期记忆工具的 Agent
    【学习知识点】
        工具设计原则：
        - remember/recall 是两个核心操作（存和取）
        - update_profile 是快捷操作（单独更新画像）
        - forget 是维护操作（清理无用记忆）
        - 每个工具的描述要精确，LLM 才能正确判断何时调用
    """

    # 工具函数：用闭包捕获 memory 实例
    def remember_tool(content: str, category: str = "general") -> str:
        """记住一个事实"""
        return memory.remember(content, category)

    def recall_tool(query: str) -> str:
        """搜索记忆"""
        # 同时搜索事实记忆和历史对话
        fact_result = memory.recall(query)
        episodes = memory.recall_episodes(query, top_k=2)
        if episodes:
            ep_text = "\n".join(
                [f"  [历史对话] {e['summary']} (相关度: {e['similarity']:.2f})" for e in episodes]
            )
            fact_result += f"\n\n📜 相关历史对话:\n{ep_text}"
        return fact_result

    def profile_tool(key: str, value: str) -> str:
        """更新用户画像"""
        return memory.update_profile(key, value)

    def forget_tool(query: str) -> str:
        """遗忘记忆"""
        return memory.forget(query)

    # 创建 Agent
    agent = Agent(
        name="记忆助手",
        system_prompt=f"""你是一个拥有长期记忆的智能助手。

当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

你的能力：
- **remember**：记住用户告诉你的信息（偏好、计划、身份等）
- **recall**：搜索过去的记忆（当用户问"我之前..."、"你还记得..."时使用）
- **update_profile**：更新用户画像（姓名、职业、偏好设置等）
- **forget**：忘记用户要求删除的信息

工作原则：
1. 当用户分享个人信息时（名字、职业、喜好、计划等），必须调用 remember 或 update_profile
2. 当用户问"我之前说过..."、"你还记得..."时，必须调用 recall 搜索记忆
3. 每次回答前，如果用户的问题与个人相关，先调用 recall 查看有没有相关记忆
4. 用记忆中的信息个性化回复（如称呼用户的名字）
5. 不要编造用户信息，一切以记忆为准""",
    )

    # 注册记忆工具
    agent.register_tool(
        "remember", remember_tool,
        "记住一条关于用户的信息。当用户说'我叫XX'、'我喜欢YY'、'我在做ZZ'时必须调用。",
        {
            "content": {"type": "string", "description": "需要记住的内容"},
            "category": {
                "type": "string",
                "description": "分类: identity(身份)/preference(偏好)/plan(计划)/knowledge(知识)",
            },
        },
    )
    agent.register_tool(
        "recall", recall_tool,
        "搜索长期记忆。当用户问'我之前说过...'、'你还记得吗'时必须调用。",
        {"query": {"type": "string", "description": "搜索查询"}},
    )
    agent.register_tool(
        "update_profile", profile_tool,
        "更新用户画像。如姓名、职业、编辑器偏好等。",
        {
            "key": {"type": "string", "description": "属性名（如 name, editor, language）"},
            "value": {"type": "string", "description": "属性值"},
        },
    )
    agent.register_tool(
        "forget", forget_tool,
        "删除指定的记忆。当用户说'忘记XX'、'不要记住YY'时调用。",
        {"query": {"type": "string", "description": "要遗忘的内容关键词"}},
    )

    # 保存引用用于 auto_extract
    agent._memory = memory
    agent._enable_auto_extract = enable_auto_extract

    return agent


# ============================================================
# 第 4 部分：交互式主程序
# ============================================================

def main():
    print("=" * 60)
    print("🧠 长期记忆 Agent — Long-Term Memory System")
    print("=" * 60)

    # 初始化记忆系统
    memory = LongTermMemory()
    stats = memory.stats()
    print(f"📊 记忆状态: {stats['facts']} 条事实 | {stats['profile_fields']} 条画像 | {stats['episodes']} 个对话片段")
    print(f"💾 数据库: {stats['db_path']}")
    print()

    # 显示已有画像（如果有的话——跨会话持久化验证）
    existing_profile = memory.get_profile()
    if existing_profile:
        print("👤 已有用户画像（来自上次会话）:")
        for k, v in existing_profile.items():
            print(f"   {k}: {v}")
        print()

    # 构建 Agent
    agent = build_agent(memory)

    print("✅ Agent 就绪！试试以下操作：")
    print("  '我叫张三，我是做前端开发的'  → Agent 自动记住")
    print("  '我平时用 WebStorm 和 VS Code' → Agent 自动记住偏好")
    print("  '我之前说我做什么工作的？'     → Agent 搜索记忆")
    print("  '忘记我用的编辑器'             → Agent 删除记忆")
    print("  'stats'  → 查看记忆统计")
    print("  'profile' → 查看用户画像")
    print("  'consolidate' → 整合重复记忆")
    print("  'quit' → 退出")
    print()
    print("💡 试试关闭程序再重新运行，看记忆是否还在！")
    print()

    while True:
        try:
            user_input = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            s = memory.stats()
            print(f"\n📊 记忆统计:")
            print(f"   事实记忆: {s['facts']} 条 ({s['fact_vectors']} 条向量索引)")
            print(f"   用户画像: {s['profile_fields']} 个字段")
            print(f"   对话片段: {s['episodes']} 条")
            print(f"   数据库: {s['db_path']}")
            print()
            continue
        if user_input.lower() == "profile":
            p = memory.get_profile()
            if p:
                print("\n👤 用户画像:")
                for k, v in p.items():
                    print(f"   {k}: {v}")
            else:
                print("\n👤 暂无画像信息")
            print()
            continue
        if user_input.lower() == "consolidate":
            print(f"\n🔄 正在整合记忆...")
            result = memory.consolidate()
            print(result)
            print()
            continue

        # 正常对话
        response = agent.run(user_input, verbose=True)

        # 自动提取记忆（对话后）
        facts = memory.auto_extract(user_input, response)
        for f in facts:
            result = memory.remember(f["content"], f.get("category", "general"), f.get("importance", 5))
            print(f"  🧠 自动记忆: {result}")

        # 记录对话片段
        episode_result = memory.remember_episode(user_input, response)
        print(f"  📝 {episode_result}")

        print(f"\n🤖 {response}")
        print("\n" + "-" * 40)

    # 退出前清理
    memory.close()
    print_summary()


# ============================================================
# 第 5 部分：概念总结
# ============================================================

def print_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              长期记忆系统 核心要点总结                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 短期记忆 vs 长期记忆                                      ║
║     self.messages 列表 = 会话记忆（关闭程序就没了）           ║
║     SQLite + ChromaDB = 长期记忆（跨会话持久化）              ║
║     两者互补：短期记忆负责当前对话，长期记忆负责跨会话        ║
║                                                              ║
║  2. 三种记忆类型（认知科学来源）                              ║
║     画像记忆(Profile)：你是谁 → SQLite key-value              ║
║     事实记忆(Facts)：关于你的离散事实 → SQLite + ChromaDB     ║
║     片段记忆(Episodes)：历史对话摘要 → ChromaDB               ║
║                                                              ║
║  3. 混合存储架构                                              ║
║     SQLite 负责：精确查询（"name = ?"）                       ║
║     ChromaDB 负责：语义搜索（"用户喜欢什么编辑器？"）         ║
║     两者配合 = 精确 + 模糊 = 完整的记忆检索                   ║
║                                                              ║
║  4. 记忆生命周期                                              ║
║     Extract（LLM自动提取）→ Store（双写SQLite+ChromaDB）     ║
║     → Retrieve（语义+精确搜索）→ Consolidate（合并重复）      ║
║     → Forget（删除/衰减）                                     ║
║                                                              ║
║  5. 知识库 vs 长期记忆                                        ║
║     知识库(RAG): 外部文档知识（公司手册、产品定价）           ║
║     长期记忆: 用户个人信息（偏好、计划、历史）                ║
║     两者都可用 ChromaDB，但目的完全不同                       ║
║                                                              ║
║  6. 记忆安全的工程原则                                         ║
║     - 只记用户主动分享的信息                                  ║
║     - 支持遗忘（用户说"忘记XX"立即删除）                      ║
║     - 不记录敏感信息（密码、身份证号等）                       ║
║     - 生产环境需加密 + 访问控制                               ║
║                                                              ║
║  7. 进阶方向                                                  ║
║     - 记忆衰减：旧记忆自动降低权重                            ║
║     - 记忆摘要：定期用 LLM 把多条记忆压缩为一条               ║
║     - 记忆图谱：用 Neo4j 建立记忆间的关联                     ║
║     - 多用户隔离：不同用户独立的记忆空间                      ║
║     - 加密存储：敏感信息的 AES 加密                           ║
║                                                              ║
║  学习路径回顾:                                                ║
║     step01-07 → 单 Agent 开发全栈                             ║
║     step08-10 → RAG + 知识库 + 文档摄入                       ║
║     step11    → 联网搜索                                      ║
║     step12    → 多智能体协作                                  ║
║     step13    → 长期记忆系统 ← 你在这里                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
