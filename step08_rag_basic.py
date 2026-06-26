"""
Step 8: RAG — Retrieval-Augmented Generation 检索增强生成

学习目标:
  1. 理解 RAG 的核心思想 — 为什么 LLM 需要"外挂知识库"
  2. 掌握文档分块（Chunking）策略
  3. 理解 Embedding — 文本如何变成向量，向量如何衡量"相似度"
  4. 学会使用向量数据库 ChromaDB 存储和检索
  5. 搭建完整 RAG 流水线: 文档 → 分块 → 向量化 → 存储 → 检索 → 生成

向量数据库选择 — ChromaDB:
  - 嵌入式数据库，pip install 即可，无需额外服务
  - SQLite 持久化，重启不丢数据
  - API 简洁，5 分钟上手
  - 适合学习、原型、中小规模应用

其他常见选择:
  ┌──────────┬─────────────────────────────────────────┐
  │ ChromaDB │ 嵌入式、Python 原生 → 学习首选           │
  │ FAISS    │ Meta 开源、C++ 核心、极快 → 大规模检索   │
  │ Qdrant   │ Rust、高性能、丰富过滤 → 生产环境        │
  │ Milvus   │ 分布式、GPU 索引 → 企业级                │
  │ LanceDB  │ 列式存储、无服务器 → 数据分析场景        │
  └──────────┴─────────────────────────────────────────┘

RAG 流程:
  准备阶段: 文档 → 分块 → 向量化(Embedding) → 存入 ChromaDB
  查询阶段: 问题 → 向量化 → 相似度检索(top-K) → 拼入 Prompt → LLM 生成

运行: python step08_rag_basic.py
前置: pip install chromadb requests
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ChromaDB 和 HTTP 请求（Embedding API）
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

# LLM 客户端（生成回答用）
llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v2"
EMBEDDING_DIM = 1536  # text-embedding-v2 的向量维度

CHUNK_SIZE = 500       # 每个文档块的最大字符数
CHUNK_OVERLAP = 80     # 相邻块的重叠字符数（保持上下文连贯）
TOP_K = 3              # 检索时返回最相关的前 K 个文档块

# ============================================================
# 1. 示例文档 — 虚构公司资料（LLM 训练数据中不存在）
# ============================================================

SAMPLE_DOCS = [
    {
        "title": "星辰科技公司简介",
        "content": """
星辰科技（StarTech）成立于 2024 年 3 月，总部位于杭州未来科技城。
公司专注于 AI 智能办公解决方案，是行业内增长最快的创业公司之一。

核心产品线:
1. StarDocs — 智能文档协作平台
2. StarMeet — AI 会议纪要助手
3. StarFlow — 自动化工作流引擎

管理团队:
- CEO: 张明远（前阿里巴巴高级总监，15 年企业服务经验）
- CTO: 李思涵（前腾讯 AI Lab 研究员，自然语言处理博士）
- COO: 王雨婷（前字节跳动产品副总裁）

公司规模: 约 120 人，其中研发团队占比 65%。
企业愿景: 让每一家企业都拥有 AI 原生的办公方式。
""",
    },
    {
        "title": "StarDocs 产品详情与定价",
        "content": """
StarDocs 是星辰科技的旗舰产品，新一代 AI 智能文档协作平台。

核心功能:
- AI 智能摘要: 一键生成长文档摘要，支持中英日韩 4 种语言
- 多语言实时翻译: 文档内选中即译，保持排版不变
- 代码协作编辑: 支持 50+ 编程语言的在线编辑和 AI 代码补全
- 智能搜索: 跨所有工作区全文搜索，语义理解
- 版本管理: Git 式文档版本控制，分支合并

定价方案（2025 年更新）:
- 个人版: ￥29/月，10GB 存储，基础 AI 功能（摘要 + 翻译）
- 团队版: ￥99/人/月，100GB 存储，高级 AI + 权限管理 + API 接入
- 企业版: ￥299/人/月，1TB 存储，私有部署 + 定制 AI 模型训练 + 专属运维

2025 年 Q1 新增: AI 智能表格、手写笔记 OCR、PDF 合同智能审核。
2025 年 Q2 计划: 发布 StarDocs Mobile 3.0，支持离线 AI。
""",
    },
    {
        "title": "StarMeet 产品详情",
        "content": """
StarMeet 是 AI 驱动的智能会议工具，帮助团队告别低效会议。

核心功能:
- 实时语音转文字: 准确率 98%，支持中、英、日、韩 4 种语言
- AI 会议纪要: 自动提取要点、决议、待办事项，会后秒出纪要
- 发言人识别: 自动区分不同发言人并标记
- 情绪分析: 分析会议氛围，标注关键争论点
- 录像分段: 自动标注会议录像的关键时间点，点击跳转回放
- 深度集成: 飞书、钉钉、企业微信、Slack、Teams

定价:
- 基础版: 免费（每月 20 小时录音时长）
- 专业版: ￥49/月（不限时长 + 高级 AI 功能）
- 企业版: ￥99/人/月（私有化部署 + 自定义 AI 模型）

里程碑: 2025 年 6 月付费用户突破 50 万，覆盖 200+ 国家和地区。
""",
    },
    {
        "title": "星辰科技融资历史与估值",
        "content": """
融资历程:
- 2024 年 3 月: 天使轮 ￥2000 万
  投资方: 红杉中国种子基金
  用途: 核心团队组建、MVP 开发

- 2024 年 9 月: A 轮 ￥1.5 亿
  领投: 高瓴资本
  跟投: 红杉中国、蓝驰创投
  投后估值: ￥8 亿
  用途: 产品迭代、市场拓展、团队扩张至 120 人

- 2025 年 6 月: B 轮 ￥5 亿（最新）
  领投: 软银愿景基金三期
  跟投: 高瓴资本、红杉中国、GGV 纪源资本
  投后估值: ￥50 亿
  用途: 海外市场拓展、AI 大模型研发、生态建设

当前状态:
- 总融资额: ￥6.7 亿
- ARR（年经常性收入）: ￥8000 万（2025 年预计）
- 团队: 120 人 → 预计年底扩至 200 人
- 下一轮: C 轮预计 2026 年 Q1，目标估值 ￥150 亿
""",
    },
]


# ============================================================
# 2. 文档分块 — 递归字符分割
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本切分为重叠的小块。

    策略: 段落 → 句子 → 定长（递归降级）
    - 优先按段落（\\n\\n）切分，保持语义完整
    - 段落仍过长则按句子（。！？\\n）切分
    - 句子仍过长则按固定长度切分
    - 块之间保留 overlap 字符重叠，避免关键信息被切断

    为什么需要 overlap？
      例如: "星辰科技成立于 2024 年。\n\nCEO 张明远说..."
      如果刚好在 "CEO 张明远" 处切开，下一块开头就是 "说..."
      有了 overlap，后一块会包含 "...CEO 张明远说..." 的上下文
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # 级别 1: 按段落切
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para:
                chunks.extend(chunk_text(para, chunk_size, overlap))
        return chunks

    # 级别 2: 按句子切（中英文标点）
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

    # 级别 3: 定长滑动窗口
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
# 3. Embedding — 调用 DashScope API 将文本转为向量
# ============================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    调用 DashScope Text Embedding API 批量获取向量。

    text-embedding-v2:
      - 维度: 1536
      - 单次最多 25 条文本
      - 中文优化，也支持英文

    API 文档: https://help.aliyun.com/document_detail/dashscope.html
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    all_embeddings = []
    batch_size = 25  # API 限制

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"batch: {batch}")
        payload = {
            "model": EMBEDDING_MODEL,
            "input": {"texts": batch},
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # 按 text_index 排序确保顺序正确
            embeddings = sorted(
                data["output"]["embeddings"],
                key=lambda x: x["text_index"],
            )
            all_embeddings.extend([e["embedding"] for e in embeddings])

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Embedding API 请求失败: {e}")
            raise
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  ⚠️ Embedding API 响应异常: {e}")
            print(f"  响应内容: {resp.text[:500]}")
            raise

    return all_embeddings


# ============================================================
# 4. 向量数据库 — ChromaDB 封装
# ============================================================

class VectorStore:
    """
    ChromaDB 向量存储封装。

    核心概念:
    - Collection（集合）= 一个知识库，类似数据库的"表"
    - 每个文档 = 一段文本 + 一个向量 + 可选的元数据
    - 检索 = 用查询向量和库中所有向量计算相似度，返回最相似的几个

    相似度计算: 默认用余弦相似度 (cosine)，值越接近 1 越相似。
    也可以选 L2 距离或内积 (inner product)。
    """

    def __init__(self, name: str = "knowledge_base", persist_dir: str = None):
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent / "chroma_db")

        # PersistentClient: 数据存到磁盘，重启不丢失
        # 学习测试时也可以用 EphemeralClient（内存中，进程结束即消失）
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.name = name

        # 删除旧 collection（每次演示重新开始），生产环境不要这样
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    def add(self, texts: list[str], metadatas: list[dict] = None):
        """将文本块向量化后存入数据库"""
        if not texts:
            return

        print(f"  📐 向量化 {len(texts)} 个文档块...")
        start = time.time()
        embeddings = get_embeddings(texts)
        elapsed = time.time() - start
        print(f"  ✅ 向量化完成，耗时 {elapsed:.1f}s，维度: {len(embeddings[0])}")

        ids = [f"chunk_{i}" for i in range(len(texts))]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(texts),
        )
        print(f"  💾 已存入 ChromaDB: {self.collection.count()} 条记录")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """检索最相关的 top_k 个文档块"""
        # 1. 把查询文本也向量化
        query_embedding = get_embeddings([query])[0]

        # 2. 在向量库中搜索最相似的文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 3. 整理结果
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # 余弦距离转相似度: 余弦距离 = 1 - 余弦相似度
                similarity = 1 - distance
                retrieved.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(similarity, 4),
                })

        return retrieved


# ============================================================
# 5. RAG 流水线 — 完整的检索增强生成
# ============================================================

class RAGPipeline:
    """
    RAG 流水线: 摄入文档 → 检索 → 增强生成

    使用:
      rag = RAGPipeline()
      rag.ingest(documents)           # 准备知识库
      answer = rag.query("问题")       # 查询
    """

    def __init__(self):
        self.vector_store: VectorStore = None
        self.is_ready = False

    def ingest(self, documents: list[dict]):
        """摄入文档: 分块 → 向量化 → 存入向量库"""
        print("\n📥 摄入文档...")

        all_chunks = []
        all_metadatas = []

        for doc in documents:
            chunks = chunk_text(doc["content"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"title": doc.get("title", ""), "chunk_size": len(chunk)})

        print(f"  📄 {len(documents)} 篇文档 → {len(all_chunks)} 个文本块")

        self.vector_store = VectorStore()
        self.vector_store.add(all_chunks, all_metadatas)
        self.is_ready = True

    def query(self, question: str, top_k: int = TOP_K, verbose: bool = True) -> dict:
        """
        执行 RAG 查询。

        返回: {"answer": str, "sources": list[dict]}
        """
        if not self.is_ready:
            return {"answer": "知识库未初始化，请先调用 ingest()", "sources": []}

        if verbose:
            print(f"\n🔍 检索相关文档...")

        # Step 1: 检索相关文档块
        sources = self.vector_store.search(question, top_k=top_k)

        if verbose:
            for i, s in enumerate(sources):
                preview = s["content"][:100].replace("\n", " ")
                print(f"  [{i+1}] 相似度={s['similarity']:.3f} | {preview}...")

        # Step 2: 拼接检索结果作为上下文
        context = "\n\n---\n\n".join([
            f"[来源: {s['metadata'].get('title', '未知')}] (相似度: {s['similarity']})\n{s['content']}"
            for s in sources
        ])

        # Step 3: 调用 LLM 基于上下文生成回答
        prompt = f"""你是一个基于知识库回答问题的助手。请严格根据以下参考资料回答问题。

参考资料:
{context}

问题: {question}

要求:
- 只使用参考资料中的信息回答，不要编造
- 如果资料中没有相关信息，请明确说"参考资料中未提及"
- 回答要准确、简洁
- 如果引用了具体数据，请注明"""

        if verbose:
            print(f"\n🤖 生成回答...")
        print(prompt)
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        return {"answer": answer, "sources": sources}


# ============================================================
# 6. 对比实验 — RAG vs 纯 LLM
# ============================================================

def ask_llm_directly(question: str) -> str:
    """不使用 RAG，直接问 LLM"""
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


# ============================================================
# 7. 向量相似度可视化 — 帮助理解 Embedding
# ============================================================

def demo_embedding_concept():
    """演示: 文本相似 → 向量相似"""
    print("\n" + "=" * 60)
    print("🧪 Embedding 概念演示")
    print("=" * 60)
    print("同义词、近义词、无关词的向量余弦相似度:\n")

    texts = [
        "人工智能技术",
        "AI 相关领域",
        "机器学习算法",
        "今天天气很好",
        "我喜欢吃水果",
    ]

    print("  ⏳ 计算向量...")
    embeddings = get_embeddings(texts)

    # 计算两两之间的余弦相似度（简化版）
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    # 以第一个文本为基准
    base = embeddings[0]
    print(f"\n  基准文本: 「{texts[0]}」\n")
    print(f"  {'对比文本':<20} {'相似度':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        sim = cosine_sim(base, emb)
        bar = "█" * int(sim * 20)
        print(f"  {text:<20} {sim:>7.4f}  {bar}")

    print("\n  💡 语义相关（AI、机器学习）分数高，无关的（天气、水果）分数低。")
    print("     这就是向量检索的原理 — 在向量空间中找到最「近」的文档。")


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("📚 RAG — 检索增强生成")
    print("=" * 60)
    print(f"LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM}维)")
    print(f"向量库: ChromaDB | 分块大小: {CHUNK_SIZE} 字符 | 重叠: {CHUNK_OVERLAP}")
    print()

    # -------- 演示 1: Embedding 概念 --------
    demo_embedding_concept()

    # -------- 演示 2: 构建知识库 --------
    print("\n" + "=" * 60)
    print("🏗️ 构建知识库")
    print("=" * 60)
    print(f"示例文档: {len(SAMPLE_DOCS)} 篇（星辰科技公司资料）")
    print("这些信息是虚构的，LLM 的训练数据中没有 → 适合测试 RAG\n")

    rag = RAGPipeline()
    rag.ingest(SAMPLE_DOCS)

    # -------- 演示 3: RAG vs 纯 LLM 对比 --------
    questions = [
        "StarDocs 的团队版定价是多少？",
        "星辰科技的 CEO 是谁？CTO 是谁？",
        "星辰科技总融资额是多少？最新估值多少？",
    ]

    print("\n" + "=" * 60)
    print("⚔️ RAG vs 纯 LLM 对比")
    print("=" * 60)

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"❓ 问题: {q}")

        # RAG 回答
        rag_result = rag.query(q, top_k=2, verbose=False)
        print(f"\n📚 RAG 回答:")
        print(f"   {rag_result['answer']}")

        # 纯 LLM 回答（无 RAG）
        print(f"\n🤖 纯 LLM（无知识库）:")
        direct_answer = ask_llm_directly(q)
        print(f"   {direct_answer}")

        print(f"\n💡 对比: RAG 基于真实文档，纯 LLM 靠「猜」（可能编造或说不知道）")

    # -------- 演示 4: 显示检索过程 --------
    print("\n" + "=" * 60)
    print("🔬 深入观察一次检索过程")
    print("=" * 60)

    question = "StarMeet 有什么核心功能？怎么收费？"
    print(f"\n问题: {question}\n")

    result = rag.query(question, top_k=3, verbose=True)
    print(f"\n📚 最终答案:\n{result['answer']}")

    # -------- 交互模式 --------
    print("\n" + "=" * 60)
    print("💬 交互问答模式")
    print("=" * 60)
    print("输入问题查询知识库，输入 'quit' 退出")
    print("试试: '张明远是谁？' 'B轮融资什么时候？'\n")

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

        result = rag.query(user_input, top_k=TOP_K, verbose=False)
        print(f"\n📚 {result['answer']}")
        if result["sources"]:
            print(f"\n📎 参考来源:")
            for s in result["sources"]:
                print(f"   [{s['similarity']}] {s['metadata'].get('title', '未知')}")
        print()

    # -------- 概念总结 --------
    print("\n" + "=" * 60)
    print("📚 RAG 核心概念总结")
    print("=" * 60)
    print("""
  1. Embedding（向量化）
     将文本映射为高维空间中的点。语义相近的文本，向量距离近。
     DashScope text-embedding-v2 输出 1536 维向量。

  2. 向量数据库
     专门存储和检索向量的数据库。通过 ANN（近似最近邻）算法，
     在海量向量中快速找到最相似的几个，不需要逐个比较。

  3. RAG 三阶段:
     - 离线准备: 文档 → 分块 → 向量化 → 存入向量库
     - 在线检索: 用户问题 → 向量化 → 相似度搜索 → 取 top-K
     - 增强生成: 检索结果 + 问题 → 拼接 Prompt → LLM 生成回答

  4. RAG 解决的问题:
     - 知识截止: LLM 不知道训练后的新信息
     - 幻觉: LLM "编造"不存在的事实
     - 私有知识: LLM 没有公司内部文档
     - 可追溯: RAG 回答可以引用来源

  5. 向量数据库选择:
     学习/原型 → ChromaDB（嵌入式，最简单）
     生产/高并发 → Qdrant 或 Milvus
     离线/批量 → FAISS（最快）

  下一步: step09_rag_agent.py — 将 RAG 集成到 Agent 框架
  """)

    print("运行 step09 继续学习: python step09_rag_agent.py")


if __name__ == "__main__":
    main()
