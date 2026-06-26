# ChromaDB 从零到企业级实战指南

> 适用版本: ChromaDB 0.5.x+ | 语言: Python 3.10+ | 最后更新: 2026-06

---

## 目录

1. [什么是 ChromaDB](#1-什么是-chromadb)
2. [核心概念](#2-核心概念)
3. [快速开始](#3-快速开始)
4. [核心 API 详解](#4-核心-api-详解)
5. [Embedding 策略](#5-embedding-策略)
6. [检索进阶](#6-检索进阶)
7. [元数据与过滤](#7-元数据与过滤)
8. [多 Collection 管理](#8-多-collection-管理)
9. [性能优化](#9-性能优化)
10. [生产环境部署](#10-生产环境部署)
11. [监控与运维](#11-监控与运维)
12. [与其他向量数据库对比](#12-与其他向量数据库对比)
13. [常见问题与反模式](#13-常见问题与反模式)
14. [企业级完整示例](#14-企业级完整示例)

---

## 1. 什么是 ChromaDB

### 1.1 一句话定义

**ChromaDB 是一个开源的向量数据库，专门用来存储和搜索 AI 生成的向量（Embedding）。**

### 1.2 类比理解

```
传统数据库（MySQL）:              向量数据库（ChromaDB）:
  存储 → 学生成绩表                  存储 → 文档向量
  查询 → SELECT * WHERE 分数>90     查询 → 找到和这句话意思最相近的文档
  索引 → B+树（按数值排序）          索引 → HNSW（按向量距离排序）
```

### 1.3 为什么需要向量数据库

LLM 的知识有两个致命问题：

| 问题 | 示例 | 向量数据库如何解决 |
|------|------|-------------------|
| 知识截止 | 问 "Python 3.13 有什么新特性？" | 把 Python 3.13 文档存进去，检索后喂给 LLM |
| 私有知识 | 问 "公司内部 API 怎么调用？" | 把内部文档向量化存储，语义搜索 |
| 幻觉 | LLM 编造不存在的 API | 基于检索到的真实文档回答 |
| Token 限制 | 100 页 PDF 塞不进 Context | 只检索最相关的 3 段，约 1500 字 |

### 1.4 ChromaDB 的定位

```
开发者友好度  高  ★★★★★  (pip install 即用)
部署复杂度    低  ★☆☆☆☆  (嵌入式，无需独立服务)
查询速度      中  ★★★☆☆  (百万级向量够用)
水平扩展      低  ★★☆☆☆  (单机为主)
生产成熟度    中  ★★★☆☆  (适合中小规模，大厂用 Milvus/Qdrant)
```

**一句话: 学习首选、原型首选、中小规模生产可用。十万到百万级文档量最适合。**

---

## 2. 核心概念

### 2.1 三个核心抽象

```
Client (客户端)
  └── Collection (集合)  ← 类似数据库的 Table
       ├── Document (文档)  ← 原始文本
       ├── Embedding (向量)  ← 文本的数学表示 [0.1, -0.3, ...]
       └── Metadata (元数据)  ← 标签、来源、时间等
```

### 2.2 Collection（集合）

一个 Collection 就是一个独立的知识库，拥有独立的索引和命名空间。

```python
# 不同业务用不同 Collection
kb_tech = client.get_or_create_collection("技术文档")
kb_hr   = client.get_or_create_collection("HR制度")
kb_faq  = client.get_or_create_collection("常见问题")
```

**企业级实践**: 按领域 / 权限 / 环境分 Collection，不要所有文档堆一个库里。

### 2.3 Embedding（向量 / 嵌入）

```
文本: "今天天气真好"
  ↓ Embedding 模型
向量: [0.12, -0.34, 0.56, 0.78, ..., -0.09]  ← 1536 个浮点数
```

**关键性质**: 语义相近的文本 → 向量距离近。这使得计算机可以"理解"文本含义。

### 2.4 相似度度量

| 度量方式 | 取值范围 | 适用场景 |
|----------|---------|----------|
| **余弦相似度 (cosine)** | [-1, 1] | 文本语义相似（ChromaDB 默认） |
| 欧氏距离 (L2) | [0, ∞) | 图像、数值特征 |
| 内积 (inner product / dot) | [-∞, ∞] | 需要原始分数的场景 |

---

## 3. 快速开始

### 3.1 安装

```bash
pip install chromadb
```

**零依赖**: 不需要 Docker，不需要数据库服务，不需要额外配置。ChromaDB 内嵌 SQLite。

### 3.2 5 分钟上手

```python
import chromadb

# 1. 创建客户端 — 数据存到磁盘
client = chromadb.PersistentClient(path="./my_db")

# 2. 创建集合
collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},  # 余弦相似度
)

# 3. 添加文档 — ChromaDB 自动做 Embedding（内置模型）
collection.add(
    documents=[
        "Chromadb 是一个向量数据库",
        "Python 是一门编程语言",
        "今天晚饭吃了火锅",
    ],
    metadatas=[
        {"source": "intro.txt"},
        {"source": "intro.txt"},
        {"source": "diary.txt"},
    ],
    ids=["doc_1", "doc_2", "doc_3"],
)

# 4. 查询 — 自动把查询也 Embedding
results = collection.query(
    query_texts=["什么是向量数据库？"],
    n_results=2,
)

# 5. 看结果
for i, (doc_id, doc, distance) in enumerate(zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0],
)):
    print(f"[{i+1}] {doc} (距离: {distance:.4f})")

# 输出:
# [1] Chromadb 是一个向量数据库 (距离: 0.2345)
# [2] Python 是一门编程语言 (距离: 0.7891)
```

### 3.3 两种客户端模式

```python
# 持久化模式 — 数据存磁盘，重启不丢（生产用）
client = chromadb.PersistentClient(path="./chroma_data")

# 内存模式 — 进程结束数据消失（测试 / 临时用）
client = chromadb.EphemeralClient()

# HTTP 客户端 — 连接远程 ChromaDB 服务器
client = chromadb.HttpClient(host="localhost", port=8000)
```

---

## 4. 核心 API 详解

### 4.1 Collection CRUD

```python
import chromadb

client = chromadb.PersistentClient(path="./db")

# ── 创建 ──
collection = client.create_collection(
    name="docs",
    metadata={
        "hnsw:space": "cosine",           # 距离度量
        "hnsw:construction_ef": 100,      # 构建时搜索宽度（越大越精确越慢）
        "hnsw:search_ef": 50,             # 搜索时搜索宽度
        "hnsw:M": 16,                     # 每个节点的最大连接数
    },
)

# ── 获取/创建（推荐） ──
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)

# ── 获取已有 ──
collection = client.get_collection(name="docs")

# ── 列出所有 ──
for col in client.list_collections():
    print(col.name, col.count())

# ── 删除 ──
client.delete_collection(name="docs")

# ── 重命名 ──
collection.modify(name="docs_v2")

# ── 查看详情 ──
print(f"名称: {collection.name}")
print(f"文档数: {collection.count()}")
print(f"配置: {collection.metadata}")
```

### 4.2 添加数据 (Add)

```python
collection.add(
    ids=["id_1", "id_2", "id_3"],         # 必填，唯一标识

    # ── 以下至少提供一个 ──
    documents=["文本A", "文本B", "文本C"],  # 原始文本
    embeddings=[                            # 手动提供向量
        [0.1, 0.2, ...],                   # 如果提供 embedding 则不会自动调用模型
        [0.3, 0.4, ...],
        [0.5, 0.6, ...],
    ],

    # ── 可选 ──
    metadatas=[                             # 元数据（过滤用）
        {"source": "a.pdf", "page": 1, "created": "2024-01-01"},
        {"source": "b.pdf", "page": 2, "created": "2024-01-02"},
        {"source": "c.pdf", "page": 3, "created": "2024-01-03"},
    ],
)
```

**规则**: `documents` 和 `embeddings` 至少提供一个。同时提供时，以 `embeddings` 为准。

### 4.3 更新与 Upsert

```python
# update — 更新已有文档（id 必须存在）
collection.update(
    ids=["id_1"],
    documents=["修改后的文本"],
    metadatas=[{"source": "a.pdf", "page": 1, "updated": True}],
)

# upsert — 有则更新，无则插入（生产环境推荐）
collection.upsert(
    ids=["id_1", "id_new"],
    documents=["文本1", "新增文本"],
)
```

### 4.4 删除数据

```python
# 按 ID 删除
collection.delete(ids=["id_1", "id_2"])

# 按元数据过滤删除（需要先查再删）
results = collection.get(where={"source": "deprecated.pdf"})
collection.delete(ids=results["ids"])
```

### 4.5 查询数据 (Get)

```python
# 获取全部（谨慎，可能数据量很大）
all_data = collection.get()

# 按 ID 获取
data = collection.get(ids=["id_1", "id_2"])

# 按元数据过滤获取
data = collection.get(
    where={"source": "a.pdf"},      # 精确匹配
    limit=10,
    offset=20,                      # 分页
    include=["documents", "metadatas"],  # 指定返回字段
)

# 查看前 N 条（不排序，调试用）
sample = collection.peek(limit=5)
```

### 4.6 查询数据 (Query — 语义搜索)

```python
# ── 方式1: 用文本查询（自动 embedding） ──
results = collection.query(
    query_texts=["如何部署到生产环境？"],  # 自动 embedding
    n_results=5,                            # 返回前 5 个
    include=["documents", "metadatas", "distances"],
)

# ── 方式2: 用向量查询（手动传入 embedding） ──
results = collection.query(
    query_embeddings=[[0.1, -0.3, 0.5, ...]],  # 预计算的向量
    n_results=5,
)

# ── 带过滤的查询 ──
results = collection.query(
    query_texts=["python语法"],
    n_results=5,
    where={"category": "技术文档"},      # 先过滤再搜索
    where_document={"$contains": "Python"},  # 文档内容过滤
)

# ── 结果结构 ──
# results = {
#     "ids": [["id_5", "id_2", "id_8"]],        # 最相关→最不相关
#     "documents": [["文本5", "文本2", "文本8"]],
#     "metadatas": [[{...}, {...}, {...}]],
#     "distances": [[0.23, 0.45, 0.67]],         # 距离越小越相关
#     "embeddings": None,                         # 默认不返回向量
# }
```

---

## 5. Embedding 策略

### 5.1 三种 Embedding 方式

```python
# ── 方式1: ChromaDB 内置模型 ──
# 优点: 零配置，开箱即用
# 缺点: 默认模型 all-MiniLM-L6-v2（英文为主），需要下载（~90MB）
collection = client.create_collection(name="docs")
# 不传 embeddings 参数 = 自动使用内置模型

# ── 方式2: 手动调用 API（推荐生产环境） ──
# 优点: 模型可控，中文友好，不依赖本地下载
def get_embeddings_from_api(texts: list[str]) -> list[list[float]]:
    """调用 DashScope / OpenAI / 自建 Embedding 服务"""
    import requests
    resp = requests.post(
        "https://your-api.com/v1/embeddings",
        json={"input": texts, "model": "text-embedding-v2"},
        headers={"Authorization": "Bearer xxx"},
    )
    return [e["embedding"] for e in resp.json()["data"]]

embeddings = get_embeddings_from_api(documents)
collection.add(ids=ids, documents=documents, embeddings=embeddings)

# ── 方式3: 自定义 Embedding Function ──
# 优点: 封装复用，一行配置
from chromadb import Documents, EmbeddingFunction, Embeddings

class DashScopeEmbedding(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # ... 调用 API ...
        return embeddings_list

collection = client.create_collection(
    name="docs",
    embedding_function=DashScopeEmbedding(),
)
# 后续 add / query 自动使用该函数
```

### 5.2 Embedding 模型选型指南

| 模型 | 维度 | 语言 | 调用方式 | 成本 | 适用场景 |
|------|------|------|----------|------|----------|
| all-MiniLM-L6-v2 | 384 | 英文 | 本地 | 免费 | 英文原型 |
| text-embedding-v2 (DashScope) | 1536 | 中英 | API | ￥0.0007/千 tokens | 中文生产 |
| text-embedding-3-small (OpenAI) | 512/1536 | 多语言 | API | $0.02/1M tokens | 多语言 |
| BGE-M3 (BAAI) | 1024 | 中英 | 本地/API | 免费 | 中文私有化 |
| GTE-large (阿里) | 1024 | 中文 | 本地 | 免费 | 中文离线场景 |

### 5.3 Embedding 维度选择

```
维度越高 → 表达能力越强 → 存储和计算成本越高

384 维:  原型、功能验证
768 维:  中等规模知识库
1024 维: 通用生产环境（BGE / GTE）
1536 维: 高精度场景（OpenAI / DashScope）
```

**企业建议**: 生产环境选 1024 或 1536 维。384 维只能做简单语义匹配。

---

## 6. 检索进阶

### 6.1 检索流程优化

```
                   用户问题
                      │
                      ▼
              查询改写 / 扩展
            （处理拼写错误、补充上下文）
                      │
                      ▼
                  Embedding
                      │
                      ▼
         ┌── 元数据过滤（缩小范围）──┐
         │    where={"env": "prod"}  │
         └─────────────────────────┘
                      │
                      ▼
              HNSW 向量检索
                      │
                      ▼
              Top-K 候选结果
                      │
                      ▼
              ┌── 重排序 (Re-rank) ──┐
              │   Cross-encoder 精排 │
              └────────────────────┘
                      │
                      ▼
              最终 Top-N 结果
```

### 6.2 查询改写

```python
def rewrite_query(original: str, history: list[str] = None) -> str:
    """
    解决用户口语化 / 指代不明的问题。

    示例:
      原始: "那个怎么配置？"    → 改写: "Kubernetes Ingress 如何配置？"
      原始: "多少钱？"          → 改写: "StarDocs 专业版价格"
    """
    context = ""
    if history:
        context = "对话历史:\n" + "\n".join(history[-3:])

    prompt = f"""{context}
将用户问题改写为独立、完整的检索查询语句（去口语化、补全省略）:
用户问题: {original}"""

    response = llm_client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()
```

### 6.3 多路召回 (Hybrid Search)

单一向量检索不够，结合关键词检索可以补全：

```python
def hybrid_search(collection, query: str, top_k: int = 10):
    """向量检索 + 关键词检索融合"""

    # 路1: 语义向量检索
    semantic_results = collection.query(query_texts=[query], n_results=top_k)

    # 路2: 关键词检索（用 ChromaDB 的 where_document）
    keywords = query.split()
    keyword_results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where_document={"$or": [
            {"$contains": kw} for kw in keywords
        ]},
    )

    # 融合: 合并 + 去重 + 按语义分数重排
    all_ids = set()
    fused = []
    for ids, docs, metas, dists in [
        (semantic_results["ids"][0], semantic_results["documents"][0],
         semantic_results["metadatas"][0], semantic_results["distances"][0])
    ]:
        for i, doc_id in enumerate(ids):
            if doc_id not in all_ids:
                all_ids.add(doc_id)
                fused.append({
                    "id": doc_id,
                    "document": docs[i],
                    "metadata": metas[i],
                    "score": 1 - dists[i],
                })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]
```

### 6.4 Top-K 选择策略

```python
# 经验法则
TOP_K_MAP = {
    "问答": 3,          # 事实性问答，3 个足够
    "总结": 5,          # 需要覆盖更多上下文
    "对比": 5,          # 对比多个来源
    "研究": 10,         # 深度分析需要更多素材
}

# 动态 Top-K
def dynamic_top_k(query: str, total_docs: int) -> int:
    if total_docs < 100:
        return 3
    elif total_docs < 10000:
        return 5
    else:
        return 10
```

---

## 7. 元数据与过滤

### 7.1 元数据设计原则

```python
# ❌ 不好 — 元数据太简陋
{"source": "doc.pdf"}

# ✅ 好 — 元数据包含过滤维度
{
    "source": "技术文档/API手册/v3.0.pdf",
    "doc_type": "api_doc",       # 文档类型
    "product": "smart_customer",  # 产品线
    "version": "3.0",             # 版本
    "language": "zh",             # 语言
    "team": "platform",           # 所属团队
    "created_at": "2025-01-15",   # 创建时间
    "access_level": "internal",   # 权限等级
    "chunk_index": 5,             # 块序号
    "total_chunks": 12,           # 总块数
}
```

### 7.2 元数据过滤操作符

```python
# ── 精确匹配 ──
collection.query(
    query_texts=["...？"],
    n_results=5,
    where={"doc_type": "api_doc"},  # doc_type == "api_doc"
)

# ── 比较操作 ──
where={"version": {"$gte": "3.0"}}     # version >= "3.0"
where={"created_at": {"$lt": "2025-06-01"}}  # created_at < "2025-06-01"
where={"chunk_index": {"$ne": 0}}      # chunk_index != 0

# ── 逻辑组合 ──
where={                                 # AND
    "$and": [
        {"doc_type": "api_doc"},
        {"version": {"$gte": "3.0"}},
    ]
}

where={                                 # OR
    "$or": [
        {"doc_type": "api_doc"},
        {"doc_type": "tutorial"},
    ]
}

# ── 列表匹配 ──
where={"product": {"$in": ["smart_customer", "smart_office"]}}

# ── 文档内容过滤（全文搜索） ──
where_document={"$contains": "Python"}       # 包含关键词
where_document={"$not_contains": "deprecated"}  # 不包含
```

### 7.3 元数据索引优化

```python
# 高频过滤字段建议建立索引
# ChromaDB 0.5+ 支持元数据索引配置
collection = client.create_collection(
    name="docs",
    metadata={
        "hnsw:space": "cosine",
        # 为 doc_type 和 product 创建倒排索引
    },
)
# 注: 具体索引配置随版本变化，详见官方文档
```

---

## 8. 多 Collection 管理

### 8.1 企业级 Collection 架构

```
chroma_db/
├── product_docs/          # 产品文档
│   ├── star_docs/         # StarDocs 产品
│   ├── star_meet/         # StarMeet 产品
│   └── star_flow/         # StarFlow 产品
├── internal_wiki/         # 内部 Wiki
│   ├── engineering/       # 工程文档
│   ├── hr_policy/         # HR 制度
│   └── onboarding/        # 新人指南
├── customer_faq/          # 客户 FAQ
└── code_base/             # 代码库
```

### 8.2 跨 Collection 搜索

```python
class MultiCollectionSearch:
    """跨多个 Collection 搜索并合并结果"""

    def __init__(self, client: chromadb.PersistentClient):
        self.client = client

    def search(self, query: str, collections: list[str],
               n_results: int = 5) -> list[dict]:
        all_results = []

        for col_name in collections:
            try:
                col = self.client.get_collection(col_name)
                results = col.query(
                    query_texts=[query],
                    n_results=n_results,
                )
                for i, doc_id in enumerate(results["ids"][0]):
                    all_results.append({
                        "collection": col_name,
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "score": 1 - results["distances"][0][i],
                    })
            except Exception:
                continue  # Collection 不存在就跳过

        # 按分数合并排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:n_results]


# 使用
search = MultiCollectionSearch(client)
results = search.search(
    "如何配置 API 网关",
    collections=["engineering", "product_docs/star_docs"],
)
```

### 8.3 迁移与备份

```python
import json
import shutil
from pathlib import Path

# ── 备份: 复制整个数据库目录 ──
def backup(source: str, target: str):
    """冷备份 — 简单可靠"""
    shutil.copytree(source, target)
    print(f"✅ 已备份到 {target}")

# ── 导出（跨平台迁移） ──
def export_collection(collection, output_path: str):
    """导出为 JSON（不含向量，只导出文本 + 元数据）"""
    data = collection.get(include=["documents", "metadatas"])
    records = []
    for i, (doc_id, doc, meta) in enumerate(zip(
        data["ids"], data["documents"], data["metadatas"]
    )):
        records.append({
            "id": doc_id,
            "document": doc,
            "metadata": meta,
        })
    Path(output_path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ── 恢复 ──
def import_collection(collection, input_path: str):
    """从 JSON 恢复（需要重新 Embedding）"""
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    ids = [r["id"] for r in records]
    docs = [r["document"] for r in records]
    metas = [r["metadata"] for r in records]
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
```

---

## 9. 性能优化

### 9.1 批量操作

```python
# ❌ 逐条添加 — 每次 API 调用只加 1 条
for i, doc in enumerate(docs):
    collection.add(ids=[f"doc_{i}"], documents=[doc])  # 极慢！

# ✅ 批量添加 — 一次 API 调用加 100 条
BATCH_SIZE = 100
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    ids = [f"doc_{i + j}" for j in range(len(batch))]
    collection.add(ids=ids, documents=batch)
```

### 9.2 HNSW 参数调优

```python
collection = client.create_collection(
    name="docs",
    metadata={
        # M: 每个节点的最大连接数 (默认 16)
        #   值越大 → 搜索越精确 → 内存和构建时间越大
        #   推荐: 小数据集(1k~10k) M=16, 大数据集(100k+) M=32~64
        "hnsw:M": 32,

        # construction_ef: 构建时搜索宽度 (默认 100)
        #   值越大 → 索引越精确 → 构建越慢
        #   推荐: 100~200 (构建是一次性的，可以大一些)
        "hnsw:construction_ef": 200,

        # search_ef: 搜索时搜索宽度 (默认 10)
        #   值越大 → 搜索越精确 → 搜索越慢
        #   推荐: 10~100 (权衡精度和延迟)
        "hnsw:search_ef": 50,

        # 批处理大小
        "hnsw:batch_size": 100,

        # 是否同步构建（默认 False）
        "hnsw:sync_threshold": 1000,
    },
)
```

### 9.3 嵌入缓存

```python
import hashlib
from functools import lru_cache

class CachedEmbedder:
    """避免重复 Embedding（API 调用很贵）"""

    def __init__(self):
        self._cache = {}

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        new_texts = []
        new_hashes = []
        results = []

        for text in texts:
            h = self._hash(text)
            if h in self._cache:
                results.append(self._cache[h])
            else:
                new_texts.append(text)
                new_hashes.append(h)
                results.append(None)  # 占位

        if new_texts:
            embeddings = get_embeddings_from_api(new_texts)
            for h, emb in zip(new_hashes, embeddings):
                self._cache[h] = emb

            # 填回占位
            emb_idx = 0
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = embeddings[emb_idx]
                    emb_idx += 1

        return results
```

### 9.4 分块策略优化

```python
# 分块大小对检索质量影响很大

# 太小的块 — 信息碎片化，容易丢上下文
CHUNK_SIZE = 100   # ❌ "函数名为 process_" → 截断了

# 太大的块 — 检索精度下降，噪音多
CHUNK_SIZE = 3000  # ❌ 一整章内容，和 query 相关性被稀释

# 推荐范围
CHUNK_SIZE_RECOMMENDED = {
    "QA对": 100,     # 一问一答，短小精悍
    "API文档": 500,  # 一个函数/接口
    "技术文章": 800, # 一个完整知识点
    "长文档": 1500,  # 一个完整章节
}

# Overlap — 10-20% 的 chunk_size
CHUNK_OVERLAP = 100  # 当 chunk_size=500 时
```

---

## 10. 生产环境部署

### 10.1 部署模式选择

| 模式 | 架构 | 适用场景 | 最大数据量 |
|------|------|----------|-----------|
| 嵌入式 | App 内嵌 ChromaDB | 原型、工具脚本 | ~100K 文档 |
| 单机服务 | ChromaDB Server | 小团队、内部工具 | ~1M 文档 |
| 读写分离 | 1 写 + N 读（手动） | 中型应用 | ~5M 文档 |
| 分布式 | 迁移至 Milvus / Qdrant | 大型企业 | 10M+ 文档 |

### 10.2 ChromaDB Server 部署

```bash
# Docker 部署 ChromaDB 服务
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v /data/chroma:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e PERSIST_DIRECTORY=/chroma/chroma \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest

# 验证
curl http://localhost:8000/api/v2/heartbeat
```

```python
# Python 客户端连接远程服务
client = chromadb.HttpClient(
    host="chromadb.internal.company.com",
    port=8000,
    ssl=True,
    headers={"X-Api-Key": "your-api-key"},
)
```

### 10.3 生产配置模板

```python
import chromadb
from pathlib import Path

class ProductionKnowledgeBase:
    """生产级知识库配置模板"""

    def __init__(self, name: str, base_path: str = "/data/chroma"):
        self.path = Path(base_path) / name
        self.path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=False,          # 禁止 reset
                is_persistent=True,
            ),
        )

    def get_collection(self, name: str, dimension: int = 1536):
        return self.client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 50,
                "dimension": dimension,
            },
        )
```

### 10.4 健康检查 & 连接池

```python
import time
from typing import Optional

class ChromaDBPool:
    """简易连接池 — ChromaDB 本身是线程安全的，这里主要做健康检查"""

    def __init__(self, path: str, max_retries: int = 3):
        self.path = path
        self.max_retries = max_retries
        self._client: Optional[chromadb.PersistentClient] = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        for attempt in range(self.max_retries):
            try:
                client = chromadb.PersistentClient(path=self.path)
                client.heartbeat()  # 验证连接
                return client
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def health_check(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False
```

---

## 11. 监控与运维

### 11.1 关键指标

```python
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ChromaDBMetrics:
    """需要监控的核心指标"""
    collection_name: str
    document_count: int = 0
    avg_query_latency_ms: float = 0
    p99_query_latency_ms: float = 0
    index_size_bytes: int = 0
    last_error: Optional[str] = None


class MonitoredCollection:
    """带监控的 Collection 包装器"""

    def __init__(self, collection, metrics: ChromaDBMetrics):
        self._col = collection
        self._metrics = metrics
        self._latencies = []

    def query_with_metrics(self, query_texts: list[str], **kwargs):
        start = time.time()
        try:
            result = self._col.query(query_texts=query_texts, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            self._record_latency(elapsed_ms)
            return result
        except Exception as e:
            self._metrics.last_error = str(e)
            raise

    def _record_latency(self, ms: float):
        self._latencies.append(ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        self._metrics.avg_query_latency_ms = sum(self._latencies) / len(self._latencies)
        self._metrics.p99_query_latency_ms = sorted(self._latencies)[int(len(self._latencies) * 0.99)]

    def refresh_counts(self):
        self._metrics.document_count = self._col.count()
```

### 11.2 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chromadb")

class LoggedCollection:
    """带日志的 Collection"""

    def __init__(self, collection):
        self._col = collection

    def add(self, ids, documents, metadatas, embeddings=None):
        logger.info(f"添加 {len(ids)} 条文档")
        try:
            self._col.add(ids=ids, documents=documents,
                          metadatas=metadatas, embeddings=embeddings)
            logger.info(f"添加成功，当前总数: {self._col.count()}")
        except Exception as e:
            logger.error(f"添加失败: {e}")
            raise

    def query(self, query_texts, n_results, **kwargs):
        logger.info(f"查询: '{query_texts[0][:80]}...' (top {n_results})")
        start = time.time()
        result = self._col.query(query_texts=query_texts,
                                 n_results=n_results, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.info(f"查询完成，耗时 {elapsed:.1f}ms，返回 {len(result['ids'][0])} 条")
        return result
```

---

## 12. 与其他向量数据库对比

### 12.1 功能对比

| 特性 | ChromaDB | FAISS | Qdrant | Milvus | LanceDB |
|------|----------|-------|--------|--------|---------|
| 部署方式 | 嵌入式 | 嵌入式 | Server | Server | 嵌入式 |
| 持久化 | SQLite | 无(需自建) | RocksDB | MinIO/S3 | Lance |
| 元数据过滤 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 全文搜索 | FTS5 | ❌ | ✅ | ❌ | ✅ |
| 分布式 | ❌ | ✅(GPU) | ✅ | ✅ | ❌ |
| Python API | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 最适合 | 学习/原型 | 批量检索 | 生产服务 | 大规模生产 | 数据分析 |

### 12.2 何时升级

```
ChromaDB 够用的场景:          需要迁移到 Milvus / Qdrant:
  - 文档量 < 100 万              - 文档量 > 100 万
  - 单机部署                     - 需要集群 / 高可用
  - QPS < 100                   - QPS > 1000
  - 团队 < 20 人                 - 多团队共享
  - 不需要 GPU 加速              - 需要 GPU 索引加速
```

### 12.3 迁移路径

```python
# ChromaDB → Qdrant 迁移脚本框架
def migrate_chroma_to_qdrant(chroma_collection, qdrant_client, collection_name: str):
    """ChromaDB 数据迁移到 Qdrant"""
    from qdrant_client.models import PointStruct

    # 分批读取 ChromaDB 数据
    BATCH = 500
    total = chroma_collection.count()

    for offset in range(0, total, BATCH):
        data = chroma_collection.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        points = []
        for i, (doc_id, doc, meta, emb) in enumerate(zip(
            data["ids"], data["documents"],
            data["metadatas"], data["embeddings"]
        )):
            points.append(PointStruct(
                id=doc_id,
                vector=emb,
                payload={
                    "document": doc,
                    **meta,
                },
            ))

        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"  已迁移 {min(offset + BATCH, total)}/{total}")
```

---

## 13. 常见问题与反模式

### 13.1 常见错误

```python
# ❌ 错误1: 不检查 Collection 是否存在
collection = client.get_collection("missing")  # 抛异常

# ✅ 正确: 使用 get_or_create
collection = client.get_or_create_collection("my_col")

# ────────────────────────────────────────────

# ❌ 错误2: ID 重复导致数据被覆盖
collection.add(ids=["doc_1"], documents=["文本A"])
collection.add(ids=["doc_1"], documents=["文本B"])  # 覆盖了 A！

# ✅ 正确: 用有意义且唯一的 ID
import uuid
collection.add(ids=[str(uuid.uuid4())], documents=[...])
# 或者
collection.upsert(ids=["doc_v2_001"], documents=[...])  # 明确 upsert 意图

# ────────────────────────────────────────────

# ❌ 错误3: 在循环中逐条 add
for doc in docs:
    collection.add(ids=[...], documents=[doc])  # 极慢

# ✅ 正确: 批量操作
collection.add(ids=ids, documents=docs)  # 快 100 倍

# ────────────────────────────────────────────

# ❌ 错误4: get() 不传 limit
all_data = collection.get()  # 可能返回几十万条，OOM！

# ✅ 正确: 始终带 limit
data = collection.get(limit=100)
# 或只获取 ID
ids = collection.get(include=[])["ids"]

# ────────────────────────────────────────────

# ❌ 错误5: 混合使用不同 Embedding 模型
# 第一次用 DashScope 写入，第二次用 ChromaDB 内置模型查询
# 向量空间完全不对齐，检索结果 = 随机

# ✅ 正确: 整个生命周期使用同一个 Embedding 方案
```

### 13.2 反模式清单

| 反模式 | 问题 | 正确做法 |
|--------|------|----------|
| 所有文档放一个 Collection | 检索慢，管理乱 | 按领域 / 权限拆分 |
| chunk_size 拍脑袋 | 检索质量差 | 根据文档类型选择 |
| 不做元数据过滤 | 噪音多 | 先过滤再搜索 |
| Top-K 固定 | 浪费 Token 或漏信息 | 根据查询复杂度动态调整 |
| 不缓存 Embedding | 重复调用 API | 用 LRU 缓存 |
| 生产用 EphemeralClient | 重启丢数据 | 用 PersistentClient |
| 忘记 delete_collection | 磁盘满了 | 定期清理过期数据 |

---

## 14. 企业级完整示例

### 14.1 带完整生命周期的知识库服务

```python
"""
企业级知识库服务 — 可直接用于生产环境

功能:
  - 多 Collection 管理
  - 文档摄入（支持批量）
  - 语义检索 + 过滤
  - RAG 问答
  - 元数据管理
  - 健康检查
  - 备份恢复
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)


class EnterpriseKnowledgeBase:
    """
    企业级知识库服务

    使用示例:
        kb = EnterpriseKnowledgeBase("/data/knowledge_base")
        col = kb.get_or_create_collection("产品文档-v3")
        col.ingest_pdfs("/data/docs/product/")
        result = col.ask("如何配置 API 网关？", filter_by={"access": "public"})
    """

    def __init__(self, base_path: str = "/data/chroma"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.base_path),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=False,
                is_persistent=True,
            ),
        )
        self._embedder = None  # 注入你的 Embedding 实现

    def set_embedder(self, embedder):
        """注入自定义 Embedding 函数"""
        self._embedder = embedder

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 1536,
        hnsw_M: int = 32,
        hnsw_ef_construct: int = 200,
        hnsw_ef_search: int = 50,
        description: str = None,
        owner: str = None,
    ) -> "ManagedCollection":
        """创建或获取一个托管 Collection"""
        col = self.client.get_or_create_collection(
            name=name,
            embedding_function=self._embedder,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": hnsw_M,
                "hnsw:construction_ef": hnsw_ef_construct,
                "hnsw:search_ef": hnsw_ef_search,
                "dimension": dimension,
                "description": description or "",
                "owner": owner or "",
                "created_at": datetime.now().isoformat(),
            },
        )
        return ManagedCollection(col)

    def list_collections(self) -> list[dict]:
        """列出所有 Collection 及状态"""
        result = []
        for col in self.client.list_collections():
            result.append({
                "name": col.name,
                "document_count": col.count(),
                "metadata": col.metadata,
            })
        return result

    def delete_collection(self, name: str, confirm: bool = False):
        """删除 Collection（需确认）"""
        if not confirm:
            raise ValueError("删除操作需要 confirm=True")
        self.client.delete_collection(name)
        logger.warning(f"已删除 Collection: {name}")

    def backup(self, target_dir: str):
        """备份整个数据库"""
        import shutil
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base_path, target / f"backup_{datetime.now():%Y%m%d_%H%M%S}")
        logger.info(f"备份完成: {target}")

    def health(self) -> dict:
        """健康检查"""
        try:
            self.client.heartbeat()
            return {
                "status": "healthy",
                "collections": len(self.list_collections()),
                "db_path": str(self.base_path),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class ManagedCollection:
    """托管 Collection — 封装最佳实践"""

    def __init__(self, collection):
        self._col = collection
        self._query_count = 0
        self._error_count = 0
        self._total_query_time = 0.0

    # ── 写入 ──

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] = None,
        ids: list[str] = None,
        embeddings: list[list[float]] = None,
    ):
        """添加文档（自动生成 ID、记录时间）"""
        ids = ids or [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]
        metadatas = metadatas or [{}] * len(documents)

        for meta in metadatas:
            meta.setdefault("ingested_at", datetime.now().isoformat())

        self._col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"添加 {len(documents)} 条，总数: {self._col.count()}")

    def upsert_documents(
        self,
        documents: list[str],
        metadatas: list[dict] = None,
        ids: list[str] = None,
    ):
        """更新或插入文档"""
        ids = ids or [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]
        metadatas = metadatas or [{}] * len(documents)

        for meta in metadatas:
            meta["updated_at"] = datetime.now().isoformat()

        self._col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ── 查询 ──

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_by: dict = None,
        min_similarity: float = 0.7,
    ) -> list[dict]:
        """
        语义搜索。

        参数:
          query: 查询文本
          top_k: 返回数量
          filter_by: 元数据过滤条件 {"team": "platform", "version": "3.0"}
          min_similarity: 最小相似度阈值（低于此值不返回）

        返回: [{"id": str, "document": str, "metadata": dict, "score": float}, ...]
        """
        start = time.time()

        kwargs = {"n_results": top_k}
        if filter_by:
            kwargs["where"] = filter_by

        results = self._col.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_by,
        )

        elapsed = time.time() - start
        self._query_count += 1
        self._total_query_time += elapsed

        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]
                if score < min_similarity:
                    continue
                docs.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(score, 4),
                })

        logger.debug(
            f"查询完成: '{query[:50]}...' → {len(docs)} 条, "
            f"{elapsed*1000:.1f}ms"
        )
        return docs

    # ── RAG 问答 ──

    def ask(
        self,
        question: str,
        llm_generate,  # Callable[[str, str], str] — (prompt, context) → answer
        top_k: int = 3,
        filter_by: dict = None,
    ) -> dict:
        """
        RAG 问答: 检索 → 拼入 prompt → LLM 生成

        返回: {"answer": str, "sources": list[dict]}
        """
        sources = self.search(question, top_k=top_k, filter_by=filter_by)

        if not sources:
            return {
                "answer": "未在知识库中找到相关信息。",
                "sources": [],
            }

        context = "\n\n---\n\n".join(
            f"[来源 {i+1}] {s['document']}"
            for i, s in enumerate(sources)
        )

        answer = llm_generate(question, context)
        return {"answer": answer, "sources": sources}

    # ── 统计 ──

    @property
    def stats(self) -> dict:
        return {
            "name": self._col.name,
            "document_count": self._col.count(),
            "query_count": self._query_count,
            "error_count": self._error_count,
            "avg_query_ms": (
                self._total_query_time / self._query_count * 1000
                if self._query_count > 0 else 0
            ),
        }

    def count(self) -> int:
        return self._col.count()

    # ── 删除 ──

    def delete_by_filter(self, filter_by: dict):
        """按元数据过滤删除"""
        results = self._col.get(where=filter_by)
        if results["ids"]:
            self._col.delete(ids=results["ids"])
            logger.info(f"删除 {len(results['ids'])} 条")

    def delete_expired(self, days: int = 90):
        """删除 N 天前创建的文档"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        results = self._col.get(
            where={"ingested_at": {"$lt": cutoff}},
        )
        if results["ids"]:
            self._col.delete(ids=results["ids"])
            logger.info(f"清理过期数据: {len(results['ids'])} 条")
```

### 14.2 使用示例

```python
# ── 初始化 ──
kb = EnterpriseKnowledgeBase("/data/enterprise_kb")

# 注入 Embedding（企业常用 DashScope / OpenAI / 自建服务）
from your_project.embedding import DashScopeEmbedder
kb.set_embedder(DashScopeEmbedder())

# ── 创建知识库 ──
tech_docs = kb.get_or_create_collection(
    name="技术文档-v3.0",
    description="智能客服系统 3.0 技术文档",
    owner="platform_team",
)

# ── 摄入文档 ──
tech_docs.add_documents(
    documents=["API 网关配置说明...", "数据库迁移指南...", "部署手册..."],
    metadatas=[
        {"category": "api", "version": "3.0", "access": "internal"},
        {"category": "database", "version": "3.0", "access": "internal"},
        {"category": "deploy", "version": "3.0", "access": "public"},
    ],
)

# ── 语义搜索 ──
results = tech_docs.search(
    query="如何部署到 Kubernetes？",
    top_k=5,
    filter_by={"access": "public"},  # 只搜公开文档
    min_similarity=0.7,               # 过滤低相关结果
)

for r in results:
    print(f"[{r['score']:.2f}] {r['document'][:80]}...")

# ── RAG 问答 ──
def my_llm(question: str, context: str) -> str:
    # 调用你的 LLM 服务
    ...

answer = tech_docs.ask(
    question="生产环境如何扩容？",
    llm_generate=my_llm,
    filter_by={"category": "deploy"},
)
print(answer["answer"])
for src in answer["sources"]:
    print(f"  参考: {src['metadata']}")

# ── 监控 ──
print(tech_docs.stats)
# {'name': '技术文档-v3.0', 'document_count': 1523, 'query_count': 42,
#  'error_count': 0, 'avg_query_ms': 12.5}

# ── 维护 ──
tech_docs.delete_expired(days=180)  # 清理半年前的文档

# ── 备份 ──
kb.backup("/backup/kb/")
```

---

## 附录: 快速参考卡片

```python
# 最常用的 10 个操作

# 1. 连接
client = chromadb.PersistentClient(path="./db")

# 2. 创建/获取
col = client.get_or_create_collection("name")

# 3. 添加
col.add(ids=["id"], documents=["text"], metadatas=[{"key": "val"}])

# 4. 更新
col.upsert(ids=["id"], documents=["new text"])

# 5. 语义搜索
col.query(query_texts=["query"], n_results=5)

# 6. 精确查找
col.get(ids=["id"]) / col.get(where={"key": "val"})

# 7. 计数
col.count()

# 8. 删除
col.delete(ids=["id"])

# 9. 查看
col.peek(limit=5)

# 10. 列表
client.list_collections()
```

---

> **学习建议**: 先用本文第 3 节「5 分钟上手」跑通一次，再用第 14 节「企业级示例」搭建你的第一个生产知识库。
