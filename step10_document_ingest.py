"""
Step 10: 文档摄入 — 将 PDF / Markdown / TXT 存入知识库

学习目标:
  1. 读取不同格式文档并提取纯文本
  2. 理解文档分块策略对不同格式的影响
  3. 将真实文件导入向量数据库作为知识库
  4. 跨文档语义搜索

支持格式:
  - PDF  (.pdf) — 需要 pip install pymupdf
  - Markdown (.md) — 内置支持
  - 文本 (.txt) — 内置支持

运行: python step10_document_ingest.py
前置: pip install chromadb requests pymupdf
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import chromadb
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("❌ 未找到 API Key！请检查 .env 文件")
    sys.exit(1)

llm_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 10


# ============================================================
# 1. 文档加载器 — 支持 PDF / Markdown / TXT
# ============================================================

class DocumentLoader:
    """从不同格式文件中提取文本"""

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本: 移除多余空行、统一换行"""
        text = re.sub(r'\n{3,}', '\n\n', text)      # 3个以上换行→2个
        text = re.sub(r'[ \t]{3,}', '  ', text)     # 多余空格
        text = re.sub(r'\x00', '', text)             # null 字符
        return text.strip()

    @staticmethod
    def load_pdf(filepath: str) -> str:
        """从 PDF 提取文本 (pymupdf)"""
        import fitz  # pymupdf
        doc = fitz.open(filepath)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return DocumentLoader._clean_text("\n\n".join(pages))

    @staticmethod
    def load_markdown(filepath: str) -> str:
        """
        读取 Markdown 文件。

        保留原始 Markdown 格式，因为:
        - 标题层级 (# ## ###) 有助于分块时保持语义
        - 代码块 (```) 适合按段落切分
        - 列表和表格保留结构信息
        """
        text = Path(filepath).read_text(encoding="utf-8")
        return DocumentLoader._clean_text(text)

    @staticmethod
    def load_text(filepath: str) -> str:
        """读取纯文本文件"""
        text = Path(filepath).read_text(encoding="utf-8")
        return DocumentLoader._clean_text(text)

    @classmethod
    def load_file(cls, filepath: str) -> dict:
        """
        自动识别文件类型并加载。

        返回: {"path": str, "type": str, "text": str, "size": int}
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        handlers = {
            ".pdf": cls.load_pdf,
            ".md": cls.load_markdown,
            ".txt": cls.load_text,
            ".py": cls.load_text,      # 代码文件当纯文本
            ".js": cls.load_text,
            ".ts": cls.load_text,
        }

        handler = handlers.get(suffix)
        if not handler:
            raise ValueError(f"不支持的文件格式: {suffix}。支持: {list(handlers.keys())}")

        text = handler(str(path))
        return {
            "path": str(path),
            "type": suffix,
            "filename": path.name,
            "text": text,
            "size": len(text),
        }

    @classmethod
    def load_directory(cls, directory: str, recursive: bool = True) -> list[dict]:
        """
        加载目录下所有支持的文档。

        参数:
          directory: 目录路径
          recursive: 是否递归子目录

        返回: [{"path": str, "type": str, "text": str, ...}, ...]
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"  ⚠️ 目录不存在: {directory}")
            return []

        supported = {".pdf", ".md", ".txt", ".py", ".js", ".ts"}
        pattern = "**/*" if recursive else "*"
        docs = []

        for filepath in dir_path.glob(pattern):
            if filepath.suffix.lower() in supported and filepath.is_file():
                try:
                    doc = cls.load_file(str(filepath))
                    docs.append(doc)
                    print(f"  ✅ {filepath.name} ({doc['size']} 字符)")
                except Exception as e:
                    print(f"  ⚠️ 加载失败 {filepath.name}: {e}")

        return docs


# ============================================================
# 2. 文档分块
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """递归字符分割: 段落 → 句子 → 定长"""
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

    # 级别 2: 按句子切
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
# 3. Embedding API
# ============================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """DashScope text-embedding-v2"""
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
# 4. 知识库 — ChromaDB 封装
# ============================================================

class KnowledgeBase:
    """基于 ChromaDB 的文档知识库"""

    def __init__(self, name: str = "documents"):
        persist_dir = str(Path(__file__).parent / "chroma_db" / name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(self, docs: list[dict]):
        """
        批量摄入文档。

        流程: 提取文本 → 分块 → 向量化 → 存入 ChromaDB
        """
        all_chunks = []
        all_metadatas = []

        for doc in docs:
            chunks = chunk_text(doc["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": doc.get("filename", ""),
                    "filepath": doc.get("path", ""),
                    "filetype": doc.get("type", ""),
                    "chunk_size": len(chunk),
                })

        if not all_chunks:
            print("  ⚠️ 没有可摄入的内容")
            return

        print(f"  📄 {len(docs)} 个文件 → {len(all_chunks)} 个文本块")
        print(f"  📐 向量化中...")
        embeddings = get_embeddings(all_chunks)
        ids = [f"doc_{i}" for i in range(len(all_chunks))]

        self.collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )
        print(f"  ✅ 已存入: {self.collection.count()} 条记录")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """语义搜索，返回最相关的文档块"""
        query_emb = get_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                retrieved.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(1 - distance, 4),
                })
        return retrieved

    def ask(self, question: str, top_k: int = TOP_K, verbose: bool = True) -> str:
        """RAG 问答: 检索 + LLM 生成"""
        sources = self.search(question, top_k=top_k)

        if verbose:
            print(f"\n🔍 检索到 {len(sources)} 个相关片段:")
            for i, s in enumerate(sources):
                preview = s["content"][:100].replace("\n", " ")
                fn = s["metadata"].get("filename", "?")
                print(f"  [{i+1}] {fn} | 相似度={s['similarity']:.3f} | {preview}...")

        context = "\n\n---\n\n".join([
            f"[来源: {s['metadata'].get('filename', '未知')}]\n{s['content']}"
            for s in sources
        ])

        prompt = f"""根据以下参考资料回答问题。只使用资料中的信息，不要编造。
如果资料中没有，请说"未在文档中找到相关信息"。

参考资料:
{context}

问题: {question}"""

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("📥 文档摄入 — PDF / Markdown / TXT → 知识库")
    print("=" * 60)

    # -------- 创建示例文档 --------
    print("\n🏗️ 创建示例文档...")
    sample_dir = Path(__file__).parent / "sample_docs"
    sample_dir.mkdir(exist_ok=True)

    # 示例 Markdown — 项目文档
    (sample_dir / "project_overview.md").write_text("""# 智能客服系统 3.0

## 项目概述
智能客服系统 3.0 是基于大语言模型的新一代客服平台，由 AI 平台部开发。

## 核心功能
- **智能会话**: 支持多轮对话，上下文记忆，意图识别准确率 95%
- **知识库检索**: 基于 RAG 技术，自动匹配 FAQ 和内部文档
- **人工转接**: 复杂问题无缝转接人工，会话上下文完整传递
- **数据分析**: 实时统计响应时间、解决率、用户满意度

## 技术架构
- 前端: React 18 + TypeScript + Ant Design 5
- 后端: Go 1.22 + gRPC + Gin
- AI 引擎: Qwen-Max + 自研 RAG 框架
- 数据: PostgreSQL 16 + Redis Cluster + Kafka
- 部署: Kubernetes (ACK) + ArgoCD

## 上线时间线
- 2025 年 1 月: 内部灰度测试，500 人
- 2025 年 3 月: 第一批客户上线，日活 5000+
- 2025 年 6 月: 全量发布，日活突破 5 万
- 2025 年 Q3 计划: 多语言支持（英、日、印尼语）

## 定价策略
- 标准版: ￥2999/月，10 坐席，基础功能
- 专业版: ￥8999/月，30 坐席，含 RAG 知识库
- 企业版: ￥29999/月，无限坐席，私有化部署 + 定制模型
""", encoding="utf-8")

    # 示例 TXT
    (sample_dir / "faq.txt").write_text("""智能客服系统常见问题 FAQ

Q1: 如何接入微信小程序？
A: 在管理后台 → 渠道管理 → 微信小程序 → 填入 AppID 和 AppSecret → 保存后自动激活。接入后支持文字和语音消息。

Q2: 知识库支持什么格式？
A: 支持 PDF、Word、Markdown、TXT、HTML。上传后系统自动解析并建立索引，约 3 分钟生效。

Q3: 对话记录保留多久？
A: 标准版保留 30 天，专业版 90 天，企业版永久保留。支持导出为 CSV 或 JSON。

Q4: 是否支持自定义回复话术？
A: 支持。在知识库 → 预设回复中配置。支持变量替换（如 {用户名}、{订单号}）。

Q5: 系统部署在哪儿？
A: 标准版/专业版部署在阿里云杭州。企业版支持私有化部署在客户自有服务器。

Q6: SLA 保障是多少？
A: 标准版 99.5%，专业版 99.9%，企业版 99.95%。超过 SLA 按比例赔付。

Q7: 如何联系技术支持？
A: 在线客服: 工作日 9:00-18:00。紧急热线: 400-888-1234（7×24 小时）。企业版配备专属技术支持经理。
""", encoding="utf-8")

    print(f"  已创建 {len(list(sample_dir.glob('*')))} 个示例文档\n")

    # -------- 加载文档 --------
    print("📂 加载文档...")
    docs = DocumentLoader.load_directory(str(sample_dir))
    print(f"  共加载 {len(docs)} 个文档\n")

    # -------- 摄入知识库 --------
    print("💾 摄入知识库...")
    kb = KnowledgeBase(name="sample_docs")
    kb.ingest_documents(docs)

    # -------- 检索演示 --------
    print("\n" + "=" * 60)
    print("🔬 检索演示")
    print("=" * 60)

    questions = [
        "智能客服系统的定价是怎样的？",
        "如何接入微信小程序？",
        "系统 SLA 保障是多少？",
        "技术架构用了哪些组件？",
    ]

    for q in questions:
        print(f"\n{'─'*50}")
        print(f"❓ {q}")
        answer = kb.ask(q, top_k=2, verbose=False)
        print(f"📚 {answer}")

    # -------- 交互模式 --------
    print("\n" + "=" * 60)
    print("💬 交互问答")
    print("=" * 60)
    print("对文档提问。试试: '知识库支持什么格式？' '什么时候全量发布的？'")
    print("输入 'quit' 退出\n")

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

        answer = kb.ask(user_input, verbose=True)
        print(f"\n🤖 {answer}\n")


if __name__ == "__main__":
    main()
