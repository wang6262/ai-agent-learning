"""
ChromaDB 内容查看工具

用法:
  python inspect_chromadb.py           # 列出所有集合
  python inspect_chromadb.py -s        # 按相似度排序（需要输入查询）
  python inspect_chromadb.py --export  # 导出为 JSON
"""
import sys, json, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import chromadb
from pathlib import Path

DB_PATH = Path(__file__).parent / "chroma_db"

def main():
    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}")
        print("  请先运行 step08_rag_basic.py 或 step09_rag_agent.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collections = client.list_collections()
    print(collections)

    print(f"📂 数据库: {DB_PATH}")
    print(f"📊 集合数: {len(collections)}\n")

    export_mode = "--export" in sys.argv

    for col in collections:
        print(f"{'='*60}")
        print(f"🗂️  集合: {col.name} | 文档数: {col.count()} | 元数据: {col.metadata}")
        print(f"{'='*60}")
        print(col)

        if export_mode:
            # 导出全部
            data = col.get(include=["documents", "metadatas"])
            export = []
            for i, (doc_id, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
                export.append({"index": i, "id": doc_id, "source": meta.get("title", ""), "content": doc})
            output_path = Path(__file__).parent / f"{col.name}_export.json"
            output_path.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ 已导出 {len(export)} 条到 {output_path}")
        else:
            data = col.peek(limit=20)
            for i, (doc_id, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
                print(f"  [{i}] {meta.get('title', 'N/A')} ({meta.get('chunk_size', '?')} 字符)")
                print(f"      ID: {doc_id}")
                print(f"      {doc[:200].replace(chr(10), ' ')}...")
                print()

        if col.count() > 20:
            print(f"  ... 共 {col.count()} 条，仅显示前 20 条。用 --export 导出全部")

if __name__ == "__main__":
    main()
