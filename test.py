import os
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI
import data_loader

load_dotenv(Path(__file__).parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


persist_dir = str(Path(__file__).parent / "chroma_db")
client = chromadb.PersistentClient(path=persist_dir)


class AliTextEmbeddingV2(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def __call__(self, texts: Documents) -> Embeddings:
        resp = self.client.embeddings.create(
            input=texts,
            model="text-embedding-v4"
        )
        return [item.embedding for item in resp.data]




ef = AliTextEmbeddingV2(api_key=DASHSCOPE_API_KEY)



input_text = [
    "衣服的质量杠杠的",
    "Chromadb 是一个向量数据库",
    "Python 是一门编程语言",
    "今天晚饭吃了火锅",
]

data = ef(input_text)
print(data)



embedingllm = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = embedingllm.embeddings.create(
    model="text-embedding-v4",
    input=input_text
)

print(completion.data)
print("-"*80)
for item in completion.data:
    print(item)
    print("-" * 80)
    print(item.embedding)



# 多模态图片嵌入函数（支持uris、images数据源）
multimodal_ef = OpenCLIPEmbeddingFunction()
# 绑定图片加载器
image_loader = ImageLoader()
data_loader = data_loader.UniversalFileDataLoader()
kb_tech = client.get_or_create_collection("tech_document",metadata={"hnsw:space":"cosine"},data_loader=image_loader)
kb_img = client.get_or_create_collection("tech_img",metadata={"hnsw:space":"cosine"},data_loader=image_loader,embedding_function=multimodal_ef)
ef_tech = client.get_or_create_collection("ef_tech",metadata={"hnsw:space":"cosine"},embedding_function=ef)





ef_tech.add(
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
# kb_img.add(
#     ids=["img1","img2","img3"],
#     uris=[
#         r"C:\Users\524\Desktop\s2024022710495043.jpg",
#         r"C:\Users\524\Desktop\s2024022710495043.jpg",
#         r"C:\Users\524\Desktop\s2024022710495043.jpg"
#     ],
# )



results1 = kb_tech.query(
    query_texts=["什么是向量数据库？"],
    n_results=5,
)
print(results1)

results2 = ef_tech.query(
    query_texts=["什么是向量数据库？"],
    n_results=5,
)
print(results2)

collections = client.list_collections()
for collection in collections:
    print(collection)
# 只查看第一个集合所有数据
# coll1 = client.get_collection("knowledge_base")
# print(coll1.get())


#
# # 只查看第二个集合所有数据
# coll2 = client.get_collection("agent_kb")
# print(coll2.get())
