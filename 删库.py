# 重建集合（指定余弦相似度）
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

# 删除旧集合
client = MilvusClient(uri="http://localhost:19530", db_name="robot")
if client.has_collection("LangChainCollection"):
    client.drop_collection("LangChainCollection")
    print("已删除旧集合，准备重建")

# 初始化新集合（使用余弦相似度）
# vectorstore = Milvus(
#     embedding_function=OllamaEmbeddings(model='bge-large-zh-v1.5'),  # 中文优化模型
#     collection_name="LangChainCollection",
#     vector_field="vector",
#     connection_args={"uri": "http://localhost:19530", "db_name": "robot"},
#     index_params={
#         "index_type": "FLAT",  # 适合小规模数据，精确匹配
#         "metric_type": "COSINE"  # 关键：指定余弦相似度
#     },
#     drop_old=True,  # 重建集合
# )

# 重新插入数据（参考之前的分批次插入逻辑）
# ...（插入代码略）