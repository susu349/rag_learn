from typing import Iterator, Any

from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings


# 连接到已存在的 Milvus 集合
vectorstore = Milvus(
    embedding_function=OllamaEmbeddings(model='qwen3-embedding:8b'),
    collection_name="LangChainCollection",
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    # 显式指定集合名（之前默认生成的名称）
    connection_args={
        "uri": "http://localhost:19530",
        "db_name": "robot"  # 指定数据库（之前存储数据的数据库）
    },
    consistency_level="Strong",
    drop_old=False,  # 关键：不删除已有数据
)

def query(str)->dict:
    results = vectorstore.similarity_search_with_score(
        query=str['query_content'], k=5
    )
    list=[]
    for res, score in results:
        list.append(f"* [SIM={score:3f}]病人问题为：{res.page_content} 其对应的数据为[{res.metadata}]")
    dict={
        "docs":list,
        "query_content":str
            }
    return dict



if __name__ == '__main__':
    quer = "感冒了饮食要注意什么？"
    query(quer)