from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama import OllamaEmbeddings

class BatchBM25VectorRetriever:
    def __init__(self, collection_name="LangChainCollection", db_name="robot", batch_size=100):
        self.collection_name = collection_name
        self.db_name = db_name
        self.batch_size = batch_size
        # 初始化向量存储，指定正确的向量字段名（如 "vector"）
        self.vectorstore = Milvus(
            embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
            builtin_function=BM25BuiltInFunction(),  # 启用 BM25
            vector_field="vector",  # 关键：使用实际的向量字段名（如 "vector"）
            collection_name=collection_name,
            connection_args={"uri": "http://localhost:19530", "db_name": db_name},
            consistency_level="Strong",
            drop_old=False,  # 不重建索引，避免冲突
            # 禁用自动创建索引（如果集合已存在索引）
            index_params=None,
        )

    def bm25_vector_hybrid_search(self, query, k_total=100):
        all_results = []
        offset = 0

        while len(all_results) < k_total:
            current_k = min(self.batch_size, k_total - len(all_results))
            batch_results = self.vectorstore.similarity_search(
                query=query,
                k=current_k,
                offset=offset,
                params={"bm25_weight": 0.3, "vector_weight": 0.7}
            )

            if not batch_results:
                break

            all_results.extend(batch_results)
            offset += current_k
            print(f"已检索 {len(all_results)}/{k_total} 条结果（批次大小：{current_k}）")

        return all_results[:k_total]

if __name__ == "__main__":
    retriever = BatchBM25VectorRetriever(batch_size=30)
    results = retriever.bm25_vector_hybrid_search(
        query="高血压患者能吃甜食吗？",
        k_total=60
    )
    for i, doc in enumerate(results[:5]):
        print(f"第 {i+1} 条：")
        print(f"内容：{doc.page_content}")
        print(f"答案：{doc.metadata.get('ans_data', '未知')}\n")