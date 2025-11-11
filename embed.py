from langchain_community.document_loaders.csv_loader import CSVLoader
import csv
from uuid import uuid4
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction,Milvus
from langchain_ollama import OllamaEmbeddings

class EMBED_DATA:
    '''初始化数据库'''
    URI = "http://localhost:19530"
    vectorstore = Milvus(
        embedding_function=OllamaEmbeddings(model='qwen3-embedding:8b'),
        connection_args={"uri": URI, "token": "root:Milvus", "db_name": "robot"},#robot
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
        consistency_level="Strong",
        drop_old=False,
    )
    def __init__(self,path='merged_df.csv'):
        self.path = path

    def embed_data(self):
        '''将数据嵌入到milvus'''
        #1,加载文档
        doc_list = []
        with open(self.path, newline='') as f:
            reader = csv.reader(f)
            next(reader)#去除首行
            for row in reader:
                qwestion_id = row[0]
                qwestion_data = row[1]
                ans_id = row[2]
                ans_data = row[3]
                #划分数据
                document = Document(
                    page_content=qwestion_data,
                    metadata={"id": qwestion_id, "ans_id": ans_id, "ans_data": ans_data},
                )
                doc_list.append(document)
        uuids = [str(uuid4()) for _ in range(len(doc_list))]
        self.vectorstore.add_documents(documents=doc_list, ids=uuids)



if __name__ == "__main__":
    '''先运行这个，进行嵌入'''
    run = EMBED_DATA()
    run.embed_data()
