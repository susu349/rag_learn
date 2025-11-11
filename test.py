from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", db_name="robot")
# 列出 robot 数据库中的所有集合
print(client.list_collections())