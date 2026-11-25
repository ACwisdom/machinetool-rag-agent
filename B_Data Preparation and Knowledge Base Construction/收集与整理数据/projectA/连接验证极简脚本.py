import os

# 极简连接测试脚本（兼容新版 py2neo）
from py2neo import Graph

# 你的配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

try:
    # 新版 py2neo 用 auth 元组
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    node_count = graph.run("MATCH (n) RETURN count(n) AS count").data()[0]["count"]
    print(f"✅ 连接成功！当前数据库节点数：{node_count}")
except Exception as e:
    print(f"❌ 连接失败：{str(e)}")