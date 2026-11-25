import json
from py2neo import Graph, Node

# 连接Neo4j数据库（替换为实际的连接信息）
graph = Graph("bolt://localhost:7687", auth=("neo4j", "machine123"))

with open("历史工单数据.json", "r", encoding="utf-8") as f:
    work_orders = json.load(f)
for order in work_orders:
    order_node = Node("WorkOrder", 故障现象=order["故障现象"], 处理时间=order["处理时间"])
    graph.create(order_node)