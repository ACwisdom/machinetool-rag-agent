from py2neo import Graph, Node, Relationship, NodeMatcher
import json
import os
import time

# ===================== 核心配置（必须改这里！） =====================
# 1. 替换成你 Neo4j Browser 登录的真实密码！！！
# 示例：如果你的密码是 123456，就改成 NEO4J_PASSWORD = "123456"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]  # 重点：改这里！改这里！改这里！

# ===================== 连接数据库（增加重试机制） =====================
graph = None
retry_times = 3  # 连接失败重试3次
for i in range(retry_times):
    try:
        # py2neo 2021.2.3 标准连接写法
        graph = Graph(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # 测试连接
        node_count = graph.run("MATCH (n) RETURN count(n) AS count").data()[0]["count"]
        print(f"✅ 数据库连接成功！当前节点数：{node_count}")
        break
    except Exception as e:
        print(f"❌ 第{i+1}次连接失败：{str(e)}")
        time.sleep(1)  # 重试前等待1秒
        if i == retry_times - 1:
            print("\n🔍 最终排查步骤：")
            print("1. 密码错误：确认是 Neo4j 登录密码，不是实例名 machine123")
            print("2. 端口被占：执行 netstat -ano | findstr :7687 杀占用进程")
            print("3. 版本问题：确保 py2neo 版本是 2021.2.3")
            exit(1)

# ===================== 工具函数（避免重复创建节点） =====================
def create_node_if_not_exists(label, name):
    """创建节点（如果不存在），返回节点对象"""
    matcher = NodeMatcher(graph)
    node = matcher.match(label, name=name).first()
    if not node:
        node = Node(label, name=name)
        graph.create(node)
    return node

# ===================== 创建基础节点和关系 =====================
# 1. 创建节点
machine1 = create_node_if_not_exists("Machine", "DMG MORI CMX 50 U")
machine2 = create_node_if_not_exists("Machine", "MAZAK VCN-530C")
spindle = create_node_if_not_exists("Part", "主轴")
bearing = create_node_if_not_exists("Part", "轴承")
tool_change_mechanism = create_node_if_not_exists("Part", "换刀机构")
wear_fault = create_node_if_not_exists("Fault", "主轴轴承磨损")
lubricant_lack_fault = create_node_if_not_exists("Fault", "换刀机构润滑油不足")
abnormal_noise_symptom = create_node_if_not_exists("Symptom", "主轴异响，运转时伴随不规则噪音")
tool_jam_symptom = create_node_if_not_exists("Symptom", "刀库换刀卡顿")
replace_bearing_sol = create_node_if_not_exists("Solution", "更换主轴轴承并重新校准")
add_lubricant_sol = create_node_if_not_exists("Solution", "添加专用润滑油并清洁导轨")

# 2. 创建关系（兼容旧版 py2neo）
def create_relationship_if_not_exists(start_node, rel_type, end_node):
    """创建关系（如果不存在）"""
    # 先查询关系是否存在
    query = f"""
    MATCH (a)-[r:{rel_type}]->(b)
    WHERE id(a) = {start_node.identity} AND id(b) = {end_node.identity}
    RETURN count(r) AS rel_count
    """
    rel_count = graph.run(query).data()[0]["rel_count"]
    if rel_count == 0:
        # 关系不存在则创建
        rel = Relationship(start_node, rel_type, end_node)
        graph.create(rel)

# 机床-包含-部件
create_relationship_if_not_exists(machine1, "包含", spindle)
create_relationship_if_not_exists(spindle, "包含", bearing)
create_relationship_if_not_exists(machine2, "包含", tool_change_mechanism)

# 部件-发生-故障
create_relationship_if_not_exists(bearing, "发生", wear_fault)
create_relationship_if_not_exists(tool_change_mechanism, "发生", lubricant_lack_fault)

# 故障-表现为-症状
create_relationship_if_not_exists(wear_fault, "表现为", abnormal_noise_symptom)
create_relationship_if_not_exists(lubricant_lack_fault, "表现为", tool_jam_symptom)

# 故障-对应-解决方案
create_relationship_if_not_exists(wear_fault, "对应", replace_bearing_sol)
create_relationship_if_not_exists(lubricant_lack_fault, "对应", add_lubricant_sol)

# ===================== 导入历史工单数据（容错处理） =====================
work_order_file = "历史工单数据.json"
if not os.path.exists(work_order_file):
    print(f"⚠️  未找到工单文件：{work_order_file}，跳过导入")
else:
    try:
        with open(work_order_file, "r", encoding="utf-8") as f:
            work_orders = json.load(f)
        for idx, order in enumerate(work_orders):
            order_node = Node(
                "WorkOrder",
                id=idx + 1,
                故障现象=order.get("故障现象", ""),
                根本原因=order.get("根本原因", ""),
                解决方案=order.get("解决方案", ""),
                处理时间=order.get("处理时间", "未记录")
            )
            graph.create(order_node)
            # 关联故障
            fault_name = order.get("根本原因")
            if fault_name:
                fault_node = NodeMatcher(graph).match("Fault", name=fault_name).first()
                if fault_node:
                    create_relationship_if_not_exists(order_node, "关联故障", fault_node)
            # 关联解决方案
            solution_name = order.get("解决方案")
            if solution_name:
                solution_node = NodeMatcher(graph).match("Solution", name=solution_name).first()
                if solution_node:
                    create_relationship_if_not_exists(order_node, "采用方案", solution_node)
        print("✅ 工单数据导入完成！")
    except Exception as e:
        print(f"⚠️  工单导入失败：{e}")

# ===================== 测试查询 =====================
def get_fault_solutions(fault_name):
    query = "MATCH (f:Fault {name: $name})-[r:对应]->(s:Solution) RETURN s.name AS solution"
    result = graph.run(query, name=fault_name).data()
    return [item["solution"] for item in result]

def diagnose_by_symptom(symptom_name):
    query = """
    MATCH (s:Symptom {name: $name})<-[:表现为]-(f:Fault)-[:对应]->(sol:Solution)
    RETURN f.name AS fault, sol.name AS solution
    """
    result = graph.run(query, name=symptom_name).data()
    return result

# 运行测试
if __name__ == "__main__":
    # 测试故障查询
    wear_solutions = get_fault_solutions("主轴轴承磨损")
    print("\n🔧 主轴轴承磨损的解决方案：", wear_solutions)
    # 测试症状诊断
    abnormal_noise_diagnosis = diagnose_by_symptom("主轴异响，运转时伴随不规则噪音")
    print("🔍 主轴异响的诊断结果：", abnormal_noise_diagnosis)
    print("\n🎉 所有操作执行完成！")