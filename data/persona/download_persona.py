from datasets import load_dataset
import os
import json

# 1. 加载并处理数据
ds = load_dataset("yxsllgz-uts/persona-domain")

agents = []

for idx, agent in enumerate(ds['train']):
    agent["id"] = idx
    agents.append(agent)

# print(ds['train'][0])

# 2. 确保输出目录存在
output_path = "/home/minzhong/Data/SOSEMNLP/data/persona/agents.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 3. 保存为单一 JSON 数组
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(agents, f, ensure_ascii=False, indent=2)

print(f"已保存到 {output_path}")
