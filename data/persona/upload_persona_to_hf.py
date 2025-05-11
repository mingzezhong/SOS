# import json
# import os

# # 读取persona.json文件
# input_file_path = "/home/minzhong/Data/SOSEMNLP/data/persona/persona.json"
# output_file_path = "/home/minzhong/Data/SOSEMNLP/data/persona/person_domain.json"

# # 确保输出目录存在
# os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# # 读取原始数据
# with open(input_file_path, 'r', encoding='utf-8') as f:
#     personas = json.load(f)

# # 提取需要的属性并重命名
# simplified_personas = []
# for persona in personas:
#     simplified_persona = {
#         "persona": persona["persona"],
#         "domain": persona["general domain (top 1 percent)"]
#     }
#     simplified_personas.append(simplified_persona)

# # 保存为新的JSON文件
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     json.dump(simplified_personas, f, ensure_ascii=False, indent=4)

# print(f"处理完成，已保存到 {output_file_path}")

from huggingface_hub import HfApi, HfFolder

api = HfApi()
token = HfFolder.get_token()

api.create_repo(
    repo_id="yxsllgz-uts/persona-domain",
    repo_type="dataset",
    exist_ok=True,
    token=token
)

api.upload_file(
    path_or_fileobj="/home/minzhong/Data/SOSEMNLP/data/persona/person_domain.json",
    path_in_repo="persona_domain.json",
    repo_id="yxsllgz-uts/persona-domain",
    repo_type="dataset",
    token=token,
)

print("✅ 数据集上传完成，请前往 https://huggingface.co/datasets/your-username/persona-domain 查看")
