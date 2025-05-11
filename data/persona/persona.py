from datasets import load_dataset
import os
import json
from collections import Counter


save_dir = "/home/minzhong/Data/SOSEMNLP/data/persona/"

persona_dataset = []

# 获取persona数据

dataset = load_dataset("proj-persona/PersonaHub", 'elite_persona', streaming=True)

for persona in dataset['train']:
    if persona["general domain (top 1 percent)"] != "None":
        persona_dataset.append(persona)

    if len(persona_dataset) == 10000:
        break

# 创建保存路径
os.makedirs(save_dir, exist_ok=True)

# 保存为JSON文件
json_path = os.path.join(save_dir, "persona.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(persona_dataset, f, ensure_ascii=False, indent=4)

print(f"Persona dataset saved to {json_path}")

# 读取persona数据
def load_persona_data(json_path=None):
    """
    读取persona数据集
    
    Args:
        json_path: persona.json文件的路径，如果为None则使用默认路径
        
    Returns:
        persona_data: 包含persona信息的列表
    """
    if json_path is None:
        json_path = os.path.join(save_dir, "persona.json")
    
    if not os.path.exists(json_path):
        print(f"Error: Persona data file not found at {json_path}")
        return []
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            persona_data = json.load(f)
        print(f"Successfully loaded {len(persona_data)} persona entries from {json_path}")
        return persona_data
    except Exception as e:
        print(f"Error loading persona data: {e}")
        return []

load_persona_dataset = load_persona_data()  # 使用默认路径，即 save_dir/persona.json

# 统计general domain (top 1 percent)的种类及数量
def count_general_domains(persona_data):
    """
    统计general domain (top 1 percent)的种类及数量
    
    Args:
        persona_data: 包含persona信息的列表
        
    Returns:
        domain_counts: 包含domain名称和对应数量的字典
    """
    domains = [item["general domain (top 1 percent)"] for item in persona_data]
    domain_counts = Counter(domains)
    
    print("General domain (top 1 percent) 统计结果:")
    for domain, count in domain_counts.most_common():
        print(f"{domain}: {count}")
    
    return domain_counts

# 执行统计
domain_statistics = count_general_domains(load_persona_dataset)

# 从persona数据集中采样
import random
from collections import defaultdict

def filter_domains_by_count(persona_data, min_count=50):
    """
    筛选出domain数量大于等于指定阈值的persona数据
    
    Args:
        persona_data: 包含persona信息的列表
        min_count: domain最小数量阈值，默认为50
        
    Returns:
        filtered_data: 筛选后的persona列表
    """
    # 统计各domain的数量
    domain_counts = Counter([item["general domain (top 1 percent)"] for item in persona_data])
    
    # 筛选出数量大于等于阈值的domain
    valid_domains = {domain for domain, count in domain_counts.items() if count >= min_count}
    
    # 筛选出属于valid_domains的persona
    filtered_data = [item for item in persona_data if item["general domain (top 1 percent)"] in valid_domains]
    
    print(f"筛选后的domain数量: {len(valid_domains)}，数据总量: {len(filtered_data)}")
    return filtered_data

def stratified_sampling(persona_data, sample_size=100):
    """
    分层采样（根据"general domain (top 1 percent)"对应的角色）
    
    Args:
        persona_data: 包含persona信息的列表
        sample_size: 采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """
    # 先筛选domain数量大于等于50的数据
    filtered_data = filter_domains_by_count(persona_data)
    
    # 按domain分组
    domain_groups = defaultdict(list)
    for item in filtered_data:
        domain = item["general domain (top 1 percent)"]
        domain_groups[domain].append(item)
    
    # 计算每个domain应采样的数量
    total_count = len(filtered_data)
    domain_sample_counts = {}
    sampled_data = []
    
    # 首先确保每个domain至少有一个样本
    remaining_size = sample_size
    for domain in domain_groups:
        if remaining_size > 0:
            domain_sample_counts[domain] = 1
            remaining_size -= 1
    
    # 按比例分配剩余的采样数量
    if remaining_size > 0:
        for domain, items in domain_groups.items():
            domain_ratio = len(items) / total_count
            additional_samples = max(0, int(round(domain_ratio * remaining_size)))
            domain_sample_counts[domain] += additional_samples
    
    # 确保总采样数量不超过sample_size
    total_allocated = sum(domain_sample_counts.values())
    if total_allocated > sample_size:
        # 从最大的domain中减少采样
        domains_sorted = sorted(domain_groups.keys(), key=lambda d: domain_sample_counts[d], reverse=True)
        for domain in domains_sorted:
            if total_allocated <= sample_size:
                break
            if domain_sample_counts[domain] > 1:
                domain_sample_counts[domain] -= 1
                total_allocated -= 1
    
    # 从每个domain中采样
    for domain, count in domain_sample_counts.items():
        domain_items = domain_groups[domain]
        if len(domain_items) < count:
            count = len(domain_items)  # 确保不超过该domain的总数
        domain_samples = random.sample(domain_items, count)
        sampled_data.extend(domain_samples)
    
    print(f"分层采样完成，采样数量: {len(sampled_data)}")
    return sampled_data

def threshold_based_sampling(persona_data, threshold=50, sample_size=100):
    """
    排除稀有职业的随机采样
    
    Args:
        persona_data: 包含persona信息的列表
        threshold: 职业数量阈值，默认为50
        sample_size: 采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """
    # 先筛选domain数量大于等于50的数据
    filtered_data = filter_domains_by_count(persona_data, threshold)
    
    # 从筛选后的数据中随机采样
    if len(filtered_data) <= sample_size:
        sampled_data = filtered_data
    else:
        sampled_data = random.sample(filtered_data, sample_size)
    
    print(f"阈值采样完成，采样数量: {len(sampled_data)}")
    return sampled_data

def uniform_per_class_sampling(persona_data, samples_per_class=2, sample_size=100):
    """
    均匀采样各职业
    
    Args:
        persona_data: 包含persona信息的列表
        samples_per_class: 每个职业采样的数量，默认为2
        sample_size: 总采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """
    # 先筛选domain数量大于等于50的数据
    filtered_data = filter_domains_by_count(persona_data)
    
    # 按domain分组
    domain_groups = defaultdict(list)
    for item in filtered_data:
        domain = item["general domain (top 1 percent)"]
        domain_groups[domain].append(item)
    
    sampled_data = []
    domains = list(domain_groups.keys())
    
    # 计算每个domain最多可以采样的数量
    max_samples_per_domain = min(samples_per_class, sample_size // len(domains))
    
    # 从每个domain中采样
    for domain in domains:
        domain_items = domain_groups[domain]
        domain_sample_count = min(max_samples_per_domain, len(domain_items))
        domain_samples = random.sample(domain_items, domain_sample_count)
        sampled_data.extend(domain_samples)
        
        if len(sampled_data) >= sample_size:
            sampled_data = sampled_data[:sample_size]
            break
    
    # 如果采样数量不足，从剩余的persona中随机采样
    if len(sampled_data) < sample_size:
        # 已采样的persona的ID
        sampled_ids = {id(item) for item in sampled_data}
        # 未采样的persona
        remaining_data = [item for item in filtered_data if id(item) not in sampled_ids]
        
        # 从剩余数据中随机采样
        additional_samples = random.sample(
            remaining_data, 
            min(sample_size - len(sampled_data), len(remaining_data))
        )
        sampled_data.extend(additional_samples)
    
    print(f"均匀采样完成，采样数量: {len(sampled_data)}，使用的domain数量: {len(domains)}")
    return sampled_data

# 示例使用
# 执行采样

stratified_samples = stratified_sampling(load_persona_dataset)
threshold_samples = threshold_based_sampling(load_persona_dataset)
uniform_samples = uniform_per_class_sampling(load_persona_dataset)

# 保存采样结果
def save_samples(samples, filename):
    """
    保存采样结果到JSON文件
    
    Args:
        samples: 采样后的persona列表
        filename: 保存的文件名
    """
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print(f"采样结果已保存到 {save_path}，共 {len(samples)} 条数据")

# 保存各种采样方法的结果

save_samples(stratified_samples, "stratified_samples.json")
save_samples(threshold_samples, "threshold_samples.json")
save_samples(uniform_samples, "uniform_samples.json")