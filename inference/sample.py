# 从persona数据集中采样
import random
from collections import defaultdict
from collections import Counter

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
    domain_counts = Counter([item["domain"] for item in persona_data])
    
    # 筛选出数量大于等于阈值的domain
    valid_domains = {domain for domain, count in domain_counts.items() if count >= min_count}
    
    # 筛选出属于valid_domains的persona
    filtered_data = [item for item in persona_data if item["domain"] in valid_domains]
    
    print(f"筛选后的domain数量: {len(valid_domains)}，数据总量: {len(filtered_data)}")
    return filtered_data

def threshold_based_sampling(persona_data, sample_size=100, min_count=50):
    """
    排除稀有职业的随机采样
    
    Args:
        persona_data: 包含persona信息的列表
        min_count: 职业数量阈值，默认为50
        sample_size: 采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """
    # 先筛选domain数量大于等于50的数据
    filtered_data = filter_domains_by_count(persona_data, min_count)
    
    # 从筛选后的数据中随机采样
    if len(filtered_data) <= sample_size:
        sampled_data = filtered_data
    else:
        sampled_data = random.sample(filtered_data, sample_size)
    
    print(f"阈值采样完成，采样数量: {len(sampled_data)}")
    return sampled_data

def threshold_based_stratified_sampling(persona_data, sample_size=100, min_count=50):
    """
    分层采样（根据"domain"对应的角色）
    
    Args:
        persona_data: 包含persona信息的列表
        sample_size: 采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """
    # 先筛选domain数量大于等于50的数据
    filtered_data = filter_domains_by_count(persona_data, min_count)
    
    # 按domain分组
    domain_groups = defaultdict(list)
    for item in filtered_data:
        domain = item["domain"]
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

def threshold_based_uniform_per_class_sampling(persona_data, sample_size=100, min_count=50, samples_per_class=2):
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
    filtered_data = filter_domains_by_count(persona_data, min_count)
    
    # 按domain分组
    domain_groups = defaultdict(list)
    for item in filtered_data:
        domain = item["domain"]
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

def sampling(data, sample_size=100):
    """
    随机采样
    
    Args:
        data: 包含persona信息的列表
        sample_size: 采样数量，默认为100
        
    Returns:
        sampled_data: 采样后的persona列表
    """

    sampled_data = random.sample(data, sample_size)
    
    print(f"阈值采样完成，采样数量: {len(sampled_data)}")
    return sampled_data