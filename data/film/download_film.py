from datasets import load_dataset
import os
import json

# 1. 加载并处理数据
ds = load_dataset("yxsllgz-uts/imdb-2025-more")

ds = ds.filter(lambda example: example['votes'] >= 30 and example['rating'] is not None)
ds = ds.map(
    lambda example: {
        'initial_avg': example['rating'],
        'initial_raters': example['votes']
    },
    remove_columns=['rating', 'votes']
)

movies = []

for idx, movie in enumerate(ds['train']):
    movie["id"] = idx
    movies.append(movie)
    movie["title"]=movie["title"].split()[1]

# 2. 确保输出目录存在
output_path = "/home/minzhong/Data/SOS/data/film/movies.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 3. 保存为单一 JSON 数组
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(movies, f, ensure_ascii=False, indent=2)

print(f"已保存到 {output_path},共 {len(movies)} 部电影")
