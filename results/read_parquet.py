import pandas as pd
import os

# 读取 Parquet 文件
parquet_file = '/home/minzhong/Data/SOSEMNLP/results/DeepSeek-V2-Lite/ratings_no_history.parquet'
df = pd.read_parquet(parquet_file)

# 显示数据
print(df)

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))