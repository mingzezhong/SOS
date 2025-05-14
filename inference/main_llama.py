import json
import random
import statistics
import re
import os
import argparse
import copy
import io, sys


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import logging
logging.getLogger("vllm").setLevel(logging.WARNING)

from vllm import LLM, SamplingParams
from contextlib import redirect_stderr

from sample import (
    threshold_based_sampling,
    threshold_based_stratified_sampling,
    threshold_based_uniform_per_class_sampling,
    sampling,
)

os.environ["HF_HOME"] = f'{os.path.dirname(os.path.abspath(__file__)).split("SOS")[0]}.cache/huggingface/'

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="处理命令行参数")

# 添加命令行参数
parser.add_argument('--hf_model_name', type=str, default='deepseek-ai/DeepSeek-V2-Lite-Chat', help="hf_model_name")
parser.add_argument('--sample_type', type=str, default='base', help="base, stratified, uniform")
parser.add_argument('--num_movies', type=int, default=100, help="num_movies")
parser.add_argument('--agents_per_movie', type=int, default=100, help="agents_per_movie")
parser.add_argument('--rate_num', type=int, default=3, help="rate_num")
parser.add_argument('--min_count', type=int, default=50, help="min_count")


# 解析命令行参数
args = parser.parse_args()

agents_input_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/persona/agents.json"
movies_input_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/film/movies.json"

# vLLM 引擎初始化
engine = LLM(
    model=args.hf_model_name,
    tokenizer=args.hf_model_name,
    dtype = 'float16',
    compilation_config = {
        "use_triton" : True,
        "use_flash_attention" : True
    },
    trust_remote_code=True,
    device="cuda",
    max_num_seqs=8,
    max_num_batched_tokens=4096,
    # 并行配置：
    tensor_parallel_size=2,        # 张量并行 2-way
    pipeline_parallel_size=1,      # 流水线并行 1-way（可调大）
    data_parallel_size=1,          # 数据并行 1-way（可调大）
)


sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.1,
    top_p=1.0,
    repetition_penalty=1.0,
)

import io
import re
import json
from contextlib import redirect_stderr

def call_vllm(prompt: str, fallback_rating: int = None) -> dict:
    """
    使用 vLLM 生成并解析 JSON 输出。
    保证返回格式为 {"rating": int} 的字典，哪怕模型输出异常。
    """

    try:
        # 捕获模型调用过程中 stderr 输出
        buf_err = io.StringIO()
        with redirect_stderr(buf_err):
            outputs = engine.generate([prompt], sampling_params)

        # 抽取模型输出文本
        out = outputs[0]
        if not out.outputs:
            raise ValueError("Model returned no outputs.")
        
        text = out.outputs[0].text.strip()

        # 直接匹配 1–10 的整数输出
        m = re.search(r"\b([1-9]|10)\b", text)
        if m:
            score = int(m.group(1))
            return {"rating": score}
        # else:
        #     print("No integer rating found, raw output:\n", text)
            
    except Exception as e:
        print("Exception during vLLM call:", e)

    # fallback：无论发生什么问题，都保证返回含 rating 的 dict
    return {"rating": fallback_rating or 0}



def aggregate_responses(responses):
    valid_ratings = [
        resp["rating"]
        for resp in responses
        if isinstance(resp.get("rating"), (int, float))
    ]

    if not valid_ratings:
        return {"rating": 0}

    avg_score = round(statistics.mean(valid_ratings))
    return {"rating": avg_score}


def prompt_a(persona, movie):
    return f"""Please provide your rating for the movie.

# Your Character Profile:    
You are {persona}

# Movie Information
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 1 = Awful/Abysmal (unwatchable)
- 5 = Mediocre/Unsure (forgettable)
- 10 = Perfect/Masterpiece (flawless)

# Output Principle
Please provide a single integer score (1-10) for the movie. **Output only the integer**, without any JSON or extra text.
"""


def prompt_b(persona, movie, avg_rating):
    return f"""Please provide your rating for the movie.

# Your Character Profile:    
You are {persona}

# Movie Information
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}
Movie average rating: {avg_rating:.2f} (1-10)

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 1 = Awful/Abysmal (unwatchable)
- 5 = Mediocre/Unsure (forgettable)
- 10 = Perfect/Masterpiece (flawless)

# Output Principle
Please provide a single integer score (1-10) for the movie. **Output only the integer**, without any JSON or extra text.
"""

def prompt_c(movie):
    return f"""Please provide your rating for the movie.

# Movie Information
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 1 = Awful/Abysmal (unwatchable)
- 5 = Mediocre/Unsure (forgettable)
- 10 = Perfect/Masterpiece (flawless)

# Output Principle
Please provide a single integer score (1-10) for the movie. **Output only the integer**, without any JSON or extra text.
"""


def prompt_d(movie, avg_rating):
    return f"""Please provide your rating for the movie.

# Movie Information
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}
Movie average rating: {avg_rating:.2f} (1-10)

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 1 = Awful/Abysmal (unwatchable)
- 5 = Mediocre/Unsure (forgettable)
- 10 = Perfect/Masterpiece (flawless)

# Output Principle
Please provide a single integer score (1-10) for the movie. **Output only the integer**, without any JSON or extra text.
"""


def intial_agents(n):
    with open(agents_input_path, "r", encoding="utf-8") as f:
        agents = json.load(f)

    # 采样，并打乱顺序
    if args.sample_type == 'base':
        _agents = threshold_based_sampling(agents, n, min_count=args.min_count)
        random.shuffle(_agents)
        return _agents
    elif args.sample_type == 'stratified':
        _agents = threshold_based_stratified_sampling(agents, n, min_count=args.min_count)
        random.shuffle(_agents)
        return _agents
    elif args.sample_type == 'uniform':
        _agents = threshold_based_uniform_per_class_sampling(agents, n, min_count=args.min_count)
        random.shuffle(_agents)
        return _agents

def intial_movies(n):
    with open(movies_input_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    # 转换类型
    for m in movies:
        m["initial_avg"] = float(m["initial_avg"])
        m["initial_raters"] = int(m["initial_raters"])
    # 打乱顺序
    _movies = sampling(movies, n)
    random.shuffle(_movies)

    return _movies

def rate_movie_both(movie, agents, n):
    """
    对每个 agent，同步做：
     - psn: 有 persona 的评分 (no_hist + with_hist)
     - no_psn: 无 persona 的评分 (no_hist + with_hist)

    给两者各自 clone 一份状态，互不影响。
    """
    # 深拷贝两份状态
    movie_psn   = copy.deepcopy(movie)
    movie_no_psn = copy.deepcopy(movie)

    # 结果列表
    psn_no_hist, psn_with_hist = [], []
    no_psn_no_hist, no_psn_with_hist = [], []

    # 初始化历史分数
    total_score_psn   = movie_psn["initial_avg"]    * movie_psn["initial_raters"]
    total_raters_psn  = movie_psn["initial_raters"]
    total_score_no_psn  = movie_no_psn["initial_avg"]   * movie_no_psn["initial_raters"]
    total_raters_no_psn = movie_no_psn["initial_raters"]

    bar = tqdm(
        agents,
        desc=f"Rating {movie['title'][:15]:15}",
        unit="agent",
        dynamic_ncols=True,
        leave=True,
        position=1
    )
    for agent in bar:
        #### 有 persona (psn) ####
        # no_hist
        prompt1 = prompt_a(agent["persona"], movie_psn)
        fallback1 = round(total_score_psn / total_raters_psn)
        resp1 = aggregate_responses(
            [call_vllm(prompt1, fallback1) for _ in range(n)]
        )
        psn_no_hist.append({
            "movie_id": movie_psn["id"],
            "agent_id": agent["id"],
            "rating": resp1["rating"],
        })

        # with_hist
        avg_psn = total_score_psn / total_raters_psn
        prompt2 = prompt_b(agent["persona"], movie_psn, avg_psn)
        fallback2 = round(avg_psn)
        resp2 = aggregate_responses(
            [call_vllm(prompt2, fallback2) for _ in range(3)]
        )
        total_score_psn  += resp2["rating"]
        total_raters_psn += 1
        psn_with_hist.append({
            "movie_id": movie_psn["id"],
            "agent_id": agent["id"],
            "rating": resp2["rating"],
            "current_history_avg": round(total_score_psn / total_raters_psn, 1)
        })

        movie_psn["initial_avg"] = round(total_score_psn / total_raters_psn, 1)

        #### 无 persona (no_psn) ####
        # no_hist
        prompt3 = prompt_c(movie_no_psn)
        fallback3 = round(total_score_no_psn / total_raters_no_psn)
        resp3 = aggregate_responses(
            [call_vllm(prompt3, fallback3) for _ in range(n)]
        )
        no_psn_no_hist.append({
            "movie_id": movie_no_psn["id"],
            "agent_id": agent["id"],
            "rating": resp3["rating"],
        })

        # with_hist
        avg_no_psn = total_score_no_psn / total_raters_no_psn
        prompt4 = prompt_d(movie_no_psn, avg_no_psn)
        fallback4 = round(avg_no_psn)
        resp4 = aggregate_responses(
            [call_vllm(prompt4, fallback4) for _ in range(3)]
        )
        total_score_no_psn  += resp4["rating"]
        total_raters_no_psn += 1
        no_psn_with_hist.append({
            "movie_id": movie_no_psn["id"],
            "agent_id": agent["id"],
            "rating": resp4["rating"],
            "current_history_avg": round(total_score_no_psn / total_raters_no_psn, 1)
        })

        movie_no_psn["initial_avg"] = round(total_score_no_psn / total_raters_no_psn, 1)

    return psn_no_hist, psn_with_hist, no_psn_no_hist, no_psn_with_hist


def run_full_experiment(num_movies=3, agents_per_movie=10, rate_num=3):
    movies = intial_movies(num_movies)

    results_psn_no_hist, results_psn_with_hist = {}, {}
    results_no_psn_no_hist, results_no_psn_with_hist = {}, {}

    for movie in tqdm(
        movies,
        desc="Movies     ",
        unit="movie",
        dynamic_ncols=True,
        leave=True,
        position=0
    ):
        agents = intial_agents(agents_per_movie)
        psn_no_hist, psn_with_hist, no_psn_no_hist, no_psn_with_hist = \
            rate_movie_both(movie, agents, rate_num)

        results_psn_no_hist[movie["title"]]     = psn_no_hist
        results_psn_with_hist[movie["title"]]   = psn_with_hist
        results_no_psn_no_hist[movie["title"]]  = no_psn_no_hist
        results_no_psn_with_hist[movie["title"]] = no_psn_with_hist

    return results_psn_no_hist, results_psn_with_hist, results_no_psn_no_hist, results_no_psn_with_hist,

def save_parquet(results, filename):
    rows = []
    for movie_title, ratings in results.items():
        for rating in ratings:
            row = {"movie_title": movie_title, **rating}
            rows.append(row)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filename)


if __name__ == "__main__":

    model_name = args.hf_model_name.split("/")[1]
    save_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/results/{args.sample_type}/{model_name}"
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    res_no_hist, res_with_hist, res_no_psn_no_hist, res_no_psn_with_hist = run_full_experiment(
        num_movies=args.num_movies, agents_per_movie=args.agents_per_movie, rate_num=args.rate_num
    )

    save_parquet(res_no_hist, os.path.join(save_path, "ratings_no_history.parquet"))
    save_parquet(res_with_hist, os.path.join(save_path, "ratings_with_history.parquet"))

    save_parquet(res_no_psn_no_hist, os.path.join(save_path, "ratings_no_persona_no_history.parquet"))
    save_parquet(res_no_psn_with_hist, os.path.join(save_path, "ratings_no_persona_with_history.parquet"))

    print("实验完成，结果已保存为 Parquet 文件。")
