import json
import random
import statistics
import re
import os
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import logging
logging.getLogger("vllm").setLevel(logging.WARNING)

from vllm import LLM, SamplingParams

from sample import (
    threshold_based_sampling,
    threshold_based_stratified_sampling,
    threshold_based_uniform_per_class_sampling,
    sampling,
)

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
    trust_remote_code=True,
    device="cuda",
    dtype="auto",
    max_num_seqs=8,
    max_num_batched_tokens=4096,
)

sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.1,
    top_p=1.0,
    repetition_penalty=1.0,
)

def call_vllm(prompt: str, fallback_rating: int = None) -> dict:
    """
    使用 vLLM 生成并解析 JSON 输出。
    解析失败时，返回 {"rating": fallback_rating}。
    """
    outputs = engine.generate([prompt], sampling_params)
    out = outputs[0]
    text = out.outputs[0].text.strip()

    # print("\n===== Raw model output =====")
    # print(text)
    # print("===== End of model output =====\n")

    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    else:
        print("No JSON structure found in model output.")

    # 解析失败时返回 fallback
    return {"rating": fallback_rating} if fallback_rating is not None else {"rating": 0}


def aggregate_responses(responses, visibility=False):
    scores = [resp["rating"] for resp in responses]
    avg_score = round(statistics.mean(scores))
    result = {"rating": avg_score}
    if visibility:
        result["visibility"] = statistics.mode([resp.get("visibility", "hide") for resp in responses])
    return result


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
Now, you have rated the above movie with rating R, and please fill in the value R into the JSON object below.
Only output this JSON object—no extra explanation or content:

# Output
{{"rating": <integer between 1 and 10>}}
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
Now, you have rated the above movie with rating R, and please fill in the value R into the JSON object below.
Only output this JSON object—no extra explanation or content:

# Output
{{"rating": <integer between 1 and 10>}}
"""

def intial_agents(n):
    with open(agents_input_path, "r", encoding="utf-8") as f:
        agents = json.load(f)
    if args.sample_type == 'base':
        return threshold_based_sampling(agents, n, min_count=args.min_count)
    elif args.sample_type == 'stratified':
        return threshold_based_stratified_sampling(agents, n, min_count=args.min_count)
    elif args.sample_type == 'uniform':
        return threshold_based_uniform_per_class_sampling(agents, n, min_count=args.min_count)

def intial_movies(n):
    with open(movies_input_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    # 转换类型
    for m in movies:
        m["initial_avg"] = float(m["initial_avg"])
        m["initial_raters"] = int(m["initial_raters"])
    return sampling(movies, n)


def rate_movie(movie, agents, n):
    no_hist_results, with_hist_results = [], []
    total_score = movie["initial_avg"] * movie["initial_raters"]
    total_raters = movie["initial_raters"]

    # 内层：agent 循环
    for agent in tqdm(
        agents,
        desc=f"Rating {movie['title'][:15]:15}",  # 限制影片名长度避免太长
        unit="agent",
        dynamic_ncols=True,
        leave=False,       # 完成后清除此进度条
        position=1         # 内层进度条在 position 1
    ):
        # 无历史评分
        prompt_no_hist = prompt_a(agent["persona"], movie)
        fallback_no = round(total_score / total_raters)
        responses_no_hist = [
            call_vllm(prompt_no_hist, fallback_rating=fallback_no)
            for _ in range(n)
        ]
        agg_no_hist = aggregate_responses(responses_no_hist)

        # 有历史评分
        current_avg = total_score / total_raters
        prompt_with_hist = prompt_b(agent["persona"], movie, current_avg)
        fallback_with = round(current_avg)
        responses_with_hist = [
            call_vllm(prompt_with_hist, fallback_rating=fallback_with)
            for _ in range(3)
        ]
        agg_with_hist = aggregate_responses(responses_with_hist, visibility=True)

        # 更新历史
        total_score += agg_with_hist["rating"]
        total_raters += 1

        no_hist_results.append({
            "movie_id": movie["id"],
            "agent_id": agent["id"],
            "rating": agg_no_hist["rating"],
        })
        with_hist_results.append({
            "movie_id": movie["id"],
            "agent_id": agent["id"],
            "rating": agg_with_hist["rating"],
            "visibility": agg_with_hist["visibility"],
            "current_history_avg": round(total_score / total_raters, 1),
        })

        movie["initial_avg"] = round(total_score / total_raters, 1)

    return no_hist_results, with_hist_results


def run_full_experiment(num_movies=3, agents_per_movie=10, rate_num=3):
    movies = intial_movies(num_movies)
    results_no_history, results_with_history = {}, {}

    # 外层：电影循环
    for movie in tqdm(
        movies,
        desc="Movies ",
        unit="movie",
        dynamic_ncols=True,
        leave=True,        # 完成后保留此进度条
        position=0         # 最外层进度条在 position 0
    ):
        agents = intial_agents(agents_per_movie)
        no_hist, with_hist = rate_movie(movie, agents, rate_num)
        results_no_history[movie["title"]] = no_hist
        results_with_history[movie["title"]] = with_hist

    return results_no_history, results_with_history


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
    save_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/results/{model_name}"
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    res_no_hist, res_with_hist = run_full_experiment(
        num_movies=args.num_movies, agents_per_movie=args.agents_per_movie, rate_num=args.rate_num
    )

    save_parquet(res_no_hist, os.path.join(save_path, "ratings_no_history.parquet"))
    save_parquet(res_with_hist, os.path.join(save_path, "ratings_with_history.parquet"))

    print("实验完成，结果已保存为 Parquet 文件。")
