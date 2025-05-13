#!/usr/bin/env python3
"""
Complete script: Use OpenAI GPT API for multi-agent movie rating experiments
"""
import os
import json
import random
import statistics
import re
import argparse
import copy

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from openai import OpenAI
import logging

# Suppress verbose OpenAI logs
logging.getLogger("openai").setLevel(logging.WARNING)

client = OpenAI()

# ------------------- Functions -------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Movie rating experiment with OpenAI GPT API"
    )
    parser.add_argument(
        '--openai_model', type=str, default='gpt-4o-mini',
        help="OpenAI model name, e.g., gpt-3.5-turbo or gpt-4"
    )
    parser.add_argument(
        '--sample_type', type=str, default='base',
        help="Sampling type: base, stratified, uniform"
    )
    parser.add_argument(
        '--num_movies', type=int, default=100,
        help="Number of movies to process"
    )
    parser.add_argument(
        '--agents_per_movie', type=int, default=100,
        help="Number of agents per movie"
    )
    parser.add_argument(
        '--rate_num', type=int, default=3,
        help="Number of calls per rating when no history is available"
    )
    parser.add_argument(
        '--min_count', type=int, default=50,
        help="Minimum count for threshold-based sampling"
    )
    return parser.parse_args()

# GPT API call, returns {'rating': int}
def call_gpt(prompt: str, fallback_rating: int, model: str) -> dict:
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if 'rating' not in parsed or not isinstance(parsed['rating'], (int, float)):
                parsed['rating'] = fallback_rating
            return parsed
    except Exception as e:
        print("call_gpt error:", e)
    return {"rating": fallback_rating}

# Aggregate responses by taking rounded mean
def aggregate_responses(responses):
    valid = [r['rating'] for r in responses if isinstance(r.get('rating'), (int, float))]
    if not valid:
        return {"rating": 0}
    return {"rating": round(statistics.mean(valid))}

# Prompt templates
def prompt_a(persona, movie):
    return f"""Please provide your rating for the movie.

# Your Character Profile:
You are {persona}

# Movie Information:
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}

# Output:
Only output a JSON object: {{"rating": <integer between 1 and 10>}}"""

def prompt_b(persona, movie, avg_rating):
    return f"""Please provide your rating for the movie.

# Your Character Profile:
You are {persona}

# Movie Information:
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}
Current average rating: {avg_rating:.1f}

# Output:
Only output a JSON object: {{"rating": <integer between 1 and 10>}}"""

def prompt_c(movie):
    return f"""Please provide your rating for the movie.

# Movie Information:
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}

# Output:
Only output a JSON object: {{"rating": <integer between 1 and 10>}}"""

def prompt_d(movie, avg_rating):
    return f"""Please provide your rating for the movie.

# Movie Information:
Title: {movie['title']}
Genres: {', '.join(movie['genres'])}
Overview: {movie['overview']}
Current average rating: {avg_rating:.1f}

# Output:
Only output a JSON object: {{"rating": <integer between 1 and 10>}}"""

# Sampling functions (import from sample.py)
from sample import (
    threshold_based_sampling,
    threshold_based_stratified_sampling,
    threshold_based_uniform_per_class_sampling,
    sampling,
)

def initial_agents(path, sample_type, n, min_count):
    with open(path, 'r', encoding='utf-8') as f:
        agents = json.load(f)
    if sample_type == 'base':
        selected = threshold_based_sampling(agents, n, min_count=min_count)
    elif sample_type == 'stratified':
        selected = threshold_based_stratified_sampling(agents, n, min_count=min_count)
    else:
        selected = threshold_based_uniform_per_class_sampling(agents, n, min_count=min_count)
    random.shuffle(selected)
    return selected

def initial_movies(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        movies = json.load(f)
    for m in movies:
        m['initial_avg'] = float(m['initial_avg'])
        m['initial_raters'] = int(m['initial_raters'])
    selected = sampling(movies, n)
    random.shuffle(selected)
    return selected

# Core rating logic
def rate_movie_both(movie, agents, rate_n, model):
    # Deep copies for persona and no-persona flows
    mp = copy.deepcopy(movie)
    mn = copy.deepcopy(movie)
    # Initialize scores and counts
    t_sp = mp['initial_avg'] * mp['initial_raters']
    t_cp = mp['initial_raters']
    t_sn = mn['initial_avg'] * mn['initial_raters']
    t_cn = mn['initial_raters']

    res = ([], [], [], [])  # psn_no, psn_hist, nps_no, nps_hist
    for agent in tqdm(agents, desc=f"Rating {movie['title'][:15]:15}", unit="agent"):
        # persona, no history
        p1 = prompt_a(agent['persona'], mp)
        fb1 = round(t_sp / t_cp)
        r1 = aggregate_responses([call_gpt(p1, fb1, model) for _ in range(rate_n)])
        res[0].append({'movie_id': mp['id'], 'agent_id': agent['id'], 'rating': r1['rating']})
        # persona, with history
        avg_p = t_sp / t_cp
        p2 = prompt_b(agent['persona'], mp, avg_p)
        fb2 = round(avg_p)
        r2 = aggregate_responses([call_gpt(p2, fb2, model) for _ in range(3)])
        t_sp += r2['rating']; t_cp += 1
        res[1].append({'movie_id': mp['id'], 'agent_id': agent['id'], 'rating': r2['rating'], 'current_history_avg': round(t_sp/t_cp,1)})
        mp['initial_avg'] = round(t_sp/t_cp,1)
        # no persona, no history
        p3 = prompt_c(mn)
        fb3 = round(t_sn / t_cn)
        r3 = aggregate_responses([call_gpt(p3, fb3, model) for _ in range(rate_n)])
        res[2].append({'movie_id': mn['id'], 'agent_id': agent['id'], 'rating': r3['rating']})
        # no persona, with history
        avg_n = t_sn / t_cn
        p4 = prompt_d(mn, avg_n)
        fb4 = round(avg_n)
        r4 = aggregate_responses([call_gpt(p4, fb4, model) for _ in range(3)])
        t_sn += r4['rating']; t_cn += 1
        res[3].append({'movie_id': mn['id'], 'agent_id': agent['id'], 'rating': r4['rating'], 'current_history_avg': round(t_sn/t_cn,1)})
        mn['initial_avg'] = round(t_sn/t_cn,1)
    return res

# Run full experiment, return four dicts
def run_full_experiment(args):
    base = os.path.dirname(os.path.abspath(__file__))
    agents_path = os.path.join(os.path.dirname(base), 'data', 'persona', 'agents.json')
    movies_path = os.path.join(os.path.dirname(base), 'data', 'film', 'movies.json')

    movies = initial_movies(movies_path, args.num_movies)
    out = ({}, {}, {}, {})
    for mv in tqdm(movies, desc="Movies", unit="movie"):
        agents = initial_agents(agents_path, args.sample_type, args.agents_per_movie, args.min_count)
        psn_no, psn_hist, nps_no, nps_hist = rate_movie_both(
            mv, agents, args.rate_num, args.openai_model
        )
        out[0][mv['title']] = psn_no
        out[1][mv['title']] = psn_hist
        out[2][mv['title']] = nps_no
        out[3][mv['title']] = nps_hist
    return out

# Save results dict to Parquet
def save_parquet(results, filepath):
    rows = []
    for title, ratings in results.items():
        for r in ratings:
            rows.append({'movie_title': title, **r})
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)

# ------------------- Main -------------------
def main():
    args = parse_args()
    results = run_full_experiment(args)

    base = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(base), 'results', args.sample_type, args.openai_model.replace('/', '_'))
    os.makedirs(save_dir, exist_ok=True)

    names = ['ratings_no_history.parquet', 'ratings_with_history.parquet',
             'ratings_no_persona_no_history.parquet', 'ratings_no_persona_with_history.parquet']
    for res, fname in zip(results, names):
        save_parquet(res, os.path.join(save_dir, fname))

    print("Experiment completed. Results saved to:", save_dir)

if __name__ == "__main__":
    main()
