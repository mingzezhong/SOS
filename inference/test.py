import json, random, statistics
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
import json
import os
import re

from sample import threshold_based_sampling
from sample import threshold_based_stratified_sampling
from sample import threshold_based_uniform_per_class_sampling
from sample import sampling

from vllm import LLM, SamplingParams

agents_input_path = "/home/minzhong/Data/SOSEMNLP/data/persona/agents.json"
movies_input_path = "/home/minzhong/Data/SOSEMNLP/data/film/movies.json"

# vLLM 引擎初始化（同你现有配置）
engine = LLM(
    model="mistralai/Ministral-8B-Instruct-2410",
    tokenizer="mistralai/Ministral-8B-Instruct-2410",
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
    repetition_penalty=1.0, # 不惩罚
)

def call_vllm(prompt: str) -> dict:
    # 1) 生成列表 outputs
    outputs = engine.generate([prompt], 
                              sampling_params)
    # 2) 取第一个 RequestOutput
    out = outputs[0]
    # 3) 取第一个生成文本
    text = out.outputs[0].text.strip()

    print("\n===== Raw model output =====")
    print(text)
    print("===== End of model output =====\n")

    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print("⚠️ JSON decode error:", e)
    else:
        print("⚠️ No JSON structure found in model output:", text)

    


def prompt_a():
    return """Based on your character profile and the movie information, please provide your rating for the movie.

# Your Character Profile: 
You are A software developer who is looking for a way to simplify the integration of GPRS technology 
into their embedded system designs. They are interested in developing a stable and efficient 
software stack for an embedded system and are willing to invest time and effort into finding 
a solution that meets their requirements. They are looking for a product that is easy to use 
and has minimal requirements for technical knowledge, while also being able to provide accurate 
and reliable data transmission. They are also interested in finding a product that is compatible 
with other network protocols and can be easily integrated into existing systems.


# Movie Information
Title: Thunderbolts
Genres: Political Drama, Superhero, Action, Adventure
Overview: After finding themselves ensnared in a death trap, an unconventional team of antiheroes 
must go on a dangerous mission that will force them to confront the darkest corners of their pasts.

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 10 means you found the movie outstanding in all aspects.
- 1 means you found the movie poor in every aspect.

# Output Principle
Now, you have rated the above movie with rating R, and please fill in the value R into the JSON object below.
Only output this JSON object—no extra explanation or content:

# Output
{{"rating": <integer between 1 and 10>}}
"""


def prompt_b():
    return """Based on your character profile and the movie information, please provide your rating for the movie.

# Your Character Profile: 
You are A software developer who is looking for a way to simplify the integration of GPRS technology 
into their embedded system designs. They are interested in developing a stable and efficient 
software stack for an embedded system and are willing to invest time and effort into finding 
a solution that meets their requirements. They are looking for a product that is easy to use 
and has minimal requirements for technical knowledge, while also being able to provide accurate 
and reliable data transmission. They are also interested in finding a product that is compatible 
with other network protocols and can be easily integrated into existing systems.

# Movie Information
Title: Thunderbolts
Genres: Political Drama, Superhero, Action, Adventure
  Overview: After finding themselves ensnared in a death trap, an unconventional team of antiheroes 
must go on a dangerous mission that will force them to confront the darkest corners of their pasts.
Movie average rating: 7.7 

# Rating Principle
Now, please you rate the above movie on an integer rating R scale from 1 to 10, where:
- 10 means you found the movie outstanding in all aspects.
- 1 means you found the movie poor in every aspect.

# Output Principle
Now, you have rated the above movie with rating R, and please fill in the value R into the JSON object below.
Only output this JSON object—no extra explanation or content:

# Output
{{"rating": <integer between 1 and 10>}}
"""

call_vllm(prompt_a())
