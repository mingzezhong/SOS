1. conda env create -f environment.yml
2. enter the path: /SOS/your-usernam/inference
3. select a huggingface model name from file "model_selection.text"
4. bash run: python main.py --hf_model_name slected_hf_model_name

example:

a. python main.py --hf_model_name deepseek-ai/DeepSeek-V2-Lite-Chat

b. python main.py --hf_model_name deepseek-ai/DeepSeek-V2-Lite-Chat --sample_type base

default: --sample_type base

  base: Random sampling of agents excluding rare professions
  stratified: Stratified sampling of agents excluding rare professions
  uniform: Uniform sampling of agents excluding rare professions

c. python main.py --hf_model_name deepseek-ai/DeepSeek-V2-Lite --sample_type stratified --num_movies 5 --agents_per_movie 15 --rate_num 5 --min_count 100

default:
--num_movies 5: The number of movies sampled randomly is 100
--agents_per_movie 15: The number of agents rating each movie is 100
--rate_num 5: To avoid randomness, each agent rates each movie 3 times
--min_count 100: Exclude agents whose number in the corresponding domain is less than 50

Here is the translation:



