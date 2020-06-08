import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm  # tqdm_notebook as tqdm
from transformers import GPT2Tokenizer

from attention_intervention_winobias import get_interventions_winobias
from attention_intervention_winogender import get_interventions_winogender
from attention_utils import perform_interventions
from experiment import Model

np.random.seed(1)
torch.manual_seed(1)

def perform_interventions_single(interventions, model, layers_to_adj, heads_to_adj, effect_types=('indirect', 'direct'), search=False):
    """Perform multiple interventions"""
    results_list = []
    for intervention in tqdm(interventions):
        results = perform_intervention_single(intervention, model, layers_to_adj, heads_to_adj, effect_types, search)
        results_list.append(results)
    return results_list

def perform_intervention_single(intervention, model, layers_to_adj, heads_to_adj, effect_types=('indirect', 'direct'), search=False):
    """Perform intervention and return results for specified effects"""
    x = intervention.base_strings_tok[0]  # E.g. The doctor asked the nurse a question. He
    x_alt = intervention.base_strings_tok[1]  # E.g. The doctor asked the nurse a question. She

    with torch.no_grad():
        candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples_multitoken(
            x,
            intervention.candidates_tok)
        candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples_multitoken(
            x_alt,
            intervention.candidates_tok)

    candidate1 = ' '.join(intervention.candidates[0]).replace('Ġ', '')
    candidate2 = ' '.join(intervention.candidates[1]).replace('Ġ', '')

    odds_base = candidate2_base_prob / candidate1_base_prob
    odds_alt = candidate2_alt_prob / candidate1_alt_prob
    total_effect = (odds_alt - odds_base) / odds_base

    results = {
        'base_string1': intervention.base_strings[0],
        'base_string2': intervention.base_strings[1],
        'candidate1': candidate1,
        'candidate2': candidate2,
        'candidate1_base_prob': candidate1_base_prob,
        'candidate2_base_prob': candidate2_base_prob,
        'odds_base': odds_base,
        'candidate1_alt_prob': candidate1_alt_prob,
        'candidate2_alt_prob': candidate2_alt_prob,
        'odds_alt': odds_alt,
        'total_effect': total_effect,
    }

    for effect_type in effect_types:
        candidate1_probs_head, candidate2_probs_head = model.attention_intervention_single_experiment(
            intervention, effect_type, layers_to_adj, heads_to_adj, search)
        odds_intervention_head = candidate2_probs_head / candidate1_probs_head
        effect_head = (odds_intervention_head - odds_base) / odds_base
        if search:
          results[effect_type + "_effect_head"] = effect_head.tolist()
        else:
          results[effect_type + "_effect_head"] = effect_head
    return results

def top_k(k, interventions, mean_effect, model, model_type, data, out_dir):
	json_data = {'head': [], 'val': []}
	for i in range(1, k+1):

		top_k = i
		idx = np.argpartition(mean_effect, mean_effect.size - top_k, axis=None)[-top_k:]

		# get top k
		res = np.column_stack(np.unravel_index(idx, mean_effect.shape))
		results = perform_interventions_single(interventions, model, layers_to_adj=res[:,0], heads_to_adj=res[:,1])

		df1 = pd.DataFrame(results)
		effect1 = np.stack(df1['indirect_effect_head'].to_numpy())  # Convert column to 2d ndarray (num_examples x num_layers)
		mean_effect1 = effect1.mean(axis=0)

		json_data['val'].append(mean_effect1)
		json_data['head'].append((res[:,0][0], res[:,1][0]))
		pickle.dump(json_data, open(out_dir + "/topk_" + model_type + "_" + data + ".pickle", "wb" ))

def get_all_contrib(model_type, model, tokenizer, interventions, data, out_dir):
	json_data = {}

	results = perform_interventions(interventions, model)
	df = pd.DataFrame(results)
	effect = np.stack(df['indirect_effect_model'].to_numpy())  # Convert column to 2d ndarray (num_examples x num_layers)
	mean_effect = effect.mean(axis=0)
	json_data['mean_effect_model'] = mean_effect

	effect = np.stack(df['indirect_effect_layer'].to_numpy())  # Convert column to 2d ndarray (num_examples x num_layers)
	mean_effect = effect.mean(axis=0)
	json_data['mean_effect_layer'] = mean_effect

	effect = np.stack(df['indirect_effect_head'].to_numpy())  # Convert column to 2d ndarray (num_examples x num_layers)
	mean_effect = effect.mean(axis=0)
	json_data['mean_effect_head'] = mean_effect

	pickle.dump(json_data, open(out_dir + "/mean_effect_" + model_type + "_" + data + ".pickle", "wb" ))
	return mean_effect


def greedy(k, interventions, model, model_type, data, out_dir):
	
	greedy_filename = out_dir + "/greedy_" + model_type + "_" + data + ".pickle"

	if os.path.exists(greedy_filename):
		print('loading precomputed greedy values')
		res = pickle.load( open(greedy_filename, "rb" )) 
		obj_list_gr = res['val']
		layer_list = [i[0] for i in res['head']]
		heads_list = [i[1] for i in res['head']]
		k = k - len(obj_list_gr)
	else:
		layer_list = []
		heads_list = []
		obj_list_gr = []

	
	json_data = {}
	json_data_inter = {}


	for i in range(k):
		results = perform_interventions_single(interventions, model, layers_to_adj=np.array(layer_list), 
		                                       heads_to_adj=np.array(heads_list), search=True)

		df = pd.DataFrame(results)

		effect = np.stack(df['indirect_effect_head'].to_numpy())  # Convert column to 2d ndarray (num_examples x num_layers)
		mean_effect1 = effect.mean(axis=0)
		json_data_inter[i] = effect

		for j in zip(layer_list, heads_list):
			mean_effect1[j] = -100

		idx = np.argpartition(mean_effect1, mean_effect1.size - 1, axis=None)[-1:]
		res = np.column_stack(np.unravel_index(idx, mean_effect1.shape))

		obj_list_gr.append(np.max(mean_effect1))
		layer_list.append(res[:,0][0])
		heads_list.append(res[:,1][0])

		pickle.dump(json_data_inter, open(out_dir + "/greedy_intermediate_" + model_type + "_" + data + ".pickle", "wb" ))

		json_data['val'] = obj_list_gr
		json_data['head'] = [i for i in zip(layer_list, heads_list)]
		pickle.dump(json_data, open(greedy_filename, "wb" ))


if __name__ == '__main__':
	ap = ArgumentParser(description="Neuron subset selection.")
	ap.add_argument('--model_type', type=str, choices=['distil-gpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
	ap.add_argument('--algo', type=str, choices=['topk', 'greedy'], default='topk')
	ap.add_argument('--k', type=int, default=1)
	ap.add_argument('--data', type=str, choices=['winobias', 'winogender'], default='winogender')
	ap.add_argument('--out_dir', type=str, default='results')

	args = ap.parse_args()

	model_type = args.model_type
	algo = args.algo
	k = args.k
	data = args.data
	out_dir = args.out_dir
    
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if args.data == 'winobias':
		data_ext = 'wb'
	else:
		data_ext = 'wg'

	tokenizer = GPT2Tokenizer.from_pretrained(model_type)
	model = Model(output_attentions=True, device='cuda', gpt2_version=model_type)

	if data == 'winobias':
		interventions, _ = get_interventions_winobias(model_type, do_filter=True, split='dev', model=model, 
														tokenizer=tokenizer, device='cuda')
	else:
		interventions, _ = get_interventions_winogender(model_type, do_filter=True, stat='bls', model=model, 
										tokenizer=tokenizer, device='cuda')

	if algo == 'topk':
		mean_effect_filename = out_dir + "/mean_effect_" + model_type + "_" + data_ext + ".pickle"
		if os.path.exists(mean_effect_filename):
			print('loading precomputed mean effect')
			mean_effect_res = pickle.load( open(mean_effect_filename, "rb" ))
			mean_effect = mean_effect_res['mean_effect_head']
		else: 
			mean_effect = get_all_contrib(model_type, model, tokenizer, interventions, data_ext, out_dir)
		top_k(int(k), interventions, mean_effect, model, model_type, data_ext, out_dir)
	else:
		greedy(int(k), interventions, model, model_type, data_ext, out_dir)
