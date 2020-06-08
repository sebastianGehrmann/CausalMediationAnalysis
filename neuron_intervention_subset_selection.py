# -*- coding: utf-8 -*-

import gc
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer

from experiment import Intervention, Model
from utils import convert_results_to_pd

np.random.seed(1)
torch.manual_seed(1)

def get_template_list():
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    return ["The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because"]

def get_intervention_types():
    return ['man_indirect',
            'woman_indirect']

def construct_interventions(base_sent, tokenizer, DEVICE, gender='female'):
  interventions = {}
  if gender == 'female':
    filename = 'experiment_data/professions_female_stereo.json'
  else:
    filename = 'experiment_data/professions_male_stereo.json'
  with open(filename, 'r') as f:
      all_word_count = 0
      used_word_count = 0
      for l in f:
          # there is only one line that eval's to an array
          for j in eval(l):
              all_word_count += 1
              biased_word = j[0]
              try: 
                interventions[biased_word] = Intervention(
                            tokenizer,
                            base_sent,
                            [biased_word, "man", "woman"],
                            ["he", "she"],
                            device=DEVICE)
                used_word_count += 1
              except:
                pass
                  # print("excepted {} due to tokenizer splitting.".format(
                  #     biased_word))

      print("Only used {}/{} neutral words due to tokenizer".format(
          used_word_count, all_word_count))
  return interventions


def compute_odds_ratio(df, gender='female', col='odds_ratio'):

  # filter some stuff out
  df['profession'] = df['base_string'].apply(lambda s: s.split()[1])
  df['definitional'] = df['profession'].apply(get_stereotypicality)
  df = df.loc[df['definitional'] < 0.75, :]
  df = df[df['candidate2_base_prob'] > 0.01]
  df = df[df['candidate1_base_prob'] > 0.01]

  if gender == 'female':
    
    odds_base = df['candidate1_base_prob'] / df['candidate2_base_prob']
    odds_intervention = df['candidate1_prob'] / df['candidate2_prob']
  else:
    
    odds_base = df['candidate2_base_prob'] / df['candidate1_base_prob']
    odds_intervention = df['candidate2_prob'] / df['candidate1_prob']

  odds_ratio = odds_intervention / odds_base
  df[col] = odds_ratio
  return df

def sort_odds_obj(df):
  df['odds_diff'] = df['odds_ratio'].apply(lambda x: x-1)

  df_sorted = df.sort_values(by=['odds_diff'], ascending=False)
  return df_sorted

def get_stereotypicality(vals):
        return abs(profession_stereotypicality[vals]['definitional'])

profession_stereotypicality = {}
with open("experiment_data/professions.json") as f:
    for l in f:
        for p in eval(l):
            profession_stereotypicality[p[0]] = {
                'stereotypicality': p[2],
                'definitional': p[1],
                'total': p[2]+p[1], 
                'max': max([p[2],p[1]], key=abs)}
# get global list 
def get_all_contrib(templates, tokenizer, out_dir=''):

  # get marginal contrib to empty set
  female_df = get_intervention_results(templates, tokenizer, gender='female')
  male_df = get_intervention_results(templates, tokenizer, gender='male')
  gc.collect()

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')
  female_df = female_df[['layer','neuron', 'odds_ratio']]
  male_df = male_df[['layer','neuron', 'odds_ratio']]
  gc.collect()
  # merge and average
  df = pd.concat([female_df, male_df])
  df = df.groupby(['layer','neuron'], as_index=False).mean()
  df_sorted = sort_odds_obj(df)
  layer_list = df_sorted['layer'].values
  neuron_list = df_sorted['neuron'].values
  odds_list = df_sorted['odds_ratio'].values

  marg_contrib = {}
  marg_contrib['layer'] = layer_list
  marg_contrib['neuron'] = neuron_list
  marg_contrib['val'] = odds_list

  pickle.dump(marg_contrib, open(out_dir + "/marg_contrib_" + model_type + ".pickle", "wb" ))
  return layer_list, neuron_list

def get_intervention_results(templates, tokenizer, DEVICE='cuda', gender='female',
                             layers_to_adj=[], neurons_to_adj=[], intervention_loc='all',
                             df_layer=None, df_neuron=None):
  if gender == 'female':
    intervention_type = 'man_indirect'
  else:
    intervention_type = 'woman_indirect'
  df = []
  for template in templates:
    # pickle.dump(template + "_" + gender, open("results/log.pickle", "wb" ) )
    interventions = construct_interventions(template, tokenizer, DEVICE, gender)
    intervention_results = model.neuron_intervention_experiment(interventions, intervention_type, 
                                                                layers_to_adj=layers_to_adj, neurons_to_adj=neurons_to_adj,
                                                                intervention_loc=intervention_loc)
    df_template = convert_results_to_pd(interventions, intervention_results, df_layer, df_neuron)
    # calc odds ratio and odds-abs 
    df.append(df_template)
    gc.collect()
  return pd.concat(df)

def get_neuron_intervention_results(templates, tokenizer, layers, neurons):
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layers, neurons_to_adj=[neurons], intervention_loc='neuron',
                                          df_layer=layers, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layers, neurons_to_adj=[neurons], intervention_loc='neuron',
                                        df_layer=layers, df_neuron=neurons[0])
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    df = pd.concat([female_df, male_df])
    return df['odds_ratio'].mean()

def top_k_by_layer(model, model_type, tokenizer, templates, layer, layer_list, neuron_list, k=50, out_dir=''):
  layer_2_ind = np.where(layer_list == layer)[0]
  neuron_2 = neuron_list[layer_2_ind]
  
  odd_abs_list = []
  for i in range(k):
    print(i)
    temp_list = list(neuron_2[:i+1])

    neurons = [temp_list]

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=layer, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=len(temp_list)*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])
    gc.collect()

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(df['odds_ratio'].mean()-1)
  
    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_type + '_' + str(layer) + ".pickle", "wb" ) )

def top_k(model, model_type, tokenizer, templates, layer_list, neuron_list, k=50, out_dir=''):
  odd_abs_list = []

  for i in range(k):
    print(i)
    n_list = list(neuron_list[:i+1])
    l_list = list(layer_list[:i+1])

    neurons = [n_list]  
    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=l_list, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=l_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=l_list, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    odd_abs_list.append(df['odds_ratio'].mean()-1)

    pickle.dump(odd_abs_list, open(out_dir + "/topk_" + model_type + ".pickle", "wb" ))


def greedy_by_layer(model, model_type, tokenizer, templates, layer, k=50, out_dir=''):
  neurons = []
  odd_abs_list = []
  neurons = []

  for i in range(k):


    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                          df_layer=layer, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    gc.collect()

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    greedy_res = {}
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(out_dir + "/greedy_" + model_type + "_" + str(layer) + ".pickle", "wb" ))

def greedy(model, model_type, tokenizer, templates, k=50, out_dir=''):
  neurons = []
  odd_abs_list = []
  layers = []

  greedy_filename = out_dir + "/greedy_" + model_type + ".pickle"

  if os.path.exists(greedy_filename):
    print('loading precomputed greedy values')
    res = pickle.load( open(greedy_filename, "rb" )) 
    odd_abs_list = res['val']
    layers = res['layer'] 
    neurons = res['neuron']
    k = k - len(odd_abs_list)
  else:
    neurons = []
    odd_abs_list = []
    layers = []

  for i in range(k):
    print(i)

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                          df_layer=None, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layers, neurons_to_adj=neurons, intervention_loc='all',
                                        df_layer=None, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')
    gc.collect()

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    neurons.append(df_sorted.head(1)['neuron'].values[0])
    layers.append(df_sorted.head(1)['layer'].values[0])
    odd_abs_list.append(df_sorted['odds_diff'].values[0])

    # memory issue
    del df
    del female_df
    del male_df
    gc.collect()

    greedy_res = {}
    greedy_res['layer'] = layers
    greedy_res['neuron'] = neurons
    greedy_res['val'] = odd_abs_list

    pickle.dump(greedy_res, open(greedy_filename, "wb" ))


def random_greedy_by_layer(layer, k=50, out_dir=''):
  neurons = []
  odd_abs_list = []
  neurons = []
  el_list = list(range(1,k+1))
  df = []
  for i in range(k):
    

    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                          df_layer=layer, df_neuron=None)
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=layer, neurons_to_adj=neurons, intervention_loc='layer',
                                        df_layer=layer, df_neuron=None)

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    df = df.groupby(['layer', 'neuron'], as_index=False).mean()
    df_sorted = sort_odds_obj(df)

    j = random.choice(el_list)
    neurons.append(df_sorted.head(j)['neuron'].values[-1])
    odd_abs_list.append(df_sorted.head(j)['odds_abs'].values[-1])

  pickle.dump(odd_abs_list, open("rand_greedy_" + str(layer) + ".pickle", "wb" ))
  pickle.dump(neurons, open("rand_greedy_neurons_" + str(layer) + ".pickle", "wb" ))

def test():
  layer_obj = []
  for layer in range(12):
    print(layer)
    neurons = [list(range(768))]
    # get marginal contrib to empty set
    female_df = get_intervention_results(templates, tokenizer, gender='female',
                                         layers_to_adj=768*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                          df_layer=layer, df_neuron=neurons[0])
    male_df = get_intervention_results(templates, tokenizer, gender='male',
                                       layers_to_adj=768*[layer], neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])

    # compute odds ratio differently for each gender
    female_df = compute_odds_ratio(female_df, gender='female')
    male_df = compute_odds_ratio(male_df, gender='male')

    # merge and average
    df = pd.concat([female_df, male_df])
    print(layer)
    # print(df_sorted['odds_abs'].values[0])
    layer_obj.append(abs(df['odds_ratio'].mean()-1))

  neurons = [12*list(range(768))]
  # get marginal contrib to empty set
  layer_list = []
  for l in range(12):
    layer_list += (768 * [l])
  female_df = get_intervention_results(templates, tokenizer, gender='female',
                                       layers_to_adj=layer_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                        df_layer=layer, df_neuron=neurons[0])
  male_df = get_intervention_results(templates, tokenizer, gender='male',
                                     layers_to_adj=layer_list, neurons_to_adj=neurons, intervention_loc='neuron',
                                      df_layer=layer, df_neuron=neurons[0])

  # compute odds ratio differently for each gender
  female_df = compute_odds_ratio(female_df, gender='female')
  male_df = compute_odds_ratio(male_df, gender='male')

  # merge and average
  df = pd.concat([female_df, male_df])
  print(layer_obj)
  print('all')
  print(abs(df['odds_ratio'].mean()-1))

if __name__ == '__main__':
    ap = ArgumentParser(description="Neuron subset selection.")
    ap.add_argument('--model_type', type=str, choices=['distil-gpt2', 'gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
    ap.add_argument('--algo', type=str, choices=['topk', 'greedy', 'random_greedy', 'test'], default='topk')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--layer', type=int, default=-1)
    ap.add_argument('--out_dir', type=str, default='results')

    args = ap.parse_args()
    
    algo = args.algo
    k = args.k
    layer = args.layer
    out_dir = args.out_dir
    model_type = args.model_type
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device='cuda')
    DEVICE = 'cuda'

    templates = get_template_list()

    if args.algo == 'topk':
        marg_contrib_path = out_dir + "/marg_contrib.pickle"
        if os.path.exists(marg_contrib_path):
            print('Using cached marginal contribution')
            marg_contrib = pickle.load( open(marg_contrib_path, "rb" )) 
            layer_list = marg_contrib['layer']
            neuron_list = marg_contrib['neuron']
        else:
            print('Computing marginal contribution')
            layer_list, neuron_list = get_all_contrib(templates, tokenizer, out_dir)
        if layer == -1:
            top_k(model, model_type, tokenizer, templates, layer_list, neuron_list, k, out_dir)
        elif layer != -1:
            top_k_by_layer(model, model_type, tokenizer, templates, layer, layer_list, neuron_list, k, out_dir)
    elif (args.algo == 'greedy') and (layer == -1):
        greedy(model, model_type, tokenizer, templates, k, out_dir)
    elif (args.algo == 'greedy') and (layer != -1):
        greedy_by_layer(model, model_type, tokenizer, templates, layer, k, out_dir)
    elif (args.algo == 'test'):
        test()
    else:
        random_greedy_by_layer(layer, k, out_dir)
