"""Collection of utilities for attention intervention"""

import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def perform_intervention(intervention, model, effect_types=('indirect', 'direct')):
    """Perform intervention and return results for specified effects"""
    x = intervention.base_strings_tok[0]  # E.g. The doctor asked the nurse a question. She
    x_alt = intervention.base_strings_tok[1]  # E.g. The doctor asked the nurse a question. He

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
        candidate1_probs_head, candidate2_probs_head, candidate1_probs_layer, candidate2_probs_layer,\
            candidate1_probs_model, candidate2_probs_model = model.attention_intervention_experiment(
            intervention, effect_type)
        odds_intervention_head = candidate2_probs_head / candidate1_probs_head
        odds_intervention_layer = candidate2_probs_layer / candidate1_probs_layer
        odds_intervention_model = candidate2_probs_model / candidate1_probs_model
        effect_head = (odds_intervention_head - odds_base) / odds_base
        effect_layer = (odds_intervention_layer - odds_base) / odds_base
        effect_model = (odds_intervention_model - odds_base) / odds_base

        results[effect_type + "_odds_head"] = odds_intervention_head.tolist()
        results[effect_type + "_effect_head"] = effect_head.tolist()
        results[effect_type + "_effect_layer"] = effect_layer.tolist()
        results[effect_type + "_effect_model"] = effect_model

    return results


def report_intervention(results, effect_types=('indirect', 'direct'), verbose=False):
    """Report results for single intervention"""

    print(f"x : {results['base_string1']}")
    print(f"x': {results['base_string2']}")
    print(f"c1: {results['candidate1']}")
    print(f"c2: {results['candidate2']}")
    print(f"\np(c2|x) / p(c1|x) = {results['odds_base']:.5f}")
    print(f"p(c2|x') / p(c1|x') = {results['odds_alt']:.5f}")
    print(f"\nTOTAL Effect: (p(c2|x') / p(c1|x')) / (p(c2|x) / p(c1|x)) - 1 = {results['total_effect']:.3f}")

    for effect_type in effect_types:
        if verbose:
            print(f'\n{effect_type.upper()} Effect')
            if effect_type == 'indirect':
                print("   Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
                print(f"   Effect = (p(c2|x, Attn(x')) / p(c1|x, Attn(x')) / (p(c2|x) / p(c1|x)) - 1")
            elif effect_type == 'direct':
                print("   Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
                print(f"   Effect = (p(c2|x', Attn(x)) / p(c1|x', Attn(x)) / (p(c2|x) / p(c1|x)) - 1")

        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(results[effect_type + '_effect_head'], annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'{effect_type.capitalize()} Effect')


def perform_interventions(interventions, model, effect_types=('indirect', 'direct')):
    """Perform multiple interventions"""
    results_list = []
    for intervention in tqdm(interventions):
        results = perform_intervention(intervention, model, effect_types)
        results_list.append(results)
    return results_list


def report_interventions_summary_by_head(results, effect_types=('indirect', 'direct'), verbose=False, k=10,
                                         show_head_examples=False):
    """Report summary results for multiple interventions by head"""

    df = pd.DataFrame(results)

    print('*** SUMMARY BY HEAD ***')
    print(f"Num interventions: {len(df)}")
    print(f"Mean total effect: {df.total_effect.mean():.3f}")

    for effect_type in effect_types:
        # Convert column to 3d ndarray (num_examples x num_layers x num_heads)
        effect = np.stack(df[effect_type + '_effect_head'].to_numpy())
        mean_effect = effect.mean(axis=0)
        if effect_type == 'indirect':
            ranking_metric = mean_effect
        else:
            ranking_metric = -mean_effect
        topk_indices = topk_indices(ranking_metric, k)

        # Compute significance levels
        all_values = effect.flatten()
        print(f'\n{effect_type.upper()} Effect (mean = {all_values.mean()})')
        print(f"Top {k} heads:")
        for ind in topk_indices:
            layer, head = np.unravel_index(ind, mean_effect.shape)
            head_values = effect[:, layer, head].flatten()
            tstatistic, pvalue = ttest_ind(head_values, all_values)
            if effect_type == 'indirect':
                assert tstatistic > 0
            else:
                assert tstatistic < 0
            one_tailed_pvalue = pvalue / 2
            print(f'   {layer} {head}: {mean_effect[layer, head]:.3f} (p={one_tailed_pvalue:.4f})')
            if effect_type == 'indirect' and show_head_examples:
                top_results_for_head = sorted(results,
                                               key=lambda result: result['indirect_effect_head'][layer][head],
                                               reverse=True)
                for result in top_results_for_head[:3]:
                    print(f'      {result["indirect_effect_head"][layer][head]:.3f} '
                        f'{result["base_string1"]} | {result["candidate1"]} | {result["candidate2"]}')
        if verbose:
            if effect_type == 'indirect':
                print("   Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
                print(f"   Effect = (p(c2|x, Attn(x')) / p(c1|x, Attn(x')) / (p(c2|x) / p(c1|x)) - 1")
            elif effect_type == 'direct':
                print("   Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
                print(f"   Effect = (p(c2|x', Attn(x)) / p(c1|x', Attn(x)) / (p(c2|x) / p(c1|x)) - 1")
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(mean_effect, annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')

def report_interventions_summary_by_layer(results, effect_types=('indirect', 'direct')):
    """Report summary results for multiple interventions by layer"""

    df = pd.DataFrame(results)

    print('*** SUMMARY BY LAYER ***')
    print(f"Num interventions: {len(df)}")
    print(f"Mean total effect: {df.total_effect.mean():.3f}")

    for effect_type in effect_types:
        # Convert column to 2d ndarray (num_examples x num_layers)
        effect = np.stack(df[effect_type + '_effect_layer'].to_numpy())
        mean_effect = effect.mean(axis=0)
        n_layers = mean_effect.shape[0]

        plt.figure(figsize=(9, 7))
        ax = sns.barplot(x=mean_effect, y=list(range(n_layers)), color="blue", saturation=.3, orient="h")
        ax.set(ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')


def get_odds_ratio(intervention, model):
    x = intervention.base_strings_tok[0]
    x_alt = intervention.base_strings_tok[1]
    with torch.no_grad():
        candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples_multitoken(
            x,
            intervention.candidates_tok)
        candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples_multitoken(
            x_alt,
            intervention.candidates_tok)

    odds_base = candidate2_base_prob / candidate1_base_prob
    odds_alt = candidate2_alt_prob / candidate1_alt_prob
    return odds_alt / odds_base

def topk_indices(arr, k):
    """Return indices of top-k values"""
    return (-arr).argsort(axis=None)[:k]


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from experiment import Intervention, Model
    from pandas import DataFrame
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(output_attentions=True)

    # Test experiment
    interventions = [
        Intervention(
            tokenizer,
            "The doctor asked the nurse a question. {}",
            ["He", "She"],
            ["asked", "answered"]),
        Intervention(
            tokenizer,
            "The doctor asked the nurse a question. {}",
            ["He", "She"],
            ["requested", "responded"])
    ]

    results = perform_interventions(interventions, model)
    report_interventions_summary_by_layer(results)



