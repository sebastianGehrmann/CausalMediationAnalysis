"""Creates summary figure of various effects for attention intervention from JSON file"""

import json

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
import pandas as pd
import os

def main():

    models = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    model_to_name = {
        'distilgpt2': 'distil',
        'gpt2': 'small',
        'gpt2-medium': 'medium',
        'gpt2-large': 'large',
        'gpt2-xl': 'xl'
    }

    sns.set_context("paper")
    sns.set_style("white")
    mpl.rcParams['hatch.linewidth'] = 0.3

    palette = sns.color_palette()

    filter = 'filtered'
    split = 'dev'
    dataset = 'winobias'

    te = []
    nde_all = []
    nie_all = []
    nie_sum = []
    model_names = []

    for model_version in models:
        fname = f"{dataset}_data/attention_intervention_{model_version}_{filter}_{split}.json"
        with open(fname) as f:
            data = json.load(f)
        df = pd.DataFrame(data['results'])
        # Convert to shape (num_examples X num_layers X num_heads)
        indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
        mean_sum_indirect_effect = indirect_by_head.sum(axis=(1, 2)).mean()
        te.append(data['mean_total_effect'])
        nde_all.append(data['mean_model_direct_effect'])
        nie_all.append(data['mean_model_indirect_effect'])
        nie_sum.append(mean_sum_indirect_effect)
        model_names.append(model_to_name[model_version])

    # Plot stacked bar chart
    plt.figure(num=1, figsize=(3, 1.2))
    width = .29
    inds = np.arange(len(models))
    spacing = 0.015
    p1 = plt.bar(inds, te, width, color=palette[2], linewidth=0, hatch='/////', edgecolor='darkgreen')
    p2 = plt.bar(inds + width + spacing, nie_all, width, color=palette[4], linewidth=0, hatch='\\\\\\',
                 edgecolor='#4E456D')
    p3 = plt.bar(inds + width + spacing, nde_all, width, bottom=nie_all, color=palette[1], linewidth=0,
                 hatch='----', edgecolor='#BB592D')
    p4 = plt.bar(inds + 2 * (width + spacing), nie_sum, width, color=palette[3], linewidth=0, hatch='///',
                 edgecolor='darkred')

    plt.gca().tick_params(axis='x', pad=0)
    plt.gca().tick_params(axis='y', pad=0)
    plt.gca().yaxis.labelpad = 3
    plt.ylabel('Effect', size=9)
    plt.xticks(inds + .3, model_names, size=7)
    for tick in plt.gca().xaxis.get_minor_ticks():
        tick.label1.set_horizontalalignment('center')
    plt.yticks(size=7)
    leg = plt.legend((p1[0], p3[0], p2[0], p4[0]), ('TE', 'NDE-all', 'NIE-all', 'NIE-sum'), loc='upper left', fontsize=7)
    for patch in leg.get_patches():
        patch.set_height(7)
        patch.set_y(-1)
    sns.despine()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.15)
    path = 'results/attention_intervention/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}effects.pdf', format='pdf')
    plt.close()

if __name__ == '__main__':
    main()