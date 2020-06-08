"""Creates figures showing attention for specific examples, based on JSON files"""

import json
import math
from operator import itemgetter

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

BLACK = '#000000'
GRAY = '#303030'

def save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices=None):
    palette = sns.color_palette('muted')
    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(3.3, 1.95))
    axs[0].yaxis.set_ticks_position('none')
    plt.rcParams.update({'axes.titlesize': 'xx-large'})
    attentions = []
    max_attn = 0
    seqs = []
    for g_index in range(2):
        prompt = prompts[g_index]
        print(prompt)
        input_ = tokenizer.encode(prompt)
        print(input_)
        batch = torch.tensor(input_).unsqueeze(0).to(device)
        attention = model(batch)[-1]
        seq = tokenizer.convert_ids_to_tokens(input_)
        print(seq)
        seq = [t.replace('Ġ', '') for t in seq]
        seqs.append(seq)
        seq_len = len(input_)
        attention = torch.stack(attention)
        attention = attention.squeeze(1)
        assert torch.allclose(attention.sum(-1), torch.tensor([1.0]))
        attentions.append(attention)
        attn_sum = torch.Tensor([0])
        for layer, head in heads:
            attn_sum = attention[layer][head][-1] + attn_sum
        if max(attn_sum) > max_attn:
            max_attn = max(attn_sum)
    xlim_upper = math.ceil(max_attn * 10) / 10
    for g_index in range(2):
        attention = attentions[g_index]
        head_names = []
        ax = axs[g_index]
        seq = seqs[g_index]
        formatted_seq = []
        if highlight_indices:
            for i, t in enumerate(seq):
                formatted_t = t
                for j in range(2):
                    if i in highlight_indices[j]:
                        if j == g_index:
                            formatted_t = f"\\textbf{{{t}}}"
                        else:
                            formatted_t = f"\\setul{{.15ex}}{{.2ex}}\\ul{{{t}}}"
                        break

                formatted_seq.append(formatted_t)
            formatted_seq[-1] = f"\\textbf{{{formatted_seq[-1]}}}"
        else:
            formatted_seq = seq
        print('formatted', formatted_seq)
        plts = []
        left = None
        for i, (layer, head) in enumerate(heads):
            attn_last_word = attention[layer][head][-1].numpy()
            seq_placeholders = [f'a{i}' for i in range(len(formatted_seq))]
            if left is None:
                print(attn_last_word)
                p = ax.barh(seq_placeholders, attn_last_word, color=palette[i], linewidth=0)
            else:
                p = ax.barh(seq_placeholders, attn_last_word, left=left, color=palette[i], linewidth=0)
            print(ax.get_yticklabels())
            ax.set_yticklabels(formatted_seq)
            if left is None:
                left = np.zeros_like(attn_last_word)
            left += attn_last_word

            if highlight_indices:
                for i in range(seq_len):
                    if i in highlight_indices[g_index]:
                        color = BLACK
                    else:
                        color = GRAY
                    ax.get_yticklabels()[i].set_color(color)
                ax.get_yticklabels()[-1].set_color(BLACK)
            plts.append(p)
            head_names.append(f"Head {layer}-{head}")

        ax.set_xlim([0, xlim_upper])
        ax.set_xticks([0, xlim_upper])
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=8, ha='right')
        ax.set_xticks([0, 0.5])
        plt.setp(ax.get_xticklabels(), fontsize=7)
        sns.despine(left=True, bottom=True)
        ax.tick_params(axis='x', pad=0, length=0)
        ax.tick_params(axis='y', pad=0)
        ax.yaxis.labelpad = 0
        ax.xaxis.labelpad = 0

    lgd = plt.figlegend(plts, head_names,'lower center', fontsize=7, borderpad=0.5, handlelength=.9,
                        handletextpad=.3, labelspacing = 0.15, bbox_to_anchor=(0.86, 0.11))

    plt.savefig(fname, format='pdf', bbox_extra_artists = (lgd,), bbox_inches = 'tight')

    plt.close()


def main():
    sns.set_context("paper")
    sns.set_style("white")
    device = 'cpu'
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "text.latex.preamble": [
            "\\usepackage{color}",
            "\\usepackage{soul}",
            "\\setulcolor{blue}"
        ]
    })
    top_heads = {
        'gpt2':[(5, 8), (5, 10), (4,6)],
        'gpt2-medium': [(10, 9), (6, 15), (10,12)],
        'gpt2-xl':[(16,15), (16, 24), (17,10)],
        'gpt2-large':[(16,19), (16,5), (15,6)],
        'distilgpt2': [(3,1), (2,6), (3,6)]
    }
    models = ['gpt2', 'gpt2-medium', 'gpt2-xl', 'gpt2-large', 'distilgpt2']

    examples_to_highlight = {
        "The guard appreciated getting treatment from the nurse": [[7], [1]],
        "The driver transported the housekeeper to the job because": [[4,5], [1]],
        "The manager promised to give a raise to the teacher": [[9], [1]],
        "The driver never drove the librarian anywhere": [[5,6], [1]],
        "The nurse examined the farmer for injuries because": [[1], [4]],
        "The CEO ordered the cleaner out of the room": [[4],[1]],
        "The hairdresser shouted at the driver because": [[1,2,3,4],[8]],
        "The chief demanded an apology from the teacher because": [[7], [1]],
        "The physician examined the auditor for sores because": [[4],[1]],
        "The laborer dug a hole for the assistant because": [[8],[1,2]]
    }

    split = 'dev'
    testing = False
    for model_version in models:
        heads = top_heads[model_version]
        if model_version == 'distilgpt2':
            filter = 'unfiltered' # In order to get canonical example
        else:
            filter = 'filtered'
        fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
        with open(fname) as f:
            data = json.load(f)
        prompts = None
        results = data['results']
        results_by_ratio = sorted(results, key=itemgetter('total_effect'), reverse=True)

        with torch.no_grad():
            # Get attention and validate
            model = GPT2Model.from_pretrained(model_version, output_attentions=True)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model.eval()
            for result_index, result in enumerate(results_by_ratio):
                prompts = (result['base_string1'], result['base_string2'])
                highlight_indices = None
                for example, indices in examples_to_highlight.items():
                    if example in prompts[0]:
                        highlight_indices = indices
                        break
                fname = f'results/attention_intervention/qualitative/winobias_{model_version}_{filter}_{split}_{result_index}.pdf'
                save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices)
                # For testing only:
                if testing:
                    break
        if testing:
            break


if __name__ == '__main__':
    main()