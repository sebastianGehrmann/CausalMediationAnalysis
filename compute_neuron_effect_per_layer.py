"""Script to generate a plot with effect per-layer.

Requirement:
You have to have run `compute_and_save_neuron_agg_effect.py` for each of the models you want
to investigate. That script will save an intermediate result csv. To reduce computational
overhead, this file expects those intermediate result files.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")


def get_top_perc_per_layer(df, n=10):
    """Get avg indirect effect of top n% neurons"""
    num_neurons = int(df.groupby("layer_direct_mean").size()[0] * n / 100.0)
    return (
        df.groupby("layer_direct_mean")
        .apply(lambda x: x.nlargest(num_neurons, ["odds_ratio_indirect_mean"]))
        .reset_index(drop=True)
        .groupby("layer_direct_mean")
        .agg("mean")[["odds_ratio_indirect_mean", "odds_ratio_indirect_std"]]
        .reset_index()
    )


def main(folder_name="results/20191114_neuron_intervention/"):
    # For plotting purposes.
    sanitize_model_names = {
        "gpt2-large": "GPT2-large",
        "gpt2-random": "GPT2-small random",
        "gpt2": "GPT2-small",
        "gpt2-medium": "GPT2-medium",
        "gpt2-xl": "GPT2-xl",
        "distilgpt2": "GPT2-distil",
    }
    cmap = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    # Load results for all the models.
    neuron_effect_fnames = [f for f in os.listdir(folder_name) if "neuron_effects" in f]
    modelname_to_effects = {}
    for f in neuron_effect_fnames:
        modelname = sanitize_model_names[f.split("_")[0]]
        path = os.path.join(folder_name, f)
        modelname_to_effects[modelname] = pd.read_csv(path)

    plt.figure(figsize=(10, 4))
    color_index = 0
    for k, v in modelname_to_effects.items():
        # Get top 2.5% neurons (empirical choice, you can vary this).
        vals = get_top_perc_per_layer(v, 2.5)
        # Plot a line for mean per layer.
        plt.plot(
            vals["layer_direct_mean"].values,
            vals["odds_ratio_indirect_mean"].values,
            label=k,
            color=cmap[color_index],
        )
        # Fill in between standard deviation.
        plt.fill_between(
            vals["layer_direct_mean"].values,
            vals["odds_ratio_indirect_mean"].values
            - vals["odds_ratio_indirect_std"].values,
            vals["odds_ratio_indirect_mean"].values
            + vals["odds_ratio_indirect_std"].values,
            alpha=0.08,
            color=cmap[color_index],
        )
        color_index += 1
    plt.xlabel("Layer index", fontsize=18)
    plt.ylabel("Indirect effect of top neurons", fontsize=18)
    plt.ylim([0.995, 1.05])
    plt.yticks([1 + i / 100 for i in range(0, 6)], [str(i / 100) for i in range(0, 6)])

    # Need to reorder legend labels to increasing model size.
    # Requires results for all 6 models, ignore otherwise.
    try:
        handles, labels = plt.gca().get_legend_handles_labels()
        # Modify to fit the order you are getting.
        order = [3, 5, 2, 4, 0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    except IndexError:
        plt.legend()
    sns.despine()
    plt.savefig(
        os.path.join(folder_name, "neuron_layer_effect.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    # Now do all available models individually.
    color_index = 0
    for k, v in modelname_to_effects.items():
        plt.figure(figsize=(10, 4))
        vals = get_top_perc_per_layer(v, 2.5)
        plt.plot(
            vals["layer_direct_mean"].values,
            vals["odds_ratio_indirect_mean"].values,
            label=k,
            color=cmap[color_index],
        )
        plt.fill_between(
            vals["layer_direct_mean"].values,
            vals["odds_ratio_indirect_mean"].values
            - vals["odds_ratio_indirect_std"].values,
            vals["odds_ratio_indirect_mean"].values
            + vals["odds_ratio_indirect_std"].values,
            alpha=0.08,
            color=cmap[color_index],
        )
        plt.title(k)
        plt.xlabel("Layer index", fontsize=18)
        plt.ylabel("Indirect effect of top neurons", fontsize=18)

        max_y = int(np.ceil((max(v["odds_ratio_indirect_mean"]) - 1) * 100))
        plt.yticks(
            [1 + i / 100 for i in range(0, max_y + 1)],
            [str(i / 100) for i in range(0, max_y + 1)],
        )

        color_index += 1
        plt.savefig(
            os.path.join(folder_name, "neuron_layer_effect_" + k + ".pdf"),
            format="pdf",
            bbox_inches="tight",
        )

    print("Success, all figures were written.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python ", sys.argv[0], "<folder_name>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]

    main(folder_name)
