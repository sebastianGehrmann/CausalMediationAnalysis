import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


def compute_total_effect(row):
    """Compute the total effect based on the bias directionality."""
    if row["base_c1_effect"] >= 1.0:
        return row["alt1_effect"] / row["base_c1_effect"]
    else:
        return row["alt2_effect"] / row["base_c2_effect"]


def filtered_mean(df, column_name, profession_stereotypicality, model_name):
    """Get the mean effects after excluding strictly definitional professions."""

    def get_profession(s):
        # Discard PADDING TEXT used in XLNet
        if model_name.startswith('xlnet'): s = s.split('<eos>')[-1]
        return s.split()[1]

    def get_stereotypicality(vals):
        return abs(profession_stereotypicality[vals]["definitional"])

    df["profession"] = df["base_string"].apply(get_profession)
    df["definitional"] = df["profession"].apply(get_stereotypicality)
    return df[df["definitional"] < 0.75][column_name].mean()


def main(folder_name="results/20191114_neuron_intervention/", model_name="distilgpt2"):
    profession_stereotypicality = {}
    with open("experiment_data/professions.json") as f:
        for l in f:
            for p in eval(l):
                profession_stereotypicality[p[0]] = {
                    "stereotypicality": p[2],
                    "definitional": p[1],
                    "total": p[2] + p[1],
                    "max": max([p[2], p[1]], key=abs),
                }

    fnames = [
        f
        for f in os.listdir(folder_name)
        if "_" + model_name + ".csv" in f and f.endswith("csv")
    ]
    paths = [os.path.join(folder_name, f) for f in fnames]
    # fnames[:5], paths[:5]
    woman_files = [
        f
        for f in paths
        if "woman_indirect" in f
        if os.path.exists(f.replace("indirect", "direct"))
    ]

    means = []
    he_means = []
    she_means = []
    # For correlations.
    all_female_effects = []
    for path in woman_files:
        temp_df = pd.read_csv(path).groupby("base_string").agg("mean").reset_index()
        temp_df["alt1_effect"] = (
            temp_df["candidate1_alt1_prob"] / temp_df["candidate2_alt1_prob"]
        )
        temp_df["alt2_effect"] = (
            temp_df["candidate2_alt2_prob"] / temp_df["candidate1_alt2_prob"]
        )
        temp_df["base_c1_effect"] = (
            temp_df["candidate1_base_prob"] / temp_df["candidate2_base_prob"]
        )
        temp_df["base_c2_effect"] = (
            temp_df["candidate2_base_prob"] / temp_df["candidate1_base_prob"]
        )
        temp_df["he_total_effect"] = temp_df["alt1_effect"] / temp_df["base_c1_effect"]
        temp_df["she_total_effect"] = temp_df["alt2_effect"] / temp_df["base_c2_effect"]
        temp_df["total_effect"] = temp_df.apply(compute_total_effect, axis=1)

        mean_he_total = filtered_mean(
            temp_df, "he_total_effect", profession_stereotypicality, model_name
        )
        mean_she_total = filtered_mean(
            temp_df, "she_total_effect", profession_stereotypicality, model_name
        )
        mean_total = filtered_mean(
            temp_df, "total_effect", profession_stereotypicality, model_name
        )
        he_means.append(mean_he_total)
        she_means.append(mean_she_total)
        means.append(mean_total)
        all_female_effects.append(temp_df[["base_string", "she_total_effect"]])

    print("The total effect of this model is {:.3f}".format(np.mean(means) - 1))
    print(
        "The total (male) effect of this model is {:.3f}".format(np.mean(he_means) - 1)
    )
    print(
        "The total (female) effect of this model is {:.3f}".format(
            np.mean(she_means) - 1
        )
    )

    # Part 2: Get correlations.

    all_female_total_effects = pd.concat(all_female_effects)
    all_female_total_effects = all_female_total_effects.rename(
        columns={"she_total_effect": "total_effect"}
    )
    x_vals = []
    y_vals = []
    labels = []
    for index, row in all_female_total_effects.iterrows():
        labels.append(row["base_string"])
        y_vals.append(row["total_effect"])
        x_vals.append(
            profession_stereotypicality[
                row["base_string"].split()[1] if not model_name.startswith('xlnet')
                else row["base_string"].split('<eos>')[-1].split()[1]
            ]["total"]
        )
    profession_df = pd.DataFrame(
        {"example": labels, "Bias": x_vals, "Total Effect": np.log(y_vals)}
    )
    plt.figure(figsize=(10, 3))
    ax = sns.lineplot(
        "Bias", "Total Effect", data=profession_df, markers=True, dashes=True
    )
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["$e^0$", "$e^1$", "$e^2$", "$e^3$", "$e^4$", "$e^5$"])
    sns.despine()
    plt.savefig(os.path.join(folder_name, "neuron_profession_correlation.pdf"))

    effect_corr = pearsonr(profession_df["Bias"], profession_df["Total Effect"])
    print("================")
    print(
        "The correlation between bias value and (log) effect is {:.2f} (p={:.3f})".format(
            effect_corr[0], effect_corr[1]
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    # gpt2, gpt2-medium, gpt2-large
    model_name = sys.argv[2]

    main(folder_name, model_name)
