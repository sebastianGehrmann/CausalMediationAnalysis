import numpy as np
import os
import pandas as pd
import sys


def compute_total_effect(row):
    if row["base_c1_effect"] >= 1.0:
        return row["alt1_effect"] / row["base_c1_effect"]
    else:
        return row["alt2_effect"] / row["base_c2_effect"]


def filtered_mean(df, column_name, profession_stereotypicality):
    def get_stereotypicality(vals):
        return abs(profession_stereotypicality[vals]["definitional"])

    df["profession"] = df["base_string"].apply(lambda s: s.split()[1])
    df["definitional"] = df["profession"].apply(get_stereotypicality)
    return df[df["definitional"] < 0.75][column_name].mean()


def main(folder_name="results/20191114_neuron_intervention/", model_name="distilgpt2"):
    profession_stereotypicality = {}
    with open("professions.json") as f:
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
            temp_df, "he_total_effect", profession_stereotypicality
        )
        mean_she_total = filtered_mean(
            temp_df, "she_total_effect", profession_stereotypicality
        )
        mean_total = filtered_mean(temp_df, "total_effect", profession_stereotypicality)
        he_means.append(mean_he_total)
        she_means.append(mean_she_total)
        means.append(mean_total)

    print("The total effect of this model is {:.3f}".format(np.mean(means) - 1))
    print(
        "The total (male) effect of this model is {:.3f}".format(np.mean(he_means) - 1)
    )
    print(
        "The total (female) effect of this model is {:.3f}".format(
            np.mean(she_means) - 1
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
