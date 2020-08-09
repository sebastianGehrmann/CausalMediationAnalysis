"""
Compute the aggregate effects for each individual neuron.
Save the effects as $model_neuron_effects.csv.

Usage:
    python compute_and_save_neuron_agg_effect.py $result_file_path $model_name
"""
import os
import sys

import pandas as pd


def analyze_effect_results(results_df, effect, word, alt, savefig=None):
    # calculate odds.
    if alt == "man":
        odds_base = (
            results_df["candidate1_base_prob"] / results_df["candidate2_base_prob"]
        )
        odds_intervention = (
            results_df["candidate1_prob"] / results_df["candidate2_prob"]
        )
    else:
        odds_base = (
            results_df["candidate2_base_prob"] / results_df["candidate1_base_prob"]
        )
        odds_intervention = (
            results_df["candidate2_prob"] / results_df["candidate1_prob"]
        )
    odds_ratio = odds_intervention / odds_base
    results_df["odds_ratio"] = odds_ratio

    if word == "all":
        # average over words
        results_df = results_df.groupby(["layer", "neuron"], as_index=False).mean()
    else:
        # choose one word
        results_df = results_df[results_df["word"] == word]

    results_df = results_df.pivot("neuron", "layer", "odds_ratio")


def get_all_effects(fname, direction="woman"):
    """
    Give fname from a direct effect file
    """
    # Step 1: Load results for current folder and gender
    print(fname)
    indirect_result_df = pd.read_csv(fname)
    analyze_effect_results(
        results_df=indirect_result_df, effect="indirect", word="all", alt=direction
    )
    fname = fname.replace("_indirect_", "_direct_")
    direct_result_df = pd.read_csv(fname)
    analyze_effect_results(
        results_df=direct_result_df, effect="direct", word="all", alt=direction
    )

    # Step 2: Join the two DF's
    total_df = direct_result_df.join(
        indirect_result_df, lsuffix="_direct", rsuffix="_indirect"
    )[
        [
            "base_string_direct",
            "layer_direct",
            "neuron_direct",
            "odds_ratio_indirect",
            "odds_ratio_direct",
        ]
    ]
    total_df["total_causal_effect"] = (
        total_df["odds_ratio_indirect"] + total_df["odds_ratio_direct"] - 1
    )

    return total_df


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
    woman_files = [
        f
        for f in paths
        if "woman_indirect" in f
        if os.path.exists(f.replace("indirect", "direct"))
    ]
    woman_dfs = []
    for path in woman_files:
        woman_dfs.append(get_all_effects(path))
    woman_df = pd.concat(woman_dfs)

    man_files = [
        f
        for f in paths
        if "_man_indirect" in f
        if os.path.exists(f.replace("indirect", "direct"))
    ]
    man_dfs = []
    for path in man_files:
        man_dfs.append(get_all_effects(path, "man"))
    man_df = pd.concat(man_dfs)

    # Compute Extra Info
    def get_profession(s):
        # Discard PADDING TEXT used in XLNet
        if model_name.startswith('xlnet'): s = s.split('<eos>')[-1]
        return s.split()[1]

    def get_template(s):
        # Discard PADDING TEXT used in XLNet
        if model_name.startswith('xlnet'): s = s.split('<eos>')[-1]
        initial_string = s.split()
        initial_string[1] = "_"
        return " ".join(initial_string)

    man_df["profession"] = man_df["base_string_direct"].apply(get_profession)
    man_df["template"] = man_df["base_string_direct"].apply(get_template)
    woman_df["profession"] = woman_df["base_string_direct"].apply(get_profession)
    woman_df["template"] = woman_df["base_string_direct"].apply(get_template)

    def get_stereotypicality(vals):
        return profession_stereotypicality[vals]["total"]

    def get_definitionality(vals):
        return abs(profession_stereotypicality[vals]["definitional"])

    man_df["stereotypicality"] = man_df["profession"].apply(get_stereotypicality)
    woman_df["stereotypicality"] = woman_df["profession"].apply(get_stereotypicality)
    # Exclude very definitional examples.
    man_df["definitional"] = man_df["profession"].apply(get_definitionality)
    woman_df["definitional"] = woman_df["profession"].apply(get_definitionality)

    man_df = man_df[man_df["definitional"] > 0.75]
    woman_df = woman_df[woman_df["definitional"] > 0.75]

    # Merge effect based on directionality.
    overall_df = pd.concat(
        [
            man_df[man_df["stereotypicality"] < 0],
            woman_df[woman_df["stereotypicality"] >= 0],
        ]
    )
    # Save some RAM, next step is _expensive_!
    del man_df
    del woman_df

    overall_df["neuron"] = (
        overall_df["layer_direct"].map(str) + "-" + overall_df["neuron_direct"].map(str)
    )
    neuron_effect_df = (
        overall_df.groupby("neuron")
        .agg(
            {
                "layer_direct": ["mean"],
                "neuron_direct": ["mean"],
                "odds_ratio_indirect": ["mean", "std"],
                "odds_ratio_direct": ["mean", "std"],
                "total_causal_effect": ["mean", "std"],
            }
        )
        .reset_index()
    )
    neuron_effect_df.columns = [
        "_".join(col).strip() for col in neuron_effect_df.columns.values
    ]
    path_name = os.path.join(folder_name, model_name + "_neuron_effects.csv")
    neuron_effect_df.to_csv(path_name)
    print("Effect csv saved to {}".format(path_name))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    # gpt2, gpt2-medium, gpt2-large
    model_name = sys.argv[2]

    main(folder_name, model_name)
