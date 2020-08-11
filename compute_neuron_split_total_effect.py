import os
import sys

import numpy as np
import pandas as pd


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

    male_means_model = []
    female_means_model = []

    male_means_def = []
    female_means_def = []
    for path in woman_files:
        df = pd.read_csv(path).groupby("base_string").agg("mean").reset_index()

        def get_profession(s):
            # Discard PADDING TEXT used in XLNet
            if model_name.startswith('xlnet'): s = s.split('<eos>')[-1]
            return s.split()[1]

        # Set up filtering by stereotypicality
        def get_definitionality(vals):
            return abs(profession_stereotypicality[vals]["definitional"])

        def get_stereotypicality(vals):
            return profession_stereotypicality[vals]["total"]

        df["profession"] = df["base_string"].apply(get_profession)
        df["definitional"] = df["profession"].apply(get_definitionality)
        df["stereotypicality"] = df["profession"].apply(get_stereotypicality)
        """
        FILTERING
        """
        # Remove examples that are too definitional
        df = df[df["definitional"] < 0.75]
        # Ignore outliers with < 1% for he or she
        df = df[df["candidate1_base_prob"] > 0.01]
        df = df[df["candidate2_base_prob"] > 0.01]
        """
        TOTAL EFFECTS
        """
        # Compute base_ratios for man (he/she) and woman (she/he)
        df["man_he_she_effect"] = (
            df["candidate1_alt1_prob"] / df["candidate2_alt1_prob"]
        )
        df["woman_she_he_effect"] = (
            df["candidate2_alt2_prob"] / df["candidate1_alt2_prob"]
        )
        # Compute profession effect
        df["base_he_she_effect"] = (
            df["candidate1_base_prob"] / df["candidate2_base_prob"]
        )
        df["base_she_he_effect"] = (
            df["candidate2_base_prob"] / df["candidate1_base_prob"]
        )
        # Compute both directions total effect
        df["he_she_total_effect"] = df["man_he_she_effect"] / df["base_he_she_effect"]
        df["she_he_total_effect"] = df["woman_she_he_effect"] / df["base_she_he_effect"]

        """
        Compute the effects of:
        male -> woman | she/he
        female -> man | he/she
        """

        # (1) Filter by model direction
        female_mean_model = df[df["base_she_he_effect"] > 1.0][
            "he_she_total_effect"
        ].values
        female_means_model.extend(female_mean_model)

        male_mean_model = df[df["base_he_she_effect"] > 1.0][
            "she_he_total_effect"
        ].values
        male_means_model.extend(male_mean_model)

        # (2) Filter by stereotype
        female_mean_def = df[df["stereotypicality"] < 0.0]["he_she_total_effect"].values
        female_means_def.extend(female_mean_def)

        male_mean_def = df[df["stereotypicality"] > 0.0]["she_he_total_effect"].values
        male_means_def.extend(male_mean_def)

    # print("The total effect of this model is {:.3f}".format(np.mean(means)-1))
    print(
        "The total (female profession (model) -> man) effect of this model is {:.3f}".format(
            np.mean(male_means_model) - 1
        )
    )
    print(
        "The total (male profession (model) -> woman) effect of this model is {:.3f}".format(
            np.mean(female_means_model) - 1
        )
    )
    print(
        "The combined effect is {:.3f}".format(
            np.mean(female_means_model + male_means_model) - 1
        )
    )

    print(
        "The total (female profession (def) -> man) effect of this model is {:.3f}".format(
            np.mean(male_means_def) - 1
        )
    )
    print(
        "The total (male profession (def) -> woman) effect of this model is {:.3f}".format(
            np.mean(female_means_def) - 1
        )
    )
    print(
        "The combined effect is {:.3f}".format(
            np.mean(female_means_def + male_means_def) - 1
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
