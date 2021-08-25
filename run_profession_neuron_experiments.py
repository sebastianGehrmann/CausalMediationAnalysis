"""Run all the extraction for a model across many templates.
"""
import argparse
import os
from datetime import datetime

import torch
from transformers import (
    GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer,
    BertTokenizer, DistilBertTokenizer, RobertaTokenizer
)

from experiment import Intervention, Model
from utils import convert_results_to_pd

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
    "-model",
    type=str,
    default="distilgpt2",
    help="""Model type [distilgpt2, gpt-2, etc.].""",
)

parser.add_argument(
    "-out_dir", default=".", type=str, help="""Path of the result folder."""
)

parser.add_argument(
    "-template_indices",
    nargs="+",
    type=int,
    help="Give the indices of templates if you want to run on only a subset",
)

parser.add_argument(
    "--randomize", default=False, action="store_true", help="Randomize model weights."
)

opt = parser.parse_args()


def get_profession_list():
    # Get the list of all considered professions
    word_list = []
    with open("experiment_data/professions.json", "r") as f:
        for l in f:
            # there is only one line that eval"s to an array
            for j in eval(l):
                word_list.append(j[0])
    return word_list


def get_template_list(indices=None):
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    templates = [
        "The {} said that",
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
        "The {} yelled because",
    ]
    if indices:
        subset_templates = [templates[i - 1] for i in indices]
        print("subset of templates:", subset_templates)
        return subset_templates

    return templates


def get_intervention_types():
    return [
        "man_direct",
        "man_indirect",
        "woman_direct",
        "woman_indirect",
    ]


def construct_interventions(base_sent, professions, tokenizer, DEVICE):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    for p in professions:
        all_word_count += 1
        try:
            interventions[p] = Intervention(
                tokenizer, base_sent, [p, "man", "woman"], ["he", "she"], device=DEVICE
            )
            used_word_count += 1
        except:
            pass
    print(
        "\t Only used {}/{} professions due to tokenizer".format(
            used_word_count, all_word_count
        )
    )
    return interventions


def run_all(
    model_type="gpt2",
    device="cuda",
    out_dir=".",
    random_weights=False,
    template_indices=None,
):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations.
    professions = get_profession_list()
    templates = get_template_list(template_indices)
    intervention_types = get_intervention_types()
    # Initialize Model and Tokenizer.
    model = Model(device=device, gpt2_version=model_type, random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                 TransfoXLTokenizer if model.is_txl else
                 XLNetTokenizer if model.is_xlnet else
                 BertTokenizer if model.is_bert else
                 DistilBertTokenizer if model.is_distilbert else
                 RobertaTokenizer).from_pretrained(model_type)

    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over all possible templates.
    for temp in templates:
        print("Running template '{}' now...".format(temp), flush=True)
        # Fill in all professions into current template
        interventions = construct_interventions(temp, professions, tokenizer, device)
        # Consider all the intervention types
        for itype in intervention_types:
            print("\t Running with intervention: {}".format(itype), flush=True)
            # Run actual exp.
            intervention_results = model.neuron_intervention_experiment(
                interventions, itype, alpha=1.0
            )

            df = convert_results_to_pd(interventions, intervention_results)
            # Generate file name.
            temp_string = "_".join(temp.replace("{}", "X").split())
            model_type_string = model_type
            fname = "_".join([temp_string, itype, model_type_string])
            # Finally, save each exp separately.
            df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_all(
        opt.model,
        device,
        opt.out_dir,
        random_weights=opt.randomize,
        template_indices=opt.template_indices,
    )
