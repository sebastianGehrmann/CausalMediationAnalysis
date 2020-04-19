## Mediation Analysis

This repository contains the code to replicate the experiments for the paper "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias".

### Neuron Experiments

#### Create Analysis CSVs

You can run all the experiments for a given model by running the `run_profession_neuron_experiments.py` script. Just set the `-model` flag to the GPT-2 version you want to use and point `-out_dir` to the base directory for your results. The resulting CSV's will be saved in `${out_dir}/results/${date}_neuron_intervention`.

#### Compute aggregate effects

If you want to compute the aggregate effect for each neuron, you can run `compute_and_save_neuron_agg_effect.py`, which will create a new file in `results/${date}_neuron_intervention` called `${model_name}_neuron_effects.csv` with the results.

Similarly, we provide two scripts `compute_neuron_split_total_effect` and `compute_neuron_total_effect` that will report the total effects for a model.
