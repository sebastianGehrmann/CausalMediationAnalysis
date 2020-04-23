## Mediation Analysis

This repository contains the code to replicate the experiments for the paper "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias".

### Neuron Experiments

#### Create Analysis CSVs

You can run all the experiments for a given model by running the `run_profession_neuron_experiments.py` script. Just set the `-model` flag to the GPT-2 version you want to use and point `-out_dir` to the base directory for your results. The resulting CSV's will be saved in `${out_dir}/results/${date}_neuron_intervention`.

#### Compute aggregate effects

If you want to compute the aggregate effect for each neuron, you can run `compute_and_save_neuron_agg_effect.py`, which will create a new file in `results/${date}_neuron_intervention` called `${model_name}_neuron_effects.csv` with the results.

Similarly, we provide two scripts `compute_neuron_split_total_effect` and `compute_neuron_total_effect` that will report the total effects for a model.

### Sparsity Experiments

#### Attention head selection

You can run experiments for attention head sparsity with `attention_intervention_subset_selection.py` using either Top-k or Greedy algorithm. Results are stored in `{out_dir}/{algo}_{model_type}_{data}.pickle`. 

Additionally, intermediate results will be cached in `{out_dir}/{algo}_intermediate_{model_type}_{data}.pickle` and mean effect (for the entire model, each layer and each head) will be stored in `{out_dir}/mean_effect_{model_type}_{data}.pickle`. 

Script takes in model_type (gpt-2 version), algo (greedy or topk), k (int), data (winobias or winogender) and out_dir (base directory for results).

`python attention_intervention_subset_selection.py --model_type gpt2 --algo greedy --k 10 \
   --data winobias --out_dir results`

#### Neuron selection

You can run experiments for neuron sparsity with `neuron_intervention_subset_selection.py` which outputs results in `{out_dir}/{algo}_{model_type}{_layer}.pickle`. If layer is specified, then neurons are only selected from the specified layer. 

Additionally, the average odds ratio for each layer and each neuron will be stored in `{out_dir}/marg_contrib.pickle`. If `{out_dir}/marg_contrib.pickle` exists, script will use data from this file and not recompute.

Script takes in model_type (gpt-2 version), algo (greedy or topk), k (int), layer (-1 to select neurons from entire model and 0-12 for specific layer) and out_dir (base directory for results). Currently, only compatible with GPT-2.

`python neuron_intervention_subset_selection.py --algo greedy --k 10 \
   --layer -1 --out_dir results`
