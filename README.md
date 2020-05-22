## Mediation Analysis

This repository contains the code to replicate the experiments for the paper [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/abs/2004.12265).

### Neuron Experiments

#### Create Analysis CSVs

You can run all the experiments for a given model by running the `run_profession_neuron_experiments.py` script. Just set the `-model` flag to the GPT-2 version you want to use and point `-out_dir` to the base directory for your results. The resulting CSV's will be saved in `${out_dir}/results/${date}_neuron_intervention`.

#### Compute total effect and correlation with professions

We provide two scripts `compute_neuron_split_total_effect` and `compute_neuron_total_effect` that will report the total effects for a model in multiple different ways.

`compute_neural_total_effect` will additionally compute the correlational value between effect sizes and the bias value of the profession and generate a plot in `${out_dir}/neuron_profession_correlation.pdf`.

#### Compute aggregate neuron effects

If you want to compute the aggregate effect for each neuron, you can run `compute_and_save_neuron_agg_effect.py`, which will create a new file in `results/${date}_neuron_intervention` called `${model_name}_neuron_effects.csv` with the results.

After you have run this for each of the models you want to investigate, you can run `compute_neuron_effect_per_layer.py` which will generate plots of the per-layer effects.
One aggregate plot will be at `${out_dir}/neuron_layer_effect.pdf` and a separate plot for each model will be saved at `${out_dir}/neuron_layer_effect_${model_name}.pdf`.

### Attention Experiments

#### Create Analysis JSON files

Note: the analysis JSON files for winogender and winobias are already available under the `winogender_data` and `winobias_data` directories respectively, so you may disregard the following instructions if you wish. The raw Winogender and Winobias datasets (the non-json datasets in those same directories) were obtained from https://github.com/rudinger/winogender-schemas and from https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino/data respectively.

If you wish to recreate the analysis files from scratch, you can run the attention intervention experiments for a specific configuration by running either the `attention_intervention_winobias.py` or `attention_intervention_winogender.py` scripts. The arguments are specified in the respective script in the `intervene_attention` method. See `attention_intervention_winobias.sh` or `attention_intervention_winogender.sh` for all possible configurations. The results will be written to the `winobias_data/` or `winogender_data/` directory.

#### Generate reports

Various reports can be generated from the JSON files by running `attention_figures1.py`,
`attention_figures2.py`, or `attention_figures3.py.`
See the respective script for a description of the reports generated. You may want to modify these scripts to only generate figures for a subset of configurations. The results are written as pdf files to subfolders in the `results/` directory.

### Sparsity Experiments

#### Attention head selection

You can run experiments for attention head sparsity with `attention_intervention_subset_selection.py` using either Top-k or Greedy algorithm. Results are stored in `{out_dir}/{algo}_{model_type}_{data}.pickle`.

Additionally, intermediate results will be cached in `{out_dir}/{algo}_intermediate_{model_type}_{data}.pickle` and mean effect (for the entire model, each layer and each head) will be stored in `{out_dir}/mean_effect_{model_type}_{data}.pickle`.

Script takes in model_type (gpt-2 version), algo (greedy or topk), k (int), data (winobias or winogender) and out_dir (base directory for results).

`python attention_intervention_subset_selection.py --model_type gpt2 --algo greedy --k 10 \ --data winobias --out_dir results`

#### Neuron selection

You can run experiments for neuron sparsity with `neuron_intervention_subset_selection.py` which outputs results in `{out_dir}/{algo}_{model_type}{_layer}.pickle`. If layer is specified, then neurons are only selected from the specified layer.

Additionally, the average odds ratio for each layer and each neuron will be stored in `{out_dir}/marg_contrib.pickle`. If `{out_dir}/marg_contrib.pickle` exists, script will use data from this file and not recompute.

Script takes in model_type (gpt-2 version), algo (greedy or topk), k (int), layer (-1 to select neurons from entire model and 0-12 for specific layer) and out_dir (base directory for results). Currently, only compatible with GPT-2.

`python neuron_intervention_subset_selection.py --algo greedy --k 10 \ --layer -1 --out_dir results`
