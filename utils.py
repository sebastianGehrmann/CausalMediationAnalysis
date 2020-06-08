from copy import deepcopy

import pandas as pd


def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx : min(ndx + bsize, total_len)])


def convert_results_to_pd(
    interventions, intervention_results, layer_fixed=None, neuron_fixed=None
):
    """
    Convert intervention results to data frame

    Args:
        interventions: dictionary from word (e.g., profession) to intervention
        intervention_results: dictionary from word to intervention results
    """

    results = []
    for word in intervention_results:
        intervention = interventions[word]
        (
            candidate1_base_prob,
            candidate2_base_prob,
            candidate1_alt1_prob,
            candidate2_alt1_prob,
            candidate1_alt2_prob,
            candidate2_alt2_prob,
            candidate1_probs,
            candidate2_probs,
        ) = intervention_results[word]
        # we have results for all layers and all neurons
        results_base = {  # strings
            "word": word,
            "base_string": intervention.base_strings[0],
            "alt_string1": intervention.base_strings[1],
            "alt_string2": intervention.base_strings[2],
            "candidate1": intervention.candidates[0],
            "candidate2": intervention.candidates[1],
            # base probs
            "candidate1_base_prob": float(candidate1_base_prob),
            "candidate2_base_prob": float(candidate2_base_prob),
            "candidate1_alt1_prob": float(candidate1_alt1_prob),
            "candidate2_alt1_prob": float(candidate2_alt1_prob),
            "candidate1_alt2_prob": float(candidate1_alt2_prob),
            "candidate2_alt2_prob": float(candidate2_alt2_prob),
        }
        if layer_fixed is None:
            for layer in range(candidate1_probs.size(0)):
                for neuron in range(candidate1_probs.size(1)):
                    c1_prob, c2_prob = (
                        candidate1_probs[layer][neuron],
                        candidate2_probs[layer][neuron],
                    )
                    results_single = deepcopy(results_base)
                    results_single.update(
                        {  # strings
                            # intervention probs
                            "candidate1_prob": float(c1_prob),
                            "candidate2_prob": float(c2_prob),
                            "layer": layer,
                            "neuron": neuron,
                        }
                    )
                    results.append(results_single)
        # we have results for all neurons in one layer
        elif neuron_fixed is None:
            for neuron in range(candidate1_probs.size(1)):
                c1_prob, c2_prob = (
                    candidate1_probs[0][neuron],
                    candidate2_probs[0][neuron],
                )
                results_single = deepcopy(results_base)
                results_single.update(
                    {  # strings
                        # intervention probs
                        "candidate1_prob": float(c1_prob),
                        "candidate2_prob": float(c2_prob),
                        "layer": layer_fixed,
                        "neuron": neuron,
                    }
                )
                results.append(results_single)
        # we have result for a specific neuron and layer
        else:
            c1_prob, c2_prob = candidate1_probs, candidate2_probs
            results_single = deepcopy(results_base)
            results_single.update(
                {  # strings
                    # intervention probs
                    "candidate1_prob": float(c1_prob),
                    "candidate2_prob": float(c2_prob),
                    "layer": layer_fixed,
                    "neuron": neuron_fixed,
                }
            )
            results.append(results_single)
    return pd.DataFrame(results)
