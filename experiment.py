
import math
import statistics
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    TransfoXLTokenizer,
    XLNetTokenizer,
    BertForMaskedLM, BertTokenizer,
    DistilBertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer
)

from transformers_modified.modeling_transfo_xl import TransfoXLLMHeadModel
from transformers_modified.modeling_xlnet import XLNetLMHeadModel
from transformers_modified.modeling_distilbert import DistilBertForMaskedLM

from attention_intervention_model import (
    AttentionOverride, TXLAttentionOverride, XLNetAttentionOverride,
    BertAttentionOverride, DistilBertAttentionOverride
)
from utils import batch, convert_results_to_pd

np.random.seed(1)
torch.manual_seed(1)


# Padding text for XLNet (from examples/text-generation/run_generation.py)
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 tokenizer,
                 base_string: str,
                 substitutes: list,
                 candidates: list,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer

        if isinstance(tokenizer, XLNetTokenizer):
            base_string = PADDING_TEXT + ' ' + base_string

        # All the initial strings
        # First item should be neutral, others tainted
        self.base_strings = [base_string.format(s)
                             for s in substitutes]
        # Tokenized bases
        self.base_strings_tok = [
            self.enc.encode(s,
                            add_special_tokens=False,
                            add_space_before_punct_symbol=True)
            for s in self.base_strings
        ]
        # print(self.base_strings_tok)
        self.base_strings_tok = torch.LongTensor(self.base_strings_tok)\
                                     .to(device)
        # Where to intervene
        if isinstance(tokenizer, XLNetTokenizer):
            diff = len(base_string.split()) - base_string.split().index('{}')
            self.position = len(self.base_strings_tok[0]) - diff
            assert len(self.base_strings_tok[0]) == len(self.base_strings_tok[1])
        else:
            self.position = base_string.split().index('{}')

        self.candidates = []
        for c in candidates:
            # 'a ' added to input so that tokenizer understand that first word follows a space.
            tokens = self.enc.tokenize(
                'a ' + c,
                add_space_before_punct_symbol=True)[1:]
            self.candidates.append(tokens)

        self.candidates_tok = [self.enc.convert_tokens_to_ids(tokens)
                               for tokens in self.candidates]


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 device='cpu',
                 output_attentions=False,
                 random_weights=False,
                 masking_approach=1,
                 gpt2_version='gpt2'):
        super()

        self.is_gpt2 = (gpt2_version.startswith('gpt2') or
                        gpt2_version.startswith('distilgpt2'))
        self.is_txl = gpt2_version.startswith('transfo-xl')
        self.is_xlnet = gpt2_version.startswith('xlnet')
        self.is_bert = gpt2_version.startswith('bert')
        self.is_distilbert = gpt2_version.startswith('distilbert')
        self.is_roberta = gpt2_version.startswith('roberta')
        assert (self.is_gpt2 or self.is_txl or self.is_xlnet or
                self.is_bert or self.is_distilbert or self.is_roberta)

        self.device = device
        self.model = (GPT2LMHeadModel if self.is_gpt2 else
                      XLNetLMHeadModel if self.is_xlnet else
                      TransfoXLLMHeadModel if self.is_txl else
                      BertForMaskedLM if self.is_bert else
                      DistilBertForMaskedLM if self.is_distilbert else
                      RobertaForMaskedLM).from_pretrained(
            gpt2_version,
            output_attentions=output_attentions)
        self.model.eval()
        self.model.to(device)
        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        # Options
        self.top_k = 5
        self.num_layers = self.model.config.num_hidden_layers
        self.num_neurons = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.masking_approach = masking_approach # Used only for masked LMs
        assert masking_approach in [1, 2, 3, 4, 5, 6]

        tokenizer = (GPT2Tokenizer if self.is_gpt2 else
                     TransfoXLTokenizer if self.is_txl else
                     XLNetTokenizer if self.is_xlnet else
                     BertTokenizer if self.is_bert else
                     DistilBertTokenizer if self.is_distilbert else
                     RobertaTokenizer).from_pretrained(gpt2_version)
        # Special token id's: (mask, cls, sep)
        self.st_ids = (tokenizer.mask_token_id,
                       tokenizer.cls_token_id,
                       tokenizer.sep_token_id)

        # To account for switched dimensions in model internals:
        # Default: [batch_size, seq_len, hidden_dim],
        # txl and xlnet: [seq_len, batch_size, hidden_dim]
        self.order_dims = lambda a: a

        if self.is_gpt2:
            self.attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_txl:
            self.attention_layer = lambda layer: self.model.transformer.layers[layer].dec_attn
            self.word_emb_layer = self.model.transformer.word_emb
            self.neuron_layer = lambda layer: self.model.transformer.layers[layer].pos_ff
            self.order_dims = lambda a: (a[1], a[0], *a[2:])
        elif self.is_xlnet:
            self.attention_layer = lambda layer: self.model.transformer.layer[layer].rel_attn
            self.word_emb_layer = self.model.transformer.word_embedding
            self.neuron_layer = lambda layer: self.model.transformer.layer[layer].ff
            self.order_dims = lambda a: (a[1], a[0], *a[2:])
        elif self.is_bert:
            self.attention_layer = lambda layer: self.model.bert.encoder.layer[layer].attention.self
            self.word_emb_layer = self.model.bert.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.bert.encoder.layer[layer].output
        elif self.is_distilbert:
            self.attention_layer = lambda layer: self.model.distilbert.transformer.layer[layer].attention
            self.word_emb_layer = self.model.distilbert.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.distilbert.transformer.layer[layer].output_layer_norm
        elif self.is_roberta:
            self.attention_layer = lambda layer: self.model.roberta.encoder.layer[layer].attention.self
            self.word_emb_layer = self.model.roberta.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.roberta.encoder.layer[layer].output

    def mlm_inputs(self, context, candidate):
        """ Return input_tokens for the masked LM sampling scheme """
        input_tokens = []
        for i in range(len(candidate)):
            combined = context + candidate[:i] + [self.st_ids[0]]
            if self.masking_approach in [2, 5]:
                combined = combined + candidate[i+1:]
            elif self.masking_approach in [3, 6]:
                combined = combined + [self.st_ids[0]] * len(candidate[i+1:])
            if self.masking_approach > 3:
                combined = [self.st_ids[1]] + combined + [self.st_ids[2]]
            pred_idx = combined.index(self.st_ids[0])
            input_tokens.append((combined, pred_idx))
        return input_tokens

    def xlnet_forward(self, batch, clen):
        """ Return the outputs of XLNet's forward pass;
            clen = length of the candidate """
        bsz, seqlen = batch.shape
        perm_mask = torch.triu(
            torch.ones((bsz, seqlen, seqlen), device=self.device), diagonal=0)
        perm_mask[:, :, :-clen] = 0
        if self.masking_approach == 2:
            perm_mask[:, -clen:, -clen:] = torch.eye(clen)
        target_mapping = torch.zeros(
            (bsz, clen, seqlen), dtype=torch.float, device=self.device)
        target_mapping[:, :, -clen:] = torch.eye(clen)
        return self.model(batch,
                          perm_mask=perm_mask,
                          target_mapping=target_mapping)

    def get_representations(self, context, position):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            # XLNet: ignore the query stream
            if self.is_xlnet and output.shape[0] == 1: return output
            representations[layer] = output[self.order_dims((0, position))]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            handles.append(self.word_emb_layer.register_forward_hook(
                partial(extract_representation_hook,
                        position=position,
                        representations=representation,
                        layer=-1)))
            # hidden layers
            for layer in range(self.num_layers):
                handles.append(self.neuron_layer(layer).register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            if self.is_xlnet:
                self.xlnet_forward(context.unsqueeze(0), clen=1)
            else:
                self.model(context.unsqueeze(0))
            for h in handles:
                h.remove()
        # print(representation[0][:5])
        return representation

    def get_probabilities_for_examples(self, context, candidates):
        """Return probabilities of single-token candidates given context"""
        for c in candidates:
            if len(c) > 1:
                raise ValueError(f"Multiple tokens not allowed: {c}")
        outputs = [c[0] for c in candidates]
        if self.is_xlnet:
            logits = self.xlnet_forward(context, clen=1)[0]
        else:
            logits = self.model(context)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[:, outputs].tolist()

    def get_probabilities_for_examples_multitoken(self, context, candidates):
        """
        Return probability of multi-token candidates given context.
        Prob of each candidate is normalized by number of tokens.

        Args:
            context: Tensor of token ids in context
            candidates: list of list of token ids in each candidate

        Returns: list containing probability for each candidate
        """
        # TODO: Combine into single batch
        mean_probs = []
        context = context.tolist()
        for candidate in candidates:
            token_log_probs = []
            if self.is_bert or self.is_distilbert or self.is_roberta:
                mlm_inputs = self.mlm_inputs(context, candidate)
                for i, c in enumerate(candidate):
                    combined, pred_idx = mlm_inputs[i]
                    batch = torch.tensor(combined).unsqueeze(dim=0).to(self.device)
                    logits = self.model(batch)[0]
                    log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
                    token_log_probs.append(log_probs[pred_idx][c].item())
            elif self.is_xlnet:
                combined = context + candidate
                batch = torch.tensor(combined).unsqueeze(dim=0).to(self.device)
                logits = self.xlnet_forward(batch, clen=len(candidate))[0]
                log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
                for i, next_token_id in enumerate(candidate):
                    token_log_probs.append(log_probs[i][next_token_id].item())
            else:
                combined = context + candidate
                # Exclude last token position when predicting next token
                batch = torch.tensor(combined[:-1]).unsqueeze(dim=0).to(self.device)
                # Shape (batch_size, seq_len, vocab_size)
                logits = self.model(batch)[0]
                # Shape (seq_len, vocab_size)
                log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
                context_end_pos = len(context) - 1
                continuation_end_pos = context_end_pos + len(candidate)
                # TODO: Vectorize this
                # Up to but not including last token position
                for i in range(context_end_pos, continuation_end_pos):
                    next_token_id = combined[i+1]
                    next_token_log_prob = log_probs[i][next_token_id].item()
                    token_log_probs.append(next_token_log_prob)
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)
            mean_probs.append(mean_token_prob)
        return mean_probs

    def neuron_intervention(self,
                            context,
                            outputs,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_type='diff',
                            alpha=1.):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output,
                              position,
                              neurons,
                              intervention,
                              intervention_type):
            # XLNet: ignore the query stream
            if self.is_xlnet and output.shape[0] == 1: return output
            # Get the neurons to intervene on
            neurons = torch.LongTensor(neurons).to(self.device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            base_slice = self.order_dims((slice(None), position, slice(None)))
            base = output[base_slice].gather(1, neurons)
            intervention_view = intervention.view_as(base)

            if intervention_type == 'replace':
                base = intervention_view
            elif intervention_type == 'diff':
                base += intervention_view
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")
            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(output, dtype=torch.bool)
            for i, v in enumerate(neurons):
                scatter_mask[self.order_dims((i, position, v))] = 1
            # Then take values from base and scatter
            output.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)
        context = context.unsqueeze(0).repeat(batch_size, 1)
        handle_list = []
        for layer in set(layers):
            neuron_loc = np.where(np.array(layers) == layer)[0]
            n_list = []
            for n in neurons:
                unsorted_n_list = [n[i] for i in neuron_loc]
                n_list.append(list(np.sort(unsorted_n_list)))
            intervention_rep = alpha * rep[layer][n_list]
            if layer == -1:
                handle_list.append(self.word_emb_layer.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            intervention_type=intervention_type)))
            else:
                handle_list.append(self.neuron_layer(layer).register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            intervention_type=intervention_type)))
        new_probabilities = self.get_probabilities_for_examples(
            context,
            outputs)
        for hndle in handle_list:
            hndle.remove()
        return new_probabilities

    def head_pruning_intervention(self,
                                  context,
                                  outputs,
                                  layer,
                                  head):
        # Recreate model and prune head
        save_model = self.model
        # TODO Make this more efficient
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.prune_heads({layer: [head]})
        self.model.eval()

        # Compute probabilities without head
        new_probabilities = self.get_probabilities_for_examples(
            context,
            outputs)

        # Reinstate original model
        # TODO Handle this in cleaner way
        self.model = save_model

        return new_probabilities

    def attention_intervention(self,
                               context,
                               outputs,
                               attn_override_data):
        """ Override attention values in specified layer

        Args:
            context: context text
            outputs: candidate outputs
            attn_override_data: list of dicts of form:
                {
                    'layer': <index of layer on which to intervene>,
                    'attention_override': <values to override the computed attention weights.
                           Shape is [batch_size, num_heads, seq_len, seq_len]>,
                    'attention_override_mask': <indicates which attention weights to override.
                                Shape is [batch_size, num_heads, seq_len, seq_len]>
                }
        """

        def intervention_hook(module, input, outputs, attn_override, attn_override_mask):
            attention_override_module = (AttentionOverride if self.is_gpt2 else
                                         TXLAttentionOverride if self.is_txl else
                                         XLNetAttentionOverride if self.is_xlnet else
                                         BertAttentionOverride if self.is_bert else
                                         DistilBertAttentionOverride if self.is_distilbert else
                                         BertAttentionOverride)(
                module, attn_override, attn_override_mask)
            return attention_override_module(*input)

        with torch.no_grad():
            if self.is_bert or self.is_distilbert or self.is_roberta:
                k = 0
                new_probabilities = []
                context = context.tolist()
                for candidate in outputs:
                    token_log_probs = []
                    mlm_inputs = self.mlm_inputs(context, candidate)
                    for i, c in enumerate(candidate):
                        hooks = []
                        for d in attn_override_data:
                            hooks.append(self.attention_layer(d['layer']).register_forward_hook(
                                partial(intervention_hook,
                                        attn_override=d['attention_override'][k],
                                        attn_override_mask=d['attention_override_mask'][k])))

                        combined, pred_idx = mlm_inputs[i]
                        batch = torch.tensor(combined).unsqueeze(dim=0).to(self.device)
                        logits = self.model(batch)[0]
                        log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
                        token_log_probs.append(log_probs[pred_idx][c].item())

                        for hook in hooks: hook.remove()
                        k += 1

                    mean_token_log_prob = statistics.mean(token_log_probs)
                    mean_token_prob = math.exp(mean_token_log_prob)
                    new_probabilities.append(mean_token_prob)
            else:
                hooks = []
                for d in attn_override_data:
                    attn_override = d['attention_override']
                    attn_override_mask = d['attention_override_mask']
                    layer = d['layer']
                    hooks.append(self.attention_layer(layer).register_forward_hook(
                        partial(intervention_hook,
                                attn_override=attn_override,
                                attn_override_mask=attn_override_mask)))

                new_probabilities = self.get_probabilities_for_examples_multitoken(
                    context,
                    outputs)

                for hook in hooks:
                    hook.remove()

            return new_probabilities

    def neuron_intervention_experiment(self,
                                       word2intervention,
                                       intervention_type,
                                       layers_to_adj=[],
                                       neurons_to_adj=[],
                                       alpha=1,
                                       intervention_loc='all'):
        """
        run multiple intervention experiments
        """

        word2intervention_results = {}
        for word in tqdm(word2intervention, desc='words'):
            word2intervention_results[word] = self.neuron_intervention_single_experiment(
                word2intervention[word], intervention_type, layers_to_adj, neurons_to_adj,
                alpha, intervention_loc=intervention_loc)

        return word2intervention_results

    def neuron_intervention_single_experiment(self,
                                              intervention,
                                              intervention_type, layers_to_adj=[],
                                              neurons_to_adj=[],
                                              alpha=100,
                                              bsize=800, intervention_loc='all'):
        """
        run one full neuron intervention experiment
        """

        if self.is_txl or self.is_xlnet: 32 # to avoid GPU memory error
        with torch.no_grad():
            '''
            Compute representations for base terms (one for each side of bias)
            '''
            if self.is_bert or self.is_distilbert or self.is_roberta or self.is_xlnet:
                num_alts = intervention.base_strings_tok.shape[0]
                masks = torch.tensor([self.st_ids[0]]).repeat(num_alts, 1).to(self.device)
                intervention.base_strings_tok = torch.cat(
                    (intervention.base_strings_tok, masks), dim=1)

            base_representations = self.get_representations(
                intervention.base_strings_tok[0],
                intervention.position)
            man_representations = self.get_representations(
                intervention.base_strings_tok[1],
                intervention.position)
            woman_representations = self.get_representations(
                intervention.base_strings_tok[2],
                intervention.position)

            # TODO: this whole logic can probably be improved
            # determine effect type and set representations

            # e.g. The teacher said that
            if intervention_type == 'man_minus_woman':
                context = intervention.base_strings_tok[0]
                rep = {k: v - woman_representations[k]
                       for k, v in man_representations.items()}
                replace_or_diff = 'diff'
            # e.g. The teacher said that
            elif intervention_type == 'woman_minus_man':
                context = intervention.base_strings_tok[0]
                rep = {k: v - man_representations[k]
                       for k, v in woman_representations.items()}
                replace_or_diff = 'diff'
            # e.g. The man said that
            elif intervention_type == 'man_direct':
                context = intervention.base_strings_tok[1]
                rep = base_representations
                replace_or_diff = 'replace'
            # e.g. The teacher said that
            elif intervention_type == 'man_indirect':
                context = intervention.base_strings_tok[0]
                rep = man_representations
                replace_or_diff = 'replace'
            # e.g. The woman said that
            elif intervention_type == 'woman_direct':
                context = intervention.base_strings_tok[2]
                rep = base_representations
                replace_or_diff = 'replace'
            # e.g. The teacher said that
            elif intervention_type == 'woman_indirect':
                context = intervention.base_strings_tok[0]
                rep = woman_representations
                replace_or_diff = 'replace'
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")

            # Probabilities without intervention (Base case)
            candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[0].unsqueeze(0),
                intervention.candidates_tok)[0]
            candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[1].unsqueeze(0),
                intervention.candidates_tok)[0]
            candidate1_alt2_prob, candidate2_alt2_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[2].unsqueeze(0),
                intervention.candidates_tok)[0]
            # Now intervening on potentially biased example
            if intervention_loc == 'all':
              candidate1_probs = torch.zeros((self.num_layers + 1, self.num_neurons))
              candidate2_probs = torch.zeros((self.num_layers + 1, self.num_neurons))

              for layer in range(-1, self.num_layers):
                for neurons in batch(range(self.num_neurons), bsize):
                    neurons_to_search = [[i] + neurons_to_adj for i in neurons]
                    layers_to_search = [layer] + layers_to_adj

                    probs = self.neuron_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        rep=rep,
                        layers=layers_to_search,
                        neurons=neurons_to_search,
                        position=intervention.position,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
                    for neuron, (p1, p2) in zip(neurons, probs):
                        candidate1_probs[layer + 1][neuron] = p1
                        candidate2_probs[layer + 1][neuron] = p2
                        # Now intervening on potentially biased example
            elif intervention_loc == 'layer':
              layers_to_search = (len(neurons_to_adj) + 1)*[layers_to_adj]
              candidate1_probs = torch.zeros((1, self.num_neurons))
              candidate2_probs = torch.zeros((1, self.num_neurons))

              for neurons in batch(range(self.num_neurons), bsize):
                neurons_to_search = [[i] + neurons_to_adj for i in neurons]

                probs = self.neuron_intervention(
                    context=context,
                    outputs=intervention.candidates_tok,
                    rep=rep,
                    layers=layers_to_search,
                    neurons=neurons_to_search,
                    position=intervention.position,
                    intervention_type=replace_or_diff,
                    alpha=alpha)
                for neuron, (p1, p2) in zip(neurons, probs):
                    candidate1_probs[0][neuron] = p1
                    candidate2_probs[0][neuron] = p2
            else:
              probs = self.neuron_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        rep=rep,
                        layers=layers_to_adj,
                        neurons=neurons_to_adj,
                        position=intervention.position,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
              for neuron, (p1, p2) in zip(neurons_to_adj, probs):
                  candidate1_probs = p1
                  candidate2_probs = p2


        return (candidate1_base_prob, candidate2_base_prob,
                candidate1_alt1_prob, candidate2_alt1_prob,
                candidate1_alt2_prob, candidate2_alt2_prob,
                candidate1_probs, candidate2_probs)

    def attention_intervention_experiment(self, intervention, effect):
        """
        Run one full attention intervention experiment
        measuring indirect or direct effect.
        """
        # E.g. The doctor asked the nurse a question. He
        x = intervention.base_strings_tok[0]
        # E.g. The doctor asked the nurse a question. She
        x_alt = intervention.base_strings_tok[1]

        if effect == 'indirect':
            input = x_alt  # Get attention for x_alt
        elif effect == 'direct':
            input = x  # Get attention for x
        else:
            raise ValueError(f"Invalid effect: {effect}")
        if self.is_bert or self.is_distilbert or self.is_roberta:
            attention_override = []
            input = input.tolist()
            for candidate in intervention.candidates_tok:
                mlm_inputs = self.mlm_inputs(input, candidate)
                for i, c in enumerate(candidate):
                    combined, _ = mlm_inputs[i]
                    batch = torch.tensor(combined).unsqueeze(0).to(self.device)
                    attention_override.append(self.model(batch)[-1])
        elif self.is_xlnet:
            batch = input.clone().detach().unsqueeze(0).to(self.device)
            target_mapping = torch.zeros(
                (1, 1, len(input)), dtype=torch.float, device=self.device)
            attention_override = self.model(
                batch, target_mapping=target_mapping)[-1]
        else:
            batch = input.clone().detach().unsqueeze(0).to(self.device)
            attention_override = self.model(batch)[-1]

        batch_size = 1
        seq_len = len(x)
        seq_len_alt = len(x_alt)
        assert seq_len == seq_len_alt

        with torch.no_grad():

            candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate1_probs_layer = torch.zeros(self.num_layers)
            candidate2_probs_layer = torch.zeros(self.num_layers)

            if effect == 'indirect':
                context = x
            else:
                context = x_alt

            # Intervene at every layer and head by overlaying attention induced by x_alt
            model_attn_override_data = [] # Save layer interventions for model-level intervention later
            for layer in range(self.num_layers):
                if self.is_bert or self.is_distilbert or self.is_roberta:
                    layer_attention_override = [a[layer] for a in attention_override]
                    attention_override_mask = [torch.ones_like(l, dtype=torch.uint8) for l in layer_attention_override]
                elif self.is_xlnet:
                    layer_attention_override = attention_override[layer]
                    attention_override_mask = torch.ones_like(layer_attention_override[0], dtype=torch.uint8)
                else:
                    layer_attention_override = attention_override[layer]
                    attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8)
                layer_attn_override_data = [{
                    'layer': layer,
                    'attention_override': layer_attention_override,
                    'attention_override_mask': attention_override_mask
                }]
                candidate1_probs_layer[layer], candidate2_probs_layer[layer] = self.attention_intervention(
                    context=context,
                    outputs=intervention.candidates_tok,
                    attn_override_data = layer_attn_override_data)
                model_attn_override_data.extend(layer_attn_override_data)
                for head in range(self.num_heads):
                    if self.is_bert or self.is_distilbert or self.is_roberta:
                        attention_override_mask = [torch.zeros_like(l, dtype=torch.uint8)
                                                   for l in layer_attention_override]
                        for a in attention_override_mask: a[0][head] = 1
                    elif self.is_xlnet:
                        attention_override_mask = torch.zeros_like(layer_attention_override[0], dtype=torch.uint8)
                        attention_override_mask[0][head] = 1
                    else:
                        attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                        attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
                    head_attn_override_data = [{
                        'layer': layer,
                        'attention_override': layer_attention_override,
                        'attention_override_mask': attention_override_mask
                    }]
                    candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        attn_override_data=head_attn_override_data)

            # Intervene on entire model by overlaying attention induced by x_alt
            candidate1_probs_model, candidate2_probs_model = self.attention_intervention(
                context=context,
                outputs=intervention.candidates_tok,
                attn_override_data=model_attn_override_data)

        return candidate1_probs_head, candidate2_probs_head, candidate1_probs_layer, candidate2_probs_layer,\
            candidate1_probs_model, candidate2_probs_model

    def attention_intervention_single_experiment(self, intervention, effect, layers_to_adj, heads_to_adj, search):
        """
        Run one full attention intervention experiment
        measuring indirect or direct effect.
        """
        # E.g. The doctor asked the nurse a question. He
        x = intervention.base_strings_tok[0]
        # E.g. The doctor asked the nurse a question. She
        x_alt = intervention.base_strings_tok[1]

        if effect == 'indirect':
            input = x_alt  # Get attention for x_alt
        elif effect == 'direct':
            input = x  # Get attention for x
        else:
            raise ValueError(f"Invalid effect: {effect}")
        batch = torch.tensor(input).unsqueeze(0).to(self.device)
        attention_override = self.model(batch)[-1]

        batch_size = 1
        seq_len = len(x)
        seq_len_alt = len(x_alt)
        assert seq_len == seq_len_alt
        assert len(attention_override) == self.num_layers
        assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len)

        with torch.no_grad():
            if search:
                candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
                candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))

            if effect == 'indirect':
                context = x
            else:
                context = x_alt

            model_attn_override_data = []
            for layer in range(self.num_layers):
                if layer in layers_to_adj:
                    layer_attention_override = attention_override[layer]

                    layer_ind = np.where(layers_to_adj == layer)[0]
                    heads_in_layer = heads_to_adj[layer_ind]
                    attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                    # set multiple heads in layer to 1
                    for head in heads_in_layer:
                        attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
                    # get head mask
                    head_attn_override_data = [{
                        'layer': layer,
                        'attention_override': layer_attention_override,
                        'attention_override_mask': attention_override_mask
                    }]
                    # should be the same length as the number of unique layers to adj
                    model_attn_override_data.extend(head_attn_override_data)

            # basically generate the mask for the layers_to_adj and heads_to_adj
            if search:
                for layer in range(self.num_layers):
                  layer_attention_override = attention_override[layer]
                  layer_ind = np.where(layers_to_adj == layer)[0]
                  heads_in_layer = heads_to_adj[layer_ind]

                  for head in range(self.num_heads):
                    if head not in heads_in_layer:
                          model_attn_override_data_search = []
                          attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                          heads_list = [head]
                          if len(heads_in_layer) > 0:
                            heads_list.extend(heads_in_layer)
                          for h in (heads_list):
                              attention_override_mask[0][h] = 1 # Set mask to 1 for single head only
                          head_attn_override_data = [{
                              'layer': layer,
                              'attention_override': layer_attention_override,
                              'attention_override_mask': attention_override_mask
                          }]
                          model_attn_override_data_search.extend(head_attn_override_data)
                          for override in model_attn_override_data:
                              if override['layer'] != layer:
                                  model_attn_override_data_search.append(override)

                          candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
                              context=context,
                              outputs=intervention.candidates_tok,
                              attn_override_data=model_attn_override_data_search)
                    else:
                        candidate1_probs_head[layer][head] = -1
                        candidate2_probs_head[layer][head] = -1


            else:
              candidate1_probs_head, candidate2_probs_head = self.attention_intervention(
                  context=context,
                  outputs=intervention.candidates_tok,
                  attn_override_data=model_attn_override_data)

        return candidate1_probs_head, candidate2_probs_head


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(device=DEVICE)

    base_sentence = "The {} said that"
    biased_word = "teacher"
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"],
            device=DEVICE)
    interventions = {biased_word: intervention}

    intervention_results = model.neuron_intervention_experiment(
        interventions, 'man_minus_woman')
    df = convert_results_to_pd(
        interventions, intervention_results)
    print('more probable candidate per layer, across all neurons in the layer')
    print(df[0:5])


if __name__ == "__main__":
    main()
