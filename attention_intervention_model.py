"""
Changes the huggingface transformer attention module to allow interventions
in the attention distribution.
"""
import torch
import torch.nn as nn
import math


class AttentionOverride(nn.Module):
    """A copy of `modeling_gpt2.Attention` class, but with overridden attention values"""

    def __init__(self, attention, attn_override, attn_override_mask):
        """
        Args:
            attention: instance of modeling_gpt2.Attention from which variables will be
                       copied.
            attn_override: values to override the computed attention weights.
                           Shape is [num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override.
                                Shape is [num_heads, seq_len, seq_len]
        """
        super(AttentionOverride, self).__init__()
        # Copy values from attention
        self.output_attentions = attention.output_attentions
        self.register_buffer("bias", attention._buffers["bias"])
        self.n_head = attention.n_head
        self.split_size = attention.split_size
        self.scale = attention.scale
        self.c_attn = attention.c_attn
        self.c_proj = attention.c_proj
        self.attn_dropout = attention.attn_dropout
        self.resid_dropout = attention.resid_dropout
        # Set attention override values
        self.attn_override = attn_override
        self.attn_override_mask = attn_override_mask

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        # attn_override and attn_override_mask are of shape
        # (batch_size, num_heads, override_seq_len, override_seq_len)
        # where override_seq_len is the length of subsequence for which attention is
        # being overridden.
        override_seq_len = self.attn_override_mask.shape[-1]
        w[:, :, :override_seq_len, :override_seq_len] = torch.where(
            self.attn_override_mask,
            self.attn_override,
            w[:, :, :override_seq_len, :override_seq_len],
        )

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = (
                layer_past[0].transpose(-2, -1),
                layer_past[1],
            )  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack(
            (key.transpose(-2, -1), value)
        )  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)
