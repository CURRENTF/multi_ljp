# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import os
from copy import deepcopy
import re
import collections
from transformers import MT5Tokenizer, MT5Config
try:
    from . import data_preprocess
    from .modified_origin_code import modeling_mt5, modeling_t5
except ImportError:
    import data_preprocess
    from modified_origin_code import modeling_mt5, modeling_t5




class FiDT5(modeling_mt5.MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('__init__ finished.')
    
    def init_wrap(self):
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            # print('input shape of input_ids', input_ids.shape)
            if input_ids.dim() == 3:
                # print('here should came...')
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        print(type(self.encoder))
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.main_input_name = self.encoder.main_input_name
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.embed_tokens = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


class LegalGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = FiDT5.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, past_key_values=None):
        if decoder_input_ids is None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
            )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )

    def resize_token(self, n):
        self.model.resize_token_embeddings(n)

    def generate(self, input_ids, attention_mask, eos_token_id=None):
        return self.model.generate(input_ids, attention_mask, max_length=128)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    print(torch.cuda.is_available())
    model_path = '../checkpoint/best_.pkl'
    device = 'cuda'
    model_name = 'google/mt5-base'
    state = torch.load(model_path, map_location=device)
    parameters = state['net']
    print(type(parameters))
    para_ = collections.OrderedDict()
    G = open('./give.txt', mode='w', encoding='utf-8')
    for key in parameters:
        if re.search(r'encoder\.(block\.[0-9]{1,2}\.layer)?', key):
            key_ = key.replace('encoder.', 'encoder.encoder.').replace('.layer.', '.module.layer.')
            para_[key_] = parameters[key]
            if key_ == 'model.encoder.encoder.final_layer_norm.weight':
                para_['model.encoder.shared.weight'] = para_['model.encoder.encoder.embed_tokens.weight']
                para_['model.encoder.embed_tokens.weight'] = para_['model.encoder.encoder.embed_tokens.weight']
            # print(key_, file=G)
        else:
            para_[key] = parameters[key]
            # print(key, file=G)
        
    parameters = para_
    config = MT5Config
    model = LegalGenerator(model_name)
    # model.model.init_wrap()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IGNORE ABOVE')
    model.resize_token(250232)
    para_dict = model.state_dict()
    O = open('./need.txt', mode='w', encoding='utf-8')
    for key in para_dict:
        print(key, ':::::', para_dict[key].shape)
        # print(key, file=O)
        
    model.load_state_dict(parameters)
    model.to(device)

    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(list(reversed(['<law_{}>'.format(key) for key in sorted(data_preprocess.get_all_articles().keys())])))
    
    x = ['一句话', '一句话', '一句话', '一句话']
    inputs = tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt').to(device)
    outputs = tokenizer(x, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)
    print(inputs)
    inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
    inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
    out = model(**inputs, labels=outputs['input_ids'])
    print(out.loss)
