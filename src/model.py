from transformers import MT5Tokenizer
from transformers.models.mt5.configuration_mt5 import MT5Config
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
try:
    from . import data_preprocess
    from .fid_t5 import FiDT5
except ImportError:
    import data_preprocess
    from fid_t5 import FiDT5
import os


class MT5ForConditionalGeneration(FiDT5):
    r"""
    This class overrides [`T5ForConditionalGeneration`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer

    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> with tokenizer.as_target_tokenizer():
    ...     labels = tokenizer(summary, return_tensors="pt")

    >>> outputs = model(**inputs, labels=labels["input_ids"])
    >>> loss = outputs.loss
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]


class LegalGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

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

    def generate(self, input_ids, eos_token_id=None):
        return self.model.generate(input_ids, eos_token_id=eos_token_id, max_length=128)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    print(torch.cuda.is_available())
    model_path = '../checkpoint/best_.pkl'
    device = 'cuda'
    model_name = 'mt5-base'
    model = LegalGenerator(model_name)
    model.resize_token(250232)
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(list(reversed(['<law_{}>'.format(key) for key in sorted(data_preprocess.get_all_articles().keys())])))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['net'])
    model.to(device)
    print('loaded.')
    x = ['一句话', '一句话', '一句话', '一句话']
    inputs = tokenizer(x, padding='max_length', max_length=512, truncation=True, return_tensors='pt').to(device)
    outputs = tokenizer(x, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)
    print(inputs)
    model(**inputs, labels=outputs['input_ids'])
