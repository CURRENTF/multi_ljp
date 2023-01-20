from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mt5.configuration_mt5 import MT5Config
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
try:
    from . import data_preprocess
except ImportError:
    import data_preprocess
import os


class MyFID(MT5ForConditionalGeneration):

    def forward_(self):
        pass


    def forward(
        self,
        n_passage=4,
        batch_size=1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids_list = input_ids
        del input_ids
        attention_mask_list = attention_mask
        del attention_mask

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        print('shape of input_ids_list for debug\n', input_ids_list.shape)
        # batch size 固定 = 1
        encoder_outputs_list = []
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            # print('n_passage display for debug\n', input_ids)
            input_ids = torch.unsqueeze(input_ids, 0)
            attention_mask = torch.unsqueeze(attention_mask, 0)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_outputs_list.append(encoder_outputs)
            
        n_hidden_states = torch.cat([encoder_outputs[0] for encoder_outputs in encoder_outputs_list], dim=1)
        n_attention_mask = torch.cat([attention_mask.unsqueeze(0) for attention_mask in attention_mask_list], dim=1)
        print('shape of n_attention_mask for debug\n', n_attention_mask.shape)
        print('shape of n_hidden_states for debug\n', n_hidden_states.shape)
        # input()
        # n_states = n_hidden_states[]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        print('shape of decoder_input_ids for debug\n', decoder_input_ids.shape)
        print('use_cache, return_dict', use_cache, return_dict)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=n_hidden_states,
            encoder_attention_mask=n_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            use_cache=use_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # if not return_dict:
        #     print('lets see return dict is {}'.format(return_dict))
        #     output = (lm_logits,) + decoder_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs0.last_hidden_state,
            encoder_hidden_states=encoder_outputs0.hidden_states,
            encoder_attentions=encoder_outputs0.attentions,
        )


class LegalGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = MyFID.from_pretrained(model_name)

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
    model_name = 'google/mt5-base'
    model = LegalGenerator(model_name)
    model.resize_token(250232)
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(list(reversed(['<law_{}>'.format(key) for key in sorted(data_preprocess.get_all_articles().keys())])))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['net'])
    model.to(device)
    print('loaded.')
    x = ['一句话', '一句话', '一句话', '一句话']
    inputs = tokenizer(x, padding='max_length', max_length=45, truncation=True, return_tensors='pt').to(device)
    outputs = tokenizer(x, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(device)
    print(inputs)
    model(**inputs, labels=outputs['input_ids'])
