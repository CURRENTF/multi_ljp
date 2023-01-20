import copy
import random
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
from collections import Counter, OrderedDict
import re
from tqdm import *
import numpy as np
import time
from sklearn.metrics import *
from gensim.models.word2vec import Word2Vec
from typing import List
import argparse
from src.fid_t5 import LegalGenerator
from src import data_preprocess
import src.train_tools as ttool
from src import data
from transformers import MT5Tokenizer, MT5Config
import sys
# import transformers.generation_utils as 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# TRAIN SETTINGS
epoch = 8
batch_size = 1
model_name = 'google/mt5-base'
model_path = './checkpoint/mt5_.pkl'
gradient_accu = 50
ex_name = 'complete'

if __name__ == '__main__':
    train_loader = data.get_dataloader(data.load_data('./data/train_data_{}.jsonl'.format(ex_name)))
    valid_loader = data.get_dataloader(data.load_data('./data/valid_data_{}.jsonl'.format(ex_name)))

    try:
        model = LegalGenerator(model_name)
        tokenizer = MT5Tokenizer.from_pretrained(model_name)
    except:
        model = LegalGenerator('/data/DataSet_IRLab/huggingface/mt5-base')
        tokenizer = MT5Tokenizer.from_pretrained('/data/DataSet_IRLab/huggingface/mt5-base')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IGNORE ABOVE')
    model.resize_token(250232)
    
    tokenizer.add_tokens(list(reversed(['<law_{}>'.format(key) for key in sorted(data_preprocess.get_all_articles().keys())])))
    

    state = torch.load(model_path, map_location=device)
    parameters = state['net']
    if model_path == './checkpoint/mt5_.pkl':

        para_ = OrderedDict()
        for key in parameters:
            if re.search(r'encoder\.(block\.[0-9]{1,2}\.layer)?', key):
                key_ = key.replace('encoder.', 'encoder.encoder.').replace('.layer.', '.module.layer.')
                para_[key_] = parameters[key]
                if key_ == 'model.encoder.encoder.final_layer_norm.weight':
                    para_['model.encoder.shared.weight'] = para_['model.encoder.encoder.embed_tokens.weight']
                    para_['model.encoder.embed_tokens.weight'] = para_['model.encoder.encoder.embed_tokens.weight']
            else:
                para_[key] = parameters[key]
        parameters = para_
        
    model.load_state_dict(parameters)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    if model_path != './checkpoint/mt5_.pkl':
        optimizer.load_state_dict(state['optimizer'])
    del state
    localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(localtime)
    early_stopping = ttool.EarlyStopping('./checkpoint/paras', 3, True, code_version=ex_name)
    OUT = open('./out_{}.txt'.format(ex_name), mode='w', encoding='utf-8')
    ERR = open('./err_{}.txt'.format(ex_name), mode='w', encoding='utf-8')
    sys.stdout = OUT
    sys.stderr = ERR
    ttool.fit(train_loader, valid_loader, model, tokenizer, optimizer, early_stopping, epoch, device, gradient_accu, True, valid_num_when_training=2000, log_interval=20, evaluate_per_steps=1000)
