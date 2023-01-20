import json
import math
import re
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from tqdm import *
import traceback
import pickle
import random
import numpy as np
from collections import Counter
import pickle as pk
import data_preprocess_for_transform as data_preprocess

# 模型的输入输出都可以从如下对象中取出
class CriminalInfo(BaseModel):
    term: str
    accu_law_states: List[Tuple[str, List[int], List[int]]]


class Case(BaseModel):
    fact_desc: str
    interpretation: Optional[str]
    laws: List[str]
    criminals_info: Dict[str, CriminalInfo]
    relations: List[Tuple[str, str, str, str]]


def transform(source_file, target_file, func):
    R = open(source_file, mode='r', encoding='utf-8')
    W = open(target_file, mode='w', encoding='utf-8')
    inputs, outputs, facts = [], [], []
    for line in R:
        input_, output_, fact_ = func(line)
        inputs += input_
        outputs += output_
        facts += fact_

    for idx in range(len(inputs)):
        js = {'id': idx,
              'question': inputs[idx],
              'target': outputs[idx],
              'answers': [outputs[idx]],
              'ctxs': facts[idx]}
        W.write(json.dumps(js).encode('utf-8').decode('unicode_escape') + '\n')
    R.close()
    W.close()


if __name__ == '__main__':
    transform('./data_train.json', './train_data_complete.jsonl', data_preprocess.prepare_complete_data)
    transform('./data_valid.json', './valid_data_complete.jsonl', data_preprocess.prepare_complete_data)
    transform('./data_test.json', './test_data_complete.jsonl', data_preprocess.prepare_complete_data)
