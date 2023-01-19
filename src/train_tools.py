import pickle
import random
import re
import sys
import time

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import *
try:
    from . import data_preprocess
    from .data_preprocess import get_article_idx, get_accusation_idx, get_penalty_num, get_all_articles
except ImportError:
    import data_preprocess
    from data_preprocess import get_article_idx, get_accusation_idx, get_penalty_num, get_all_articles

tasks = ['accusation', 'articles', 'imprisonment', 'status']
blank = {'accusation': [0] * 23, 'articles': [0] * 132, 'imprisonment': [1], 'status': [0] * 16}
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, output_path, patience=7, verbose=False, delta=0, code_version=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # self.val_score = 0
        self.delta = delta
        self.output_path = output_path
        self.code_version = code_version

    def __call__(self, val_score, model, optimizer, epoch):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, optimizer, epoch)
            return True
        elif val_score >= self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, optimizer, epoch)
            self.counter = 0
            return True

    def save_checkpoint(self, val_score, model, optimizer, epoch):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            logger.info(f'Validation score decreased ({self.val_loss_min:.6f} --> {val_score:.6f}).  Saving model ...')
        with open('./debug/restarted_task_{}.flag'.format(self.code_version), mode='w', encoding='utf-8') as out1:
            out1.write(self.output_path)
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, self.output_path + self.code_version)
        self.val_loss_min = val_score


def metrics_(x, y):
    print(x.shape, y.shape)
    accuracy_metric = accuracy_score(y, x)
    macro_recall = recall_score(y, x, average='macro')
    macro_precision = precision_score(y, x, average='macro')
    macro_f1 = f1_score(y, x, average='macro')
    return {'acc': accuracy_metric, 'recall': macro_recall, 'precision': macro_precision, 'f1': macro_f1}


def fit(train_dataloader, valid_dataloader, model, tokenizer, optimizer, early_stopping, epochs,
        device, gradient_accelerator, evaluate_per_epoch, log_interval, evaluate_per_steps, valid_num_when_training=1000, max_len=512,
        eos_token_id=None, ignore_signals=None):
    # model.to(device)
    log_interval = log_interval * gradient_accelerator
    data_len = len(train_dataloader)
    last_gradient_accelerator = data_len % gradient_accelerator
    last_gradient_idx = data_len // gradient_accelerator * gradient_accelerator
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        losses = []
        optimizer.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_dataloader, desc='训练中')):
            question = data['question'][0]
            contexts = [question + p[0] for p in data['passages']]
            if len(contexts) < 4:
                for i in range(4 - len(contexts)):
                    contexts.append('')
            target = data['target'][0]
            inputs = tokenizer(contexts, padding='max_length', max_length=512, truncation=True, return_tensors='pt').to(device)
            outputs = tokenizer(target, padding='max_length', max_length=128, truncation=True, return_tensors='pt').to(device)
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
            # print(inputs.shape)

            # print('????', model.is_cuda, inputs.is_cuda, outputs.is_cuda)
            loss = model(**inputs, labels=outputs['input_ids']).loss
            loss = torch.mean(loss)
            total_loss += loss.item()
            losses.append(loss.item())

            if batch_idx >= last_gradient_idx:
                loss /= last_gradient_accelerator
            else:
                loss /= gradient_accelerator

            loss.backward()

            if (batch_idx + 1) % gradient_accelerator == 0 or batch_idx == data_len - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % evaluate_per_steps == 0:
                    train_loss = np.mean(losses)
                    logger.info('Step {}. Train set: Average Loss: {:.6f}'.format(step, train_loss))
                    losses = []
                    # val_loss = test_epoch(dev_dataloader, model, device)
                    # message = 'Step {}. Validation set: Average loss: {:.6f}'.format(step, val_loss)
                    val_metrics = predict(valid_dataloader, model, tokenizer, device, valid_num_when_training=valid_num_when_training, predict_path='./debug/predictions.txt')
                    # del something ---
                    message = 'Step {}. Validation set: '.format(step)
                    for metric in val_metrics:
                        message += '{}: {:.6f}\t'.format(metric, val_metrics[metric])
                    val_score = sum([val_metrics[k] if 'f1' in k else 0 for k in val_metrics])
                    logger.info("val_score is : {}".format(val_score))
                    # flag = early_stopping(val_loss, model, optimizer, step)
                    flag = early_stopping(-val_score, model, optimizer, step)
                    if flag:
                        message += ' *'
                    logger.info(message)
                    if early_stopping.early_stop:
                        break
                    model.train()

            if (batch_idx + 1) % log_interval == 0:
                logger.info('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                                .format(batch_idx, data_len, 100. * batch_idx / data_len, np.mean(losses)))
                losses = []

        if evaluate_per_epoch:
            total_loss /= data_len
            logger.info('Epoch: {}/{}. Train set: Average loss: {:.6f}'.format(epoch + 1, epochs, total_loss))
            val_metrics = predict(valid_dataloader, model, tokenizer, device, predict_path='./debug/predictions.txt')
         
            # del something ---

            message = 'Epoch: {}/{}. Validation set: '.format(epoch + 1, epochs)
            for metric in val_metrics:
                message += '{}: {:.6f}\t'.format(metric, val_metrics[metric])
            logger.info(message)
            val_score = sum([val_metrics[k] if 'f1' in k else 0 for k in val_metrics])
            early_stopping(-val_score, model, optimizer, step)

        if early_stopping.early_stop:
            logger.info('Early stopping!')
            break


def predict(dataloader, model, tokenizer, device, valid_num_when_training=None, predict_path=None, eos_token_id=None, ignore_signals=[]):
    # model.to(device)
    model.eval()
    with torch.no_grad():
        res = []
        targets = []
        cnt = 0
        for data in tqdm(dataloader, desc='预测中'):
            if valid_num_when_training and cnt > valid_num_when_training:
                break
            cnt += 1
            question = data['question'][0]
            contexts = [question + p[0] for p in data['passages']]
            if len(contexts) < 4:
                for i in range(4 - len(contexts)):
                    contexts.append('')
            target = data['target'][0]
            inputs = tokenizer(contexts, padding='max_length', max_length=512, truncation=True, return_tensors='pt').to(device)
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)

            pred = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
            res.extend(get_labels(pred, tokenizer, predict_path, ignore_signals=ignore_signals))

            target = match_content(target)
            # print(target)
            targets.append(target)

    # print(targets)
    pickle.dump([res, targets], open('./test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def prepare_inputs(question, data, tokenizer, device):
    contexts = [question + p[0] for p in data['passages']]
    for i in range(4 - len(contexts)):
        contexts.append('')

    inputs = tokenizer(contexts, padding='max_length', max_length=512, truncation=True,
                       return_tensors='pt').to(device)
    inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
    inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
    return inputs


def strict_predict_case2(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    model.eval()
    with torch.no_grad():
        res = []
        targets = []
        sub_case_cnt = 0
        name_format = '\[被告[A-Z][A-Z]?\]'
        relation_format = r'\[被告[A-Z][A-Z]?\][帮助胁从教唆]{2}\[被告[A-Z][A-Z]?\]'
        for data in tqdm(dataloader, desc='预测中'):
            question: str = data['question'][0]
            target: str = data['target'][0]

            # NEW CODE
            if sub_case_cnt:
                if not target.startswith('此被告刑期'):
                    targets.append(match_content(target))
                sub_case_cnt -= 1
            elif target.startswith('他们之间的关系是'):
                targets.append(match_content(target))
                names = re.findall(name_format, question)
                sub_case_cnt = len(names) * 2

                # PROMPT 1
                inputs = prepare_inputs(question, data, tokenizer, device)
                pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                res.extend(pred_dict_list)
                pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                pred_relations_str = pred_text.replace('他们之间的关系是', '').replace('。', '')
                relations = re.findall(relation_format, pred_text)
                for name in names:
                    name_relations_list = []
                    for relation in relations:
                        if name in relation:
                            name_relations_list.append(relation)
                    name_relation_str = '，'.join(name_relations_list)

                    # PROMPT 3
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'. \
                            format(name, name_relation_str)
                    else:
                        question = '着重考虑被告{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                            name)
                    question = '问题： ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)

            else:
                gg

    # print(targets)
    pickle.dump([res, targets], open('./strict_test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def strict_predict_case1(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    model.eval()
    with torch.no_grad():
        res = []
        targets = []
        sub_case_cnt = 0
        name_format = '\[被告[A-Z][A-Z]?\]'
        relation_format = r'\[被告[A-Z][A-Z]?\][帮助胁从教唆]{2}\[被告[A-Z][A-Z]?\]'
        for data in tqdm(dataloader, desc='预测中'):
            question: str = data['question'][0]
            target: str = data['target'][0]

            # NEW CODE
            if sub_case_cnt:
                sub_case_cnt -= 1
                if sub_case_cnt == 0:
                    continue
                targets.append(match_content(target))

            elif target.startswith('此被告存在的量刑情节'):
                targets.append(match_content(target))
                names = re.findall(name_format, question)
                sub_case_cnt = 2

                # PROMPT 2
                inputs = prepare_inputs(question, data, tokenizer, device)
                pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                res.extend(pred_dict_list)
                pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                pred_states_str = pred_text.replace('此被告存在的量刑情节为', '').replace('。', '')
                for name in names:
                    # PROMPT 3 START
                    question = '着重考虑被告{}具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(
                            name, pred_states_str)
                    question = '问题： ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    # PROMPT 3 END
            else:
                gg

    # print(targets)
    pickle.dump([res, targets], open('./strict_test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def strict_predict_case5(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    model.eval()
    with torch.no_grad():
        res = []
        targets = []
        cnt = 0
        for data in tqdm(dataloader, desc='预测中'):
            cnt += 1
            question = data['question'][0]
            contexts = [question + p[0] for p in data['passages']]
            if len(contexts) < 4:
                for i in range(4 - len(contexts)):
                    contexts.append('')
            target = data['target'][0]

            if target.startswith('此被告刑期为'):
                continue

            inputs = tokenizer(contexts, padding='max_length', max_length=512, truncation=True, return_tensors='pt').to(
                device)
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)

            pred = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
            res.extend(get_labels(pred, tokenizer, predict_path, ignore_signals=ignore_signals))

            target = match_content(target)
            # print(target)
            targets.append(target)

    # print(targets)
    pickle.dump([res, targets], open('./test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def strict_predict_case6(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    with torch.no_grad():
        res = []
        targets = []
        sub_case_cnt = 0
        name_format = '\[被告[A-Z][A-Z]?\]'
        relation_format = r'\[被告[A-Z][A-Z]?\][帮助胁从教唆]{2}\[被告[A-Z][A-Z]?\]'
        cnt = 0
        for data in tqdm(dataloader, desc='预测中'):
            question: str = data['question'][0]
            target: str = data['target'][0]
            if sub_case_cnt:
                targets.append(match_content(target))
                sub_case_cnt -= 1
            elif target.startswith('他们之间的关系是'):
                print(question)
                targets.append(match_content(target))
                names = re.findall(name_format, question)
                sub_case_cnt = len(names) * 2

                # PROMPT 1
                inputs = prepare_inputs(question, data, tokenizer, device)
                pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                res.extend(pred_dict_list)
                pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                pred_relations_str = pred_text.replace('他们之间的关系是', '').replace('。', '')
                relations = re.findall(relation_format, pred_text)
                for name in names:
                    name_relations_list = []
                    for relation in relations:
                        if name in relation:
                            name_relations_list.append(relation)
                    name_relation_str = '，'.join(name_relations_list)
                    # PROMPT 2
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为'.format(name) + name_relation_str + '。那么该被告存在什么量刑情节？'
                    else:
                        question = '着重考虑被告{}。该被告存在什么量刑情节？'.format(name)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                    states = pred_text.replace('此被告存在的量刑情节为', '').replace('。', '')

                    # PROMPT 4
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期应该是？对应的罪名是？相关的法条为？'.\
                            format(name, name_relation_str, states)
                    else:
                        question = '着重考虑被告{}，具有的量刑情节为{}。其刑期应该是？对应的罪名是？相关的法条为？'.format(name, states)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)

            else:
                gg

    if len(res) != len(targets):
        gg

    # print(targets)
    pickle.dump([res, targets], open('./strict_test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def strict_predict_case7(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    with torch.no_grad():
        res = []
        targets = []
        sub_case_cnt = 0
        name_format = '\[被告[A-Z][A-Z]?\]'
        relation_format = r'\[被告[A-Z][A-Z]?\][帮助胁从教唆]{2}\[被告[A-Z][A-Z]?\]'
        cnt = 0
        for data in tqdm(dataloader, desc='预测中'):
            question: str = data['question'][0]
            target: str = data['target'][0]
            if sub_case_cnt:
                targets.append(match_content(target))
                sub_case_cnt -= 1
            elif target.startswith('他们之间的关系是'):
                print(question)
                targets.append(match_content(target))
                names = re.findall(name_format, question)
                sub_case_cnt = len(names) * 4

                # PROMPT 1
                inputs = prepare_inputs(question, data, tokenizer, device)
                pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                res.extend(pred_dict_list)
                pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                pred_relations_str = pred_text.replace('他们之间的关系是', '').replace('。', '')
                relations = re.findall(relation_format, pred_text)
                for name in names:
                    name_relations_list = []
                    for relation in relations:
                        if name in relation:
                            name_relations_list.append(relation)
                    name_relation_str = '，'.join(name_relations_list)
                    # PROMPT 2
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为'.format(name) + name_relation_str + '。那么该被告存在什么量刑情节？'
                    else:
                        question = '着重考虑被告{}。该被告存在什么量刑情节？'.format(name)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                    states = pred_text.replace('此被告存在的量刑情节为', '').replace('。', '')

                    # PROMPT 3.1 START
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条是？'.format(
                                name, pred_relations_str, states)
                    else:
                        question = '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条是？'.format(name, states)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    # PROMPT 3.1 END

                    # PROMPT 3.2 START
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告的罪名是？'.format(
                                name, pred_relations_str, states)
                    else:
                        question = '着重考虑被告{}，具有的量刑情节为{}。该被告的罪名是？'.format(
                                name, states)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    # PROMPT 3.2 END

                    # PROMPT 3.3 START
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告刑期是？'.format(
                                name, pred_relations_str, states)
                    else:
                        question = '着重考虑被告{}，具有的量刑情节为{}。该被告刑期是？'.format(
                                name, states)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    # PROMPT 3.3 END

            else:
                gg

    if len(res) != len(targets):
        gg

    # print(targets)
    pickle.dump([res, targets], open('./strict_test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def strict_predict(dataloader, model, tokenizer, device, predict_path=None, eos_token_id=None, mode=None, ignore_signals=[]):
    if mode == 'case1':
        return strict_predict_case1(dataloader, model, tokenizer, device, predict_path, eos_token_id, mode,
                                    ignore_signals)
    if mode == 'case2':
        return strict_predict_case2(dataloader, model, tokenizer, device, predict_path, eos_token_id, mode,
                                    ignore_signals)
    if mode == 'case4':
        return predict(dataloader, model, tokenizer, device)
    if mode == 'case5':
        return strict_predict_case5(dataloader, model, tokenizer, device, predict_path, eos_token_id, mode,
                                    ignore_signals)
    if mode == 'case6':
        return strict_predict_case6(dataloader, model, tokenizer, device, predict_path, eos_token_id, mode,
                                    ignore_signals)
    if mode == 'case7':
        return strict_predict_case7(dataloader, model, tokenizer, device, predict_path, eos_token_id, mode,
                                    ignore_signals)
    print('complete or case3')
    model.eval()
    with torch.no_grad():
        res = []
        targets = []
        sub_case_cnt = 0
        name_format = '\[被告[A-Z][A-Z]?\]'
        relation_format = r'\[被告[A-Z][A-Z]?\][帮助胁从教唆]{2}\[被告[A-Z][A-Z]?\]'
        cnt = 0
        for data in tqdm(dataloader, desc='预测中'):
            question: str = data['question'][0]
            target: str = data['target'][0]
            # cnt += 1
            # if cnt == 100:
            #     break

            # NEW CODE
            if sub_case_cnt:
                if not target.startswith('此被告刑期'):
                    # print(sub_case_cnt)
                    # print(target)
                    targets.append(match_content(target))
                sub_case_cnt -= 1
            elif target.startswith('他们之间的关系是'):
                print(question)
                targets.append(match_content(target))
                names = re.findall(name_format, question)
                if mode == 'case3':
                    sub_case_cnt = len(names) * 2
                else:
                    sub_case_cnt = len(names) * 3

                # PROMPT 1
                inputs = prepare_inputs(question, data, tokenizer, device)
                pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                res.extend(pred_dict_list)
                pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                pred_relations_str = pred_text.replace('他们之间的关系是', '').replace('。', '')
                relations = re.findall(relation_format, pred_text)
                for name in names:
                    name_relations_list = []
                    for relation in relations:
                        if name in relation:
                            name_relations_list.append(relation)
                    name_relation_str = '，'.join(name_relations_list)
                    # PROMPT 2
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为'.format(name) + name_relation_str + '。那么该被告存在什么量刑情节？'
                    else:
                        question = '着重考虑被告{}。该被告存在什么量刑情节？'.format(name)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)
                    pred_text = tokenizer.decode(pred_idx_list[0], skip_special_tokens=True)
                    states = pred_text.replace('此被告存在的量刑情节为', '').replace('。', '')

                    # PROMPT 3
                    if pred_relations_str != '无':
                        question = '已知被告{}涉及到的犯罪关系为{}，具有的量刑情节为{}。那么该被告相关的法条和对应的罪名是？其刑期是？'.\
                            format(name, name_relation_str, states)
                    else:
                        question = '着重考虑被告{}，具有的量刑情节为{}。该被告相关的法条和对应的罪名是？其刑期是？'.format(name, states)
                    question = '问题: ' + question
                    inputs = prepare_inputs(question, data, tokenizer, device)
                    pred_idx_list = model.generate(**inputs, eos_token_id=eos_token_id).detach().cpu().numpy()
                    pred_dict_list = get_labels(pred_idx_list, tokenizer, predict_path, ignore_signals=ignore_signals)
                    res.extend(pred_dict_list)

            else:
                gg

    if len(res) != len(targets):
        gg

    # print(targets)
    pickle.dump([res, targets], open('./strict_test_res.pkl', mode='wb'))
    metrics = get_metrics(res, targets)
    print(metrics)
    return metrics


def test_pkl():
    res, targets = pickle.load(open('../strict_test_res.pkl', mode='rb'))
    metrics = get_metrics(res, targets, '12')
    print(metrics)

    # res, targets = pickle.load(open('../test_res.pkl', mode='rb'))
    # metrics = get_metrics(res, targets, '123')
    # print(metrics)


def test_epoch(dataloader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = {k: data[k].to(device) for k in data}
            loss = model(**data).loss
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss


def match_content(content):
    res = {}
    try:
        if re.search(r'[0-9]{2,3}:[是否]', content) is not None:
            res['articles'] = [0] * 132
            tt = re.findall(r'([0-9]{2,3}):[是]', content)
            for article in tt:
                idx = get_article_idx('第' + article + '条')
                if idx is not None:
                    res['articles'][idx] = 1
        elif re.search(r'<law_[0-9]{1,3}> ?:?[是否]', content) is not None:
            res['articles'] = [0] * 132
            tt = re.findall(r'<law_([0-9]{1,3})> ?:?[是]', content)
            for article in tt:
                idx = get_article_idx('第' + article + '条')
                if idx is not None:
                    res['articles_if'][idx] = 1
        elif re.search(r'<law_[0-9]{1,3}>', content) is not None:
            res['articles'] = [0] * 132
            tt = re.findall(r'<law_([0-9]{1,3})>', content)
            for article in tt:
                idx = get_article_idx('第' + article + '条')
                if idx is not None:
                    res['articles'][idx] = 1
    except KeyError:
        pass
    try:
        if re.search(r'(因此罪名为|此被告罪名为)(.+?罪){1,5}?', content) is not None:
            res['accusation'] = [0] * 23
            txt = re.search(r'(因此罪名为|此被告罪名为)(.+?罪){1,5}?', content).group().replace('因此罪名为', '').replace('此被告罪名为', '').replace(',', '').replace('，', '')
            for accu in re.findall(r'(.{2,10}?罪)', txt):
                idx = get_accusation_idx(accu)
                if idx is not None:
                    res['accusation'][idx] = 1
    except KeyError:
        pass
    try:
        if re.search('刑期为((有期徒刑|死刑|无期徒刑).{0,10})', content) is not None or content == '死刑' or content == '无期徒刑':
            res['imprisonment'] = [1]
            txt = re.search('刑期为((有期徒刑|死刑|无期徒刑).{0,10})', content).group(1)
            idx = get_penalty_num(txt)
            if idx < 0:
                print('我', content)
            if idx is not None:
                res['imprisonment'][0] = idx
    except KeyError:
        pass
    try:
        pass
        # if re.search(r'\[被告([A-Z][A-Z]?)]', content) is not None:
        #     res['names']
        #     for obj in re.finditer(r'\[被告([A-Z][A-Z]?)]', content):
        #         s = obj.group(1)
        #         if len(s) == 1:
        #             res['names'][ord(s) - ord('A')] = 1
        #         else:
        #             try:
        #                 res['names'][(ord(s[0]) - ord('A')) * 26 + ord(s[1]) - ord('A')] = 1
        #             except IndexError:
        #                 pass
    except KeyError:
        pass
    try:
        if re.search(r'[^_0-9][0-9]{1,2}', content) is not None:
            res['status'] = [0] * 16
            for obj in re.finditer(r'[^_0-9]([0-9]{1,2})', content):
                s = int(obj.group(1))
                if 0 < s < 17:
                    res['status'][s - 1] = 1
    except KeyError:
        pass
    return res


def get_labels(preds, tokenizer, predict_path, ignore_signals=[]):
    results = []
    for pred in preds:

        text = tokenizer.decode(pred, skip_special_tokens=True)

        if predict_path is not None:
            with open(predict_path, 'w', encoding='utf-8') as f:
                f.write(text + '\n')

        with open('./debug_print.txt', 'w', encoding='utf-8') as O:
            O.write('{} : {}'.format(time.time(), text) + '\n')

        res = match_content(text)
        results.append(res)
    return results


def get_metrics(preds, targets, elements='123'):
    dict_task_list_pred = {task: [] for task in tasks}
    dict_task_list_target = {task: [] for task in tasks}
    for p, t in zip(preds, targets):
        for task in tasks:
            if task in t and task not in p:
                p[task] = blank[task]
            if task not in t:
                continue
            if task == 'imprisonment':
                x = F.one_hot(torch.tensor(p['imprisonment']), 11)
                y = F.one_hot(torch.tensor(t['imprisonment']), 11)
            else:
                x = torch.tensor([p[task]])
                y = torch.tensor([t[task]])
            dict_task_list_pred[task].append(x)
            dict_task_list_target[task].append(y)

    return cat_list_then_cal_metrics(dict_task_list_pred, dict_task_list_target)


def cat_list_then_cal_metrics(label, truth):
    metrics_dict = {}
    for task in tasks:
        x_list = label[task]
        y_list = truth[task]
        if y_list == []:
            continue
        metrics_dict.update(
            {'{}-'.format(task) + key: item for key, item in
             metrics_(torch.cat(x_list, dim=0).numpy(), torch.cat(y_list, dim=0).numpy()).items()}
        )

    return metrics_dict


if __name__ == '__main__':
    test_pkl()