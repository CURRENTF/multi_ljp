# -*- coding:utf-8 -*-
import re
import json
import sys

import torch
import random
from tqdm import tqdm


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
         '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十':
        hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


# 这里处理数据，所以每一条数据生成多个训练样本
# N + 1 = 被告个数 + 1
def preprocess_data(tokenizer, data_path, elements, data_size=None, shuffle=False,
                    supervise=False, use_article_content=False):
    with open(data_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        if shuffle:
            random.seed(1024)
            random.shuffle(all_lines)
        if data_size is not None:
            all_lines = all_lines[:data_size]

    input_datas, targets = [], []
    if supervise:
        for line in all_lines:
            input_data, target = prepare_supervised_data(line, elements)
            input_datas += input_data
            targets += target
    else:
        for line in all_lines:
            fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(
                line)
            input_data, target = prepare_unsupervised_data(
                tokenizer, line, elements, use_article_content, 0)
            input_datas += input_data
            targets += target
            for i in range(len(criminals_info)):
                input_data, target = prepare_unsupervised_data(
                    tokenizer, line, elements, use_article_content, i + 1)
                input_datas += input_data
                targets += target

    print('test inputs: facts:\n {}\n\ntargets: {}'.format('\n'.join(input_datas[:5]), '\n'.join(targets[:5])),
          file=sys.stderr)

    return input_datas, targets


def generate_target(lines):
    results = []
    
    for line in tqdm(lines):
        data = loads_str(line)
        res = {"names": [0] * 50, "articles": [0] * 132}
        relevant_articles = data['laws']
        for article in relevant_articles:
            article = '第' + str(article) + '条'
            res['articles'][get_article_idx(article)] = 1
        for idx in range(len(data['criminals_info'])):
            res['names'][idx] = 1
        results.append(res)
        res = {'articles_if': [0] * 132}
        relevant_articles = data['laws']
        for article in relevant_articles:
            article = '第' + str(article) + '条'
            res['articles_if'][get_article_idx(article)] = 1
        results.append(res)
        for name in data['criminals_info']:
            res = {"accusation": [0] * 23, "status": [0] * 16, "imprisonment": []}
            accusation = [tup[0] for tup in data['criminals_info'][name]['accu_law_states']]
            for acc in accusation:
                acc = acc.replace('[', '').replace(']', '')
                tp = get_accusation_idx(acc.strip())
                if tp is not None:
                    res['accusation'][tp] = 1
                else:
                    # print('出错的罪名 {}'.format(acc), file=sys.stderr)
                    pass
            status = []
            for tup in data['criminals_info'][name]['accu_law_states']:
                status += tup[2]
            status = set(status)
            for i in status:
                if i:
                    res['status'][i - 1] = 1
            term_of_imprisonment = data['criminals_info'][name]['term']
            num = get_penalty_num(term_of_imprisonment)
            res['imprisonment'] = [num]
            results.append(res)

    # print(len(results))
    return results

def loads_str(data_str):
    cnt = 0
    this_w = []
    while True:
        cnt += 1
        try:
            result = json.loads(data_str, strict=False)
            # print("最终json加载结果：{}".format(result))
            return result
        except Exception as e:
            # print("异常信息e：{}".format(e))
            error_index = re.findall(r"char (\d+)\)", str(e))
            if error_index:
                error_str = data_str[int(error_index[0])]
                this_w.append(
                    data_str[int(error_index[0]) - 3:int(error_index[0]) + 3])
                print('带来错误的字符上下文:{}'.format(this_w), file=sys.stderr)
                data_str = data_str.replace(error_str, "")
                # print("替换异常字符串{} 后的文本内容{}".format(error_str, data_str))
                # 该处将处理结果继续递归处理
                # return loads_str(data_str)
            else:
                break
        if cnt >= 100:
            print(this_w)
            break


def get_elements_from_line(line):
    data = loads_str(line)
    fact = data['fact_desc'].replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace(
        '<SIGN_LINE>', '')
    view = re.sub(r'本院认为.', '', data.get('interpretation', '').strip())
    view = view.replace('\n', '').replace('\t', '').replace('\r', '') \
        .replace('&times；', 'x').replace('&divide；', '÷').replace('&amp；', '&').replace('&yen；', '￥')
    view = re.sub(
        r'&ldquo；|&rdquo；|&middot；|&quot；|&plusmn；|&hellip；|&lsquo；|&rsquo；|&permil；|&mdash；|\|', '', view)
    # articles = ','.join(['第' + str(num) + '条' for num in sorted(data['laws'])])
    articles = [str(i) for i in sorted(data['laws'])]
    criminals_info = data['criminals_info']
    article_contents = articles
    relations = data['relations']
    return fact, articles, criminals_info, view, article_contents, relations


def prepare_supervised_data(line, elements):
    laws_many = [64, 68, 27, 25, 347, 61, 65, 67, 356, 348, 69, 293, 72, 73, 266, 53, 26, 303, 48, 57, 52, 264, 234, 42,
                 44, 77, 2, 3, 5, 45, 47, 50, 292, 23, 224, 55, 56, 70, 62, 1, 76, 232, 238, 63, 4, 59, 357, 71, 277,
                 17, 36, 8, 198, 349, 37, 14, 19, 6, 383]
    laws_many = [str(i) for i in laws_many]
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(
        line)
    input_datas, targets = [], []

    # input_datas.append('预测涉案被告代号及关系' + fact)
    # t_lis = []
    # for relation in relations:
    #     t_lis.append(
    #         '[被告' + relation[1] + ']' + relation[2] + '了' + '[被告' + relation[3] + ']')
    # targets.append('代号为{}。他们之间的关系是{}'.format(','.join(list(criminals_info.keys())),
    #                                                   (','.join(t_lis)) if (','.join(t_lis)) else '无'))

    input_datas.append('预测涉案被告代号，他们的关系和本案所有相关的法条' + fact)
    t_lis = []
    for relation in relations:
        t_lis.append(
            '[被告' + relation[1] + ']' + relation[2] + '了' + '[被告' + relation[3] + ']')
    t = ''.join(
        ['{}'.format('<law_{}>'.format(key)) if key in articles else '' for key in
         sorted(get_all_articles().keys())])
    targets.append('代号为{}。他们之间的关系是{}。本案涉及的法条为{}'.format(','.join(list(criminals_info.keys())),
                                                        (','.join(t_lis)) if (','.join(t_lis)) else '无', t))

    input_datas.append('以是否的形式预测本案所有相关法条：' + fact)

    t = ''.join(
        ['{}是'.format('<law_{}>'.format(key)) if key in articles else '{}否'.format('<law_{}>'.format(key)) for key in
         laws_many])
    # articles = ''.join(
    #     ['{}'.format('<law_{}>'.format(key)) if key in articles else '' for key in
    #      sorted(get_all_articles().keys())])
    targets.append('{}'.format(t))

    for name in criminals_info:
        input_datas.append('预测{}罪名涉及的法条和对应的罪名，相关的情节与其刑期：'.format(name) + fact)
        targets.append(
            '此被告罪名涉及的法条为{}，因此罪名为{}。情节为{}。刑期为{}。'.format(''.join([''.join(['<law_{}>'.format(str(___)) for ___ in l[1]])
                                                                 for l in
                                                                 criminals_info[name]['accu_law_states']]),
                                                        ','.join([l[0] for l in
                                                                  criminals_info[name]['accu_law_states']]),
                                                        '；'.join([','.join([str(___) for ___ in l[2]])
                                                                  for l in
                                                                  criminals_info[name]['accu_law_states']]),
                                                        re.split(r'[，,]?(缓刑|拘役)', criminals_info[name]['term'])[
                                                            0]))

    return input_datas, targets


def get_related_relations(relations, person):
    t_dic = {}
    for relation in relations:
        if relation[1] == person:
            if relation[2] in t_dic:
                t_dic[relation[2]].append('[被告{}]'.format(relation[3]))
            else:
                t_dic[relation[2]] = ['[被告{}]'.format(relation[3])]

    res = ''
    for key in t_dic:
        res += '[被告{}]{}了{}。'.format(person, key, '，'.join(t_dic[key]))
    return res


def prepare_unsupervised_data(tokenizer, line, elements, use_article_content, mode):
    facts, targets = [], []
    fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(
        line)
    special_token_maps = {i: ' <extra_id_{}>'.format(i) for i in range(8)}
    add_text = ''
    target = ''
    token_counter = 0

    if mode == 0:
        add_text += '综上，涉案被告共有' + special_token_maps[token_counter] + '几位，'
        target += special_token_maps[token_counter] + \
            ','.join(list(criminals_info.keys()))
        token_counter += 1

        add_text += '他们之间的关系是' + special_token_maps[token_counter] + '。'
        target += special_token_maps[token_counter]
        t_lis = []
        for relation in relations:
            t_lis.append(
                '[被告' + relation[1] + ']' + relation[2] + '了' + '[被告' + relation[3] + ']')
        target += (','.join(t_lis))[:50]
        token_counter += 1

        add_text += '综上，依据《中华人民共和国刑法》' + special_token_maps[token_counter]
        target += special_token_maps[token_counter] + \
            articles.replace('第', '').replace('条', '')
        token_counter += 1

    elif mode >= 1:
        chr_name = chr(mode - 1 + ord('A')) if mode <= 26 else chr((mode - 1) // 26 + ord('A')) + chr(
            (mode - 1) % 26 + ord('A'))
        criminal_name_here = '[被告{}]'.format(chr_name)

        add_text += '综上，涉案被告共有' + ','.join(list(criminals_info.keys())) + '几位，在本次预测中，我们着重考虑{}。'.format(
            criminal_name_here)

        add_text += '他们之间的关系是' + \
            get_related_relations(relations, chr_name) + '。'

        # if 'accusation' in elements:
        add_text += '被告人行为构成' + special_token_maps[token_counter] + '，'
        target += special_token_maps[token_counter] + ','.join(
            [l[0] for l in criminals_info[criminal_name_here]['accu_law_states']])
        token_counter += 1

        add_text += '考虑到被告有如下情节：' + special_token_maps[token_counter] + '。'
        target += special_token_maps[token_counter] + '；'.join(
            [','.join([str(___) for ___ in l[2]]) for l in criminals_info[criminal_name_here]['accu_law_states']])
        token_counter += 1

        # if 'penalty' in elements:
        add_text += '应判处' + special_token_maps[token_counter]
        target += special_token_maps[token_counter] + \
            criminals_info[criminal_name_here]['term']
        token_counter += 1

    else:
        print('mode设置有错', file=sys.stderr)

    # cut input length
    input_data = add_text + fact
    input_ids = tokenizer.encode(input_data, truncation=True, max_length=556)
    if len(input_ids) == 556:
        # print('现在token counter的大小是:', token_counter)
        # print(special_token_maps)
        # print(input_ids)
        # print(tokenizer.encode(special_token_maps[0]+special_token_maps[1]+special_token_maps[2]))
        # print('现在add_text本身的长度是:', len(add_text))
        # print('内容是:', add_text)
        idx = 250100 - token_counter
        fact = tokenizer.decode(input_ids[input_ids.index(idx) + 2:-1]) + '…'
    facts.append(fact + add_text)
    targets.append(target.strip() + special_token_maps[token_counter])

    return facts, targets


def generate_decoder_input(data_path, article_content_dict, tokenizer, test_path):
    decoder_inputs = []
    with open(test_path, 'r', encoding='utf-8') as f_2:
        for target in f_2.readlines():
            fact, articles, criminals_info, view, article_contents, relations = get_elements_from_line(
                target)
            article_contents = article_contents.split(',')
            for i, article in enumerate(article_contents):
                article_contents[i] = tokenizer.decode(
                    article_content_dict[article])
            article_contents = ';'.join(article_contents)
            decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>'
            # decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>' + accusations + ' <extra_id_2>'
            # decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>' + article_contents + ' <extra_id_2>' + \
            #                 accusations + ' <extra_id_3>'
            decoder_inputs.append(decoder_input)

    return decoder_inputs


def get_article_idx(article):
    p = re.search(r'第?([0-9]{1,3})条?', article)
    if p:
        article = p.group(1)
    article2idx = get_all_articles()
    if article in article2idx:
        return article2idx[article]
    else:
        # print('出错的法律{}'.format(article), file=sys.stderr)
        return 0


def get_all_articles():
    return {'67': 0, '72': 1, '293': 2, '25': 3, '26': 4, '65': 5, '73': 6, '264': 7, '52': 8, '53': 9, '27': 10,
            '64': 11, '347': 12, '348': 13, '234': 14, '232': 15, '69': 16, '68': 17, '292': 18, '77': 19, '303': 20,
            '266': 21, '23': 22, '4': 23, '356': 24, '56': 25, '71': 26, '86': 27, '55': 28, '61': 29, '70': 30,
            '238': 31, '59': 32, '57': 33, '2': 34, '1': 35, '47': 36, '45': 37, '48': 38, '38': 39, '41': 40, '76': 41,
            '62': 42, '8': 43, '44': 44, '42': 45, '349': 46, '3': 47, '6': 48, '75': 49, '36': 50, '312': 51,
            '224': 52, '63': 53, '19': 54, '390': 55, '383': 56, '7': 57, '382': 58, '389': 59, '5': 60, '277': 61,
            '11': 62, '357': 63, '12': 64, '17': 65, '9': 66, '22': 67, '198': 68, '50': 69, '133': 70, '275': 71,
            '29': 72, '74': 73, '51': 74, '196': 75, '385': 76, '397': 77, '386': 78, '155': 79, '37': 80, '14': 81,
            '263': 82, '274': 83, '128': 84, '279': 85, '345': 86, '365': 87, '20': 88, '24': 89, '93': 90, '15': 91,
            '302': 92, '54': 93, '58': 94, '97': 95, '157': 96, '31': 97, '81': 98, '203': 99, '13': 100, '125': 101,
            '344': 102, '354': 103, '384': 104, '18': 105, '310': 106, '87': 107, '176': 108, '172': 109, '241': 110,
            '334': 111, '280': 112, '290': 113, '30': 114, '39': 115, '40': 116, '393': 117, '60': 118, '43': 119,
            '34': 120, '16': 121, '201': 122, '388': 123, '193': 124, '88': 125, '226': 126, '28': 127, '307': 128,
            '83': 129, '21': 130, '269': 131}


def get_accusation_mapper():
    return {'诈骗罪': 0, '合同诈骗罪': 1, '保险诈骗罪': 2, '贷款诈骗罪': 3, '招摇撞骗罪': 4, '盗窃罪': 5, '盗伐林木罪': 6, '故意伤害罪': 7,
            '寻衅滋事罪': 8, '聚众斗殴罪': 9, '故意杀人罪': 10, '赌博罪': 11, '开设赌场罪': 12, '受贿罪': 13, '行贿罪': 14, '贪污罪': 15,
            '妨害公务罪': 16, '非法拘禁罪': 17, '敲诈勒索罪': 18, '贩卖毒品罪': 19, '贩卖、运输毒品罪': 19, '运输毒品罪': 19, '制造毒品罪': 19,
            '贩卖、制造毒品罪': 19, '走私、贩卖毒品罪': 19, '走私、贩卖、运输毒品罪': 19, '走私、运输毒品罪': 19, '走私毒品罪': 19, '贩卖、运输、制造毒品罪': 19,
            '走私、贩卖、运输、制造毒品罪': 19, '制造、贩卖毒品罪': 19, '非法持有毒品罪': 20, '窝藏毒品罪': 21, '窝藏、转移毒品罪': 21, '转移毒品罪': 21,
            '包庇毒品犯罪分子罪': 22}


def get_accusation_idx(accu):
    accusation2idx = get_accusation_mapper()
    if accu in accusation2idx:
        return accusation2idx[accu]
    else:
        return None


def get_class_of_month(num):
    if num > 20 * 12:
        num = 0
    elif num > 10 * 12:
        num = 1
    elif num > 7 * 12:
        num = 2
    elif num > 5 * 12:
        num = 3
    elif num > 3 * 12:
        num = 4
    elif num > 2 * 12:
        num = 5
    elif num > 1 * 12:
        num = 6
    elif num > 9:
        num = 7
    elif num > 6:
        num = 8
    elif num > 0:
        num = 9
    else:
        num = 10
    return num


def get_penalty_num(penalty):
    if '死刑' in penalty:
        return 0
    elif '无期徒刑' in penalty:
        return 0
    else:
        p = re.search(
            r'(有期徒刑|拘役|管制)(([一二三四五六七八九十零]{1,3})个?年)?(([一二三四五六七八九十零]{1,3})个?月)?', penalty)
        if not p:
            # print('出错的刑期 {}'.format(penalty), file=sys.stderr)
            return 0
        else:
            if p.group(3):
                year = int(hanzi_to_num(p.group(3)))
            else:
                year = 0
            if p.group(5):
                month = int(hanzi_to_num(p.group(5)))
            else:
                month = 0
            num = year * 12 + month
            return get_class_of_month(num)


if __name__ == "__main__":
    art2id = get_all_articles()
    with open('../neurjudge/data/article2id.json', mode='w', encoding='utf-8') as O:
        js: str = json.dumps(art2id)
        O.write(js)

    accu2id = get_accusation_mapper()
    with open('../neurjudge/data/charge2id.json', mode='w', encoding='utf-8') as O:
        js: str = json.dumps(accu2id)
        O.write(js.encode('utf-8').decode('unicode_escape'))

    with open('../neurjudge/data/time2id.json', mode='r', encoding='utf-8') as I:
        t: dict = json.load(I)
        time2id = {num: get_class_of_month(int(num)) for num in t}

    with open('../neurjudge/data/time2id.json', mode='w', encoding='utf-8') as O:
        js: str = json.dumps(time2id)
        O.write(js)
