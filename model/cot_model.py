from collections import defaultdict
import os
import random
from tqdm import tqdm

import torch.nn as nn

from data.prefix import *
from model.utils import label2desc, completions_with_gpt_backoff, format_en_cot_entity, format_cn_cot_entity


class COTEnNERModel(nn.Module):
    def __init__(self, dataset, k, tgt_vocab, llm_model_name="text-davinci-003", random_ins=1,
                 encoding_type='bio', temperature=0.5):
        super(COTEnNERModel, self).__init__()
        self.dataset = dataset
        self.k = k
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
        self.encoding_type = encoding_type
        self.temperature = temperature
        if random_ins:
            self.rng = random.Random()
        else:
            self.rng = random.Random(123)
        # all labels
        label_list = list(tgt_vocab.word2idx.keys())
        self.label_list = [label2desc[l.upper()] for l in label_list]

        # instructions
        self.prefix_all = "Please identify " + ', '.join([label for label in self.label_list]) + \
                          " Entity from the given sentences.\n"
        if 'ace' in dataset:
            self.prefix_all += " Notice that there may exist nested entities."
        self.prefix_all += '\n'
        # cot examples sets
        if dataset == 'BC5CDR-disease-IOB':
            prefix_cot = prefix_cot_diseases
        elif dataset == 'BC5CDR-chem-IOB':
            prefix_cot = prefix_cot_chems
        elif dataset == 'BC2GM-IOB':
            prefix_cot = prefix_cot_genes
        elif dataset == 'ontonotes5':
            prefix_cot = prefix_cot_ontonotes5
        elif dataset == 'conll03':
            prefix_cot = get_conll_trigger('../../data/TriggerNER/CONLL/trigger_20.txt')
        elif dataset == 'ace04':
            prefix_cot = prefix_cot_ace04
        elif dataset == 'ace05':
            prefix_cot = prefix_cot_ace05
        else:
            raise ValueError(f'Invalid dataset: {dataset}')
        self.prefix_cot = prefix_cot

        # output files
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if not os.path.exists('outputs/invalid'):
            os.mkdir('outputs/invalid')
        if not os.path.exists('outputs/wrong'):
            os.mkdir('outputs/wrong')
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_cot_{str(self.k)}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_cot_{str(self.k)}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_cot_{str(self.k)}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_spans):
        demonstrations = self.rng.sample(self.prefix_cot, self.k)

        invalid_sents = 0
        format_negatives = 0
        span_negatives = 0
        false_positive = 0
        true_positive = 0
        negative = 0
        pred = []
        for idx, _raw_words in enumerate(tqdm(raw_words)):
            sent = 'Sentence: ' + ' '.join(_raw_words) + '\n'
            ner_prefix = self.prefix_all + ''.join(demonstrations)
            prompt = ner_prefix + sent + 'Entity:'
            response = completions_with_gpt_backoff(model=self.llm_model_name, prompt=prompt,
                                                    temperature=self.temperature, max_tokens=512)
            pred_text = response['choices'][0]['text']
            pred_entities, sent_invalid, format_negative, span_negative = format_en_cot_entity(pred_text, _raw_words,
                                                                                               self.label_list)
            self.f_all.write(
                prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx]) + '\n\n')

            format_negatives += format_negative
            span_negatives += span_negative
            if sent_invalid:
                negative_type = ''
                invalid_sents += 1
                if format_negative:
                    negative_type += 'format_negative '
                if span_negative:
                    negative_type += 'span_negative '
                if pred_entities != tgt_spans[idx]:
                    negative_type += 'Wrong '
                self.f_invalid.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx])
                    + '\nNegative type: ' + negative_type + '\n\n')
            if pred_entities != tgt_spans[idx]:
                wrong_type = ''
                negative += 1
                if sent_invalid:
                    wrong_type += 'has invalid'
                self.f_wrong.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx])
                    + '\nInvalid: ' + wrong_type + '\n\n')
            else:
                if sent_invalid:
                    false_positive += 1
                else:
                    true_positive += 1
            pred.append(pred_entities)

        assert true_positive + false_positive + negative == len(raw_words)

        return {'pred': pred,
                'target': tgt_spans,
                'invalid_sent': invalid_sents,
                'format_negative': format_negatives,
                'span_negative': span_negatives,
                'false_positive': false_positive,
                'true_positive': true_positive,
                'negative': negative}


class COTZhNERModel(nn.Module):
    def __init__(self, dataset, k, tgt_vocab, llm_model_name="text-davinci-003", random_ins=1,
                 encoding_type='bio', temperature=0.5):
        super(COTZhNERModel, self).__init__()
        self.dataset = dataset
        self.k = k
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
        self.encoding_type = encoding_type
        self.temperature = temperature
        if random_ins:
            self.rng = random.Random()
        else:
            self.rng = random.Random(123)
            # all labels

        label_list = list(tgt_vocab.word2idx.keys())
        self.label_list = [label2desc[l.upper()] for l in label_list]

        # instruction
        self.prefix_all = "请识别给定句子中的" + '、'.join([label for label in self.label_list]) + "实体。\n"
        # cot example set
        if dataset == 'msra':
            prefix_cot = prefix_cot_mrsa
        elif dataset == 'ontonotes4':
            prefix_cot = prefix_cot_ontonotes4
        else:
            raise ValueError(f'Invalid dataset: {dataset}')
        self.prefix_cot = prefix_cot

        # output files
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if not os.path.exists('outputs/invalid'):
            os.mkdir('outputs/invalid')
        if not os.path.exists('outputs/wrong'):
            os.mkdir('outputs/wrong')
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_cot_{str(self.k)}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_cot_{str(self.k)}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_cot_{str(self.k)}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_spans):
        demonstrations = self.rng.sample(self.prefix_cot, self.k)
        ner_prefix = self.prefix_all + ''.join(demonstrations)

        invalid_sents = 0
        format_negatives = 0
        span_negatives = 0
        false_positive = 0
        true_positive = 0
        negative = 0
        pred = []
        for idx, _raw_words in enumerate(tqdm(raw_words)):
            sent = '句子：' + ''.join(_raw_words) + '\n'
            prompt = ner_prefix + sent + '实体：'
            response = completions_with_gpt_backoff(model=self.llm_model_name, prompt=prompt,
                                                    temperature=self.temperature, max_tokens=512)
            pred_text = response['choices'][0]['text']
            pred_entities, sent_invalid, format_negative, span_negative = format_cn_cot_entity(pred_text, _raw_words,
                                                                                               self.label_list)
            self.f_all.write(
                prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx]) + '\n\n')
            format_negatives += format_negative
            span_negatives += span_negative
            if sent_invalid:
                negative_type = ''
                invalid_sents += 1
                if format_negative:
                    negative_type += 'format_negative '
                if span_negative:
                    negative_type += 'span_negative '
                if pred_entities != tgt_spans[idx]:
                    negative_type += 'Wrong '
                self.f_invalid.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx])
                    + '\nNegative type: ' + negative_type + '\n\n')
            if pred_entities != tgt_spans[idx]:
                wrong_type = ''
                negative += 1
                if sent_invalid:
                    wrong_type += 'has invalid'
                self.f_wrong.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_spans[idx])
                    + '\nInvalid: ' + wrong_type + '\n\n')
            else:
                if sent_invalid:
                    false_positive += 1
                else:
                    true_positive += 1

            pred.append(pred_entities)
        assert true_positive + false_positive + negative == len(raw_words)

        return {'pred': pred,
                'target': tgt_spans,
                'invalid_sent': invalid_sents,
                'format_negative': format_negatives,
                'span_negative': span_negatives,
                'false_positive': false_positive,
                'true_positive': true_positive,
                'negative': negative}


def get_conll_trigger(trigger_path):
    with open(trigger_path) as f:
        raw_triggers = f.readlines()

    sents = []
    prefixs = []
    ins_prefix = ''
    sent = []
    label = []
    for line in raw_triggers:
        if line == '\n':
            if sent not in sents:
                if len(ins_prefix):
                    if 'Therefore,' in ins_prefix:
                        prefixs.append(ins_prefix + '\n')
                    else:
                        prefixs.append(ins_prefix + 'None.\n')
                ins_prefix = "Sentence: " + ' '.join(sent) + '\n' + 'Entity: '
                sents.append(sent)
            ins_triggers = defaultdict(list)
            ins_label = []
            for idx in range(len(sent)):
                if '-' in label[idx]:
                    tag = label[idx].split('-')[0]
                    if tag == 'T':
                        ins_triggers[label[idx]].append(sent[idx])
                    else:
                        ins_label.append(sent[idx])
                        ins_type = label[idx].split('-')[1]
            ins_label = ' '.join(ins_label)
            ins_triggers = [' '.join(ins_triggers[tag]) for tag in ins_triggers]
            ins_prefix += ins_label + ' is described by ' + ', '.join(ins_triggers) + \
                          '. Therefore, ' + ins_label + ' is a ' + label2desc[ins_type] + ' entity; '
            sent = []
            label = []
        else:
            sent.append(line.split()[0])
            label.append(line.split()[1])

    if len(ins_prefix):
        prefixs.append(ins_prefix + '\n')
    return prefixs
