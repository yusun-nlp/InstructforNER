from collections import defaultdict
import os
import random
from tqdm import tqdm

import torch.nn as nn

from model.utils import label2desc, format_en_entities, format_cn_entities, completions_with_gpt_backoff


class ICLEnNERModel(nn.Module):
    def __init__(self, dataset, trainset, k, tgt_vocab, llm_model_name="text-davinci-003", random_ins=1,
                 temperature=0.5):
        super(ICLEnNERModel, self).__init__()
        self.dataset = dataset
        self.trainset = trainset
        self.k = k
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
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
                      " Entity from the given sentence."
        if 'ace' in dataset:
            self.prefix_all += " Notice that there may exist nested entities."
        self.prefix_all += '\n'
        # in-context examples set
        prefix_icl = []
        for ins in trainset:
            ins_prefix = 'Sentence: ' + ' '.join(ins['raw_words']) + '\n' + "Entity: "
            tgt = ins['tgt_spans']
            tgt_dict = defaultdict(list)
            for t in tgt:
                tgt_dict[t[0]].append(' '.join(ins['raw_words'][t[1][0]:t[1][1]]))
            if len(tgt_dict) == 0:
                ins_prefix += 'None\n'
            elif len(self.label_list) == 1:
                ins_prefix += ', '.join(tgt_dict[list(tgt_dict.keys())[0]]) + '\n'
            else:
                label_dict = {key.lower(): value for key, value in label2desc.items()}
                for k, v in tgt_dict.items():
                    ins_prefix += label_dict[k] + ': ' + ', '.join(v) + '\n'
            prefix_icl.append(ins_prefix)
        self.prefix_icl = prefix_icl

        # output files
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if not os.path.exists('outputs/invalid'):
            os.mkdir('outputs/invalid')
        if not os.path.exists('outputs/wrong'):
            os.mkdir('outputs/wrong')
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_icl_{str(self.k)}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_icl_{str(self.k)}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_icl_{str(self.k)}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_spans):
        demonstrations = self.rng.sample(self.prefix_icl, self.k)

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
                                                    temperature=self.temperature)
            pred_text = response['choices'][0]['text']

            pred_entities, sent_invalid, format_negative, span_negative = format_en_entities(pred_text, _raw_words,
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


class ICLZhNERModel(nn.Module):
    def __init__(self, dataset, trainset, k, tgt_vocab, llm_model_name="text-davinci-003", random_ins=1,
                 temperature=0.5):
        super(ICLZhNERModel, self).__init__()
        self.dataset = dataset
        self.trainset = trainset
        self.k = k
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        if random_ins:
            self.rng = random.Random()
        else:
            self.rng = random.Random(123)

        # all labels
        label_list = list(tgt_vocab.word2idx.keys())
        self.label_list = [label2desc[l.upper()] for l in label_list]
        self.num_labels = len(self.label_list)

        # instructions
        self.prefix_all = "请识别给定句子中的" + '、'.join([label for label in self.label_list]) + "实体。\n"
        # in-context-examples set
        prefix_icl = []
        for ins in trainset:
            ins_prefix = '句子：' + ''.join(ins['raw_words']) + '\n' + "实体："
            tgt = ins['tgt_spans']
            tgt_dict = defaultdict(list)
            for t in tgt:
                tgt_dict[t[0]].append(''.join(ins['raw_words'][t[1][0]:t[1][1]]))
            if len(tgt_dict) == 0:
                ins_prefix += '无'
            elif len(self.label_list) == 1:
                ins_prefix += '，'.join(tgt_dict[list(tgt_dict.keys())[0]])
            else:
                label_dict = {key.lower(): value for key, value in label2desc.items()}
                labels_preifx = []
                for k, v in tgt_dict.items():
                    labels_preifx.append(label_dict[k] + '：' + '，'.join(v))
                ins_prefix += '；'.join(labels_preifx)
            prefix_icl.append(ins_prefix + '\n')
        self.prefix_icl = prefix_icl

        # output files
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        if not os.path.exists('outputs/invalid'):
            os.mkdir('outputs/invalid')
        if not os.path.exists('outputs/wrong'):
            os.mkdir('outputs/wrong')
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_icl_{str(self.k)}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_icl_{str(self.k)}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_icl_{str(self.k)}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_spans):
        demonstrations = self.rng.sample(self.prefix_icl, self.k)

        invalid_sents = 0
        format_negatives = 0
        span_negatives = 0
        false_positive = 0
        true_positive = 0
        negative = 0
        pred = []
        for idx, _raw_words in enumerate(tqdm(raw_words)):
            sent = '句子：' + ''.join(_raw_words) + '\n'
            ner_prefix = self.prefix_all + ''.join(demonstrations)
            prompt = ner_prefix + sent + '实体：'
            response = completions_with_gpt_backoff(model=self.llm_model_name, prompt=prompt,
                                                    temperature=self.temperature, max_tokens=512)
            pred_text = response['choices'][0]['text']
            pred_entities, sent_invalid, format_negative, span_negative = format_cn_entities(pred_text, _raw_words,
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
