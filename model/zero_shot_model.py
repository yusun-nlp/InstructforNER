import os
from tqdm import tqdm

import backoff
import openai
from openai.error import RateLimitError
import torch.nn as nn

from model.utils import label2desc, format_en_entities, format_cn_entities

# TODO: add your API key here
api_keys = ['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']


@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_gpt_backoff(**kwargs):
    api_idx = 0
    timeout_ = True
    while timeout_:
        openai.api_key = api_keys[api_idx]
        try:
            response = openai.Completion.create(**kwargs)
            timeout_ = False
        except Exception:
            timeout_ = True
            api_idx += 1
            if api_idx == len(api_keys):
                raise
    return response


class ZeroEnNERModel(nn.Module):
    def __init__(self, dataset, tgt_vocab, prompt_type, encoding_type, llm_model_name="text-davinci-003",
                 temperature=0.5):
        super(ZeroEnNERModel, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
        self.encoding_type = encoding_type
        self.temperature = temperature
        # 所有的label
        label_list = list(tgt_vocab.word2idx.keys())
        self.label_list = [label2desc[l.upper()] for l in label_list]
        self.num_labels = len(self.label_list)
        # instructions
        self.prefix = "Please identify " + ', '.join([label for label in self.label_list]) + \
                      " Entity from the given sentence."
        if 'ace' in dataset:
            self.prefix += " Notice that there may exist nested entities."

        if prompt_type == 'struct':
            self.prefix += " Each entity type in a line, and use \",\" to separate entities with same entity type. " \
                          "If no entity, output None."
        self.prefix += '\n'
        # output files
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_all_{prompt_type}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_all_{prompt_type}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_all_{prompt_type}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_tokens):
        invalid_sents = 0
        format_negatives = 0
        span_negatives = 0
        false_positive = 0
        true_positive = 0
        negative = 0
        pred = []
        for idx, _raw_words in enumerate(tqdm(raw_words)):
            sent = "Sentence: " + ' '.join(_raw_words) + '\n'
            prompt = self.prefix + sent + "Entity:"
            response = completions_with_gpt_backoff(model=self.llm_model_name, prompt=prompt,
                                                    temperature=self.temperature)
            pred_text = response['choices'][0]['text']
            pred_entities, sent_invalid, format_negative, span_negative = format_en_entities(pred_text, _raw_words,
                                                                                             self.label_list)
            self.f_all.write(
                prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx]) + '\n\n')
            format_negatives += format_negative
            span_negatives += span_negative
            if sent_invalid:
                negative_type = ''
                invalid_sents += 1
                if format_negative:
                    negative_type += 'format_negative '
                if span_negative:
                    negative_type += 'span_negative '
                if pred_entities != tgt_tokens[idx]:
                    negative_type += 'Wrong '
                self.f_invalid.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx])
                    + '\nNegative type: ' + negative_type + '\n\n')
            if pred_entities != tgt_tokens[idx]:
                wrong_type = ''
                negative += 1
                if sent_invalid:
                    wrong_type += 'has invalid'
                self.f_wrong.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx])
                    + '\nInvalid: ' + wrong_type + '\n\n')
            else:
                if sent_invalid:
                    false_positive += 1
                else:
                    true_positive += 1

            pred.append(pred_entities)
        assert true_positive + false_positive + negative == len(raw_words)

        return {'pred': pred,
                'target': tgt_tokens,
                'invalid_sent': invalid_sents,
                'format_negative': format_negatives,
                'span_negative': span_negatives,
                'false_positive': false_positive,
                'true_positive': true_positive,
                'negative': negative}


class ZeroZhNERModel(nn.Module):
    def __init__(self, dataset, tgt_vocab, prompt_type, encoding_type, llm_model_name="text-davinci-003",
                 temperature=0.5):
        super(ZeroZhNERModel, self).__init__()
        self.tgt_vocab = tgt_vocab
        self.llm_model_name = llm_model_name
        self.encoding_type = encoding_type
        self.temperature = temperature
        # 所有的label
        label_list = list(tgt_vocab.word2idx.keys())
        self.label_list = [label2desc[l.upper()] for l in label_list]
        self.num_labels = len(self.label_list)
        # instructions
        # instructions
        if prompt_type == 'raw':
            self.system = "你是一个从句子中抽取命名实体的智能专家。"
            self.prefix = "请识别给定句子中的" + '、'.join([label for label in self.label_list]) + "实体。\n"
        else:
            self.system = "你是一个从句子中抽取命名实体的智能专家。你应该按照我要求的格式输出。"
            self.prefix = '请识别给定句子中的' + '、'.join([label for label in self.label_list]) + \
                          '实体。实体类型之间用分号隔开，相同实体类型的每个实体用逗号隔开，形式为实体类型：实体。如果没有实体，返回无。\n'
        # output files
        cur_time = os.environ['FASTNLP_LAUNCH_TIME']
        self.f_invalid = open(f'outputs/invalid/{dataset}_all_{prompt_type}_invalid_{llm_model_name}_{cur_time}.txt',
                              'a')
        self.f_wrong = open(f'outputs/wrong/{dataset}_all_{prompt_type}_wrong_{llm_model_name}_{cur_time}.txt', 'a')
        self.f_all = open(f'outputs/{dataset}_all_{prompt_type}_{llm_model_name}_{cur_time}.txt', 'a')

    def forward(self, raw_words, tgt_tokens):
        invalid_sents = 0
        format_negatives = 0
        span_negatives = 0
        false_positive = 0
        true_positive = 0
        negative = 0
        pred = []
        for idx, _raw_words in enumerate(tqdm(raw_words)):
            sent = "句子：" + ''.join(_raw_words) + '\n'
            prompt = self.prefix + sent + "实体："
            response = completions_with_gpt_backoff(model=self.llm_model_name, prompt=prompt,
                                                    temperature=self.temperature, max_tokens=512)
            pred_text = response['choices'][0]['text']
            pred_entities, sent_invalid, format_negative, span_negative = format_cn_entities(pred_text, _raw_words,
                                                                                             self.label_list)
            self.f_all.write(
                prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx]) + '\n\n')
            format_negatives += format_negative
            span_negatives += span_negative
            if sent_invalid:
                negative_type = ''
                invalid_sents += 1
                if format_negative:
                    negative_type += 'format_negative '
                if span_negative:
                    negative_type += 'span_negative '
                if pred_entities != tgt_tokens[idx]:
                    negative_type += 'Wrong '
                self.f_invalid.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx])
                    + '\nNegative type: ' + negative_type + '\n\n')
            if pred_entities != tgt_tokens[idx]:
                wrong_type = ''
                negative += 1
                if sent_invalid:
                    wrong_type += 'has invalid'
                self.f_wrong.write(
                    prompt + pred_text + '\nPred: ' + str(pred_entities) + '\nTarget: ' + str(tgt_tokens[idx])
                    + '\nInvalid: ' + wrong_type + '\n\n')
            else:
                if sent_invalid:
                    false_positive += 1
                else:
                    true_positive += 1

            pred.append(pred_entities)
        assert true_positive + false_positive + negative == len(raw_words)

        return {'pred': pred,
                'target': tgt_tokens,
                'invalid_sent': invalid_sents,
                'format_negative': format_negatives,
                'span_negative': span_negatives,
                'false_positive': false_positive,
                'true_positive': true_positive,
                'negative': negative}
