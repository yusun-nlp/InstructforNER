from functools import cmp_to_key
import json
from tqdm import tqdm

from datasets import load_dataset
from fastNLP import Vocabulary, DataSet, Instance
from fastNLP.io import DataBundle, Pipe, Loader, Conll2003NERLoader, MsraNERLoader
from fastNLP.core.metrics.span_f1_pre_rec_metric import _bio_tag_to_spans, _bmeso_tag_to_spans
import numpy as np


class NERPipe(Pipe):
    def __init__(self, dataset):
        super(NERPipe, self).__init__()
        self.dataset = dataset
        self.vocab = Vocabulary(unknown=None, padding=None)

    def process(self, data_bundle):
        def label_process(ins):
            self.vocab.update(ins['entity_tags'])
            tgt_labels = []
            for idx, tag in enumerate(ins['entity_tags']):
                tgt_labels.append((tag, (ins['entity_spans'][idx][0], ins['entity_spans'][idx][1])))
            return Instance(tgt_tokens=tgt_labels)

        data_bundle.apply_more(label_process, progress_bar='rich', progress_desc='get vocab')
        data_bundle.set_vocab(self.vocab, 'tgt_tokens')
        data_bundle.set_ignore('entity_spans', 'entity_tags', 'prev_contxt', 'after_contxt')
        return data_bundle

    def process_from_file(self, paths):
        if 'ace' in self.dataset:
            data_bundle = NestedLoader().load(paths)
        elif self.dataset == 'conll03':
            data_bundle = Conll2003NERLoader().load(paths)
            data_bundle.apply_field(_bio_tag_to_spans, field_name='target', new_field_name='target')
        elif self.dataset == 'ontonotes4':
            data_bundle = OntoNotes4NERLoader().load(paths)
            data_bundle.apply_field(_bmeso_tag_to_spans, field_name='target', new_field_name='target')
        elif self.dataset == 'msra':
            data_bundle = MsraNERLoader().load()
            data_bundle.rename_field('raw_chars', 'raw_words')
            data_bundle.apply_field(_bio_tag_to_spans, field_name='target', new_field_name='target')
        elif self.dataset == 'ontonotes5':
            data_bundle = OntoNotes5NERLoader().load(paths)
            data_bundle.apply_field(_bio_tag_to_spans, field_name='target', new_field_name='target')
        else:
            if 'disease' in self.dataset:
                vocab = ['O', 'B-Disease', 'I-Disease']
            elif 'chem' in self.dataset:
                vocab = ['O', 'B-Chemical', 'I-Chemical']
            elif 'GM' in self.dataset:
                vocab = ['O', 'B-Gene', 'I-Gene']
            data_bundle = DataBundle()
            dataset = load_dataset('EMBO/BLURB', self.dataset)
            dataset = dataset.rename_columns({'tokens': 'raw_words', 'ner_tags': 'target'})
            data_bundle.set_dataset(DataSet(dataset['train'].to_dict())[:-1], 'train')
            data_bundle.set_dataset(DataSet(dataset['validation'].to_dict())[:-1], 'dev')
            data_bundle.set_dataset(DataSet(dataset['test'].to_dict())[:-1], 'test')
            data_bundle.apply_field(lambda x: [vocab[idx] for idx in x], field_name='target', new_field_name='target',
                                    progress_desc='convert index to label')
            data_bundle.apply_field(_bio_tag_to_spans, field_name='target', new_field_name='target')
        data_bundle = self.process(data_bundle)
        return data_bundle


class NestedLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__()
        self.max_sent_len = 10000

    def _load(self, path):
        def cmp(v1, v2):
            v1 = v1[1]
            v2 = v2[1]
            if v1[0] == v2[0]:
                return v1[1] - v2[1]
            return v1[0] - v2[0]

        ds = DataSet()
        invalid_ent = 0
        max_len = 0
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), leave=False):
                data = json.loads(line.strip())

                all_entities = data['ners']
                all_sentences = data['sentences']

                assert len(all_entities) == len(all_sentences)

                new_all_sentences = []
                new_all_entities = []
                for idx, sent in enumerate(all_sentences):
                    ents = all_entities[idx]
                    if len(sent) > self.max_sent_len:
                        has_entity_cross = np.zeros(len(sent))
                        for (start, end, tag) in ents:
                            has_entity_cross[start:end + 1] = 1  # 如果1为1的地方说明是有span穿过的

                        punct_indexes = []
                        for idx, word in enumerate(sent):
                            if word.endswith('.') and has_entity_cross[idx] == 0:
                                punct_indexes.append(idx)
                        last_index = 0
                        for idx in punct_indexes:
                            if idx - last_index > self.max_sent_len:
                                new_all_sentences.append(sent[last_index:idx + 1])
                                new_ents = [(e[0] - last_index, e[1] - last_index, e[2]) for e in ents if
                                            last_index <= e[1] <= idx]  # 是闭区间
                                new_all_entities.append(new_ents)
                                last_index = idx + 1
                        if last_index < len(sent):
                            new_all_sentences.append(sent[last_index:])
                            new_ents = [(e[0] - last_index, e[1] - last_index, e[2]) for e in ents if
                                        last_index <= e[1]]  # 是闭区间
                            new_all_entities.append(new_ents)
                    else:
                        new_all_sentences.append(sent)
                        new_all_entities.append(ents)
                if sum(map(len, all_entities)) != sum(map(len, new_all_entities)):
                    print("Mismatch number sentences")
                if sum(map(len, all_sentences)) != sum(map(len, new_all_sentences)):
                    print("Mismatch number entities")

                all_entities = new_all_entities
                all_sentences = new_all_sentences

                for i in range(len(all_entities)):
                    all_spans = []
                    raw_words = all_sentences[i]
                    max_len = max(len(raw_words), max_len)
                    ents = all_entities[i]
                    for start, end, tag in ents:
                        # assert start<=end, (start, end, i)
                        if start > end:
                            invalid_ent += 1
                            continue
                        all_spans.append((tag, (start, end + 1)))
                        assert end < len(raw_words), (end, len(raw_words))

                    all_spans = sorted(all_spans, key=cmp_to_key(cmp))

                    entities = []
                    entity_tags = []
                    entity_spans = []
                    for tag, (start, end) in all_spans:
                        entities.append(raw_words[start:end])
                        entity_tags.append(tag.lower())
                        entity_spans.append([start, end])

                    prev_contxt = []
                    after_contxt = []

                    if i > 0:
                        prev_contxt = all_sentences[:i]
                    if i < len(all_sentences) - 1:
                        after_contxt = all_sentences[i + 1:]

                    assert len(after_contxt) + len(prev_contxt) == len(all_sentences) - 1

                    ds.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                                       entity_spans=entity_spans,
                                       prev_contxt=prev_contxt, after_contxt=after_contxt))
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        print(f"for `{path}`, {invalid_ent} invalid entities. max sentence has {max_len} tokens")
        return ds


class OntoNotes4NERLoader(Loader):
    def __init__(self):
        super(OntoNotes4NERLoader, self).__init__()

    def _load(self, path):
        with open(path) as f:
            raw_input = f.readlines()
        raw_words = []
        cur_sent = []
        target = []
        cur_tgt = []
        for word in raw_input:
            if word == '\n' and len(cur_sent) != 0:
                raw_words.append(cur_sent)
                target.append(cur_tgt)
                cur_sent = []
                cur_tgt = []
            else:
                cur_sent.append(word.split()[0])
                cur_tgt.append(word.split()[1])
        if cur_sent:
            raw_words.append(cur_sent)
            target.append(cur_tgt)
        return DataSet({"raw_words": raw_words,
                        "target": target})


class OntoNotes5NERLoader(Loader):
    def __init__(self):
        super(OntoNotes5NERLoader, self).__init__()

    def _load(self, path):
        with open(path) as f:
            raw_input = f.readlines()
        raw_words = []
        cur_sent = []
        target = []
        cur_tgt = []
        for word in raw_input:
            if word == '\n' and len(cur_sent) != 0:
                raw_words.append(cur_sent)
                target.append(cur_tgt)
                cur_sent = []
                cur_tgt = []
            else:
                cur_sent.append(word.split()[0])
                if word.split()[1].split('-')[-1] in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL',
                                                      'CARDINAL']:
                    cur_tgt.append('O')
                else:
                    cur_tgt.append(word.split()[1])
        if cur_sent:
            raw_words.append(cur_sent)
            target.append(cur_tgt)
        return DataSet({"raw_words": raw_words,
                        "target": target})


def cmp(v1, v2):
    v1 = v1[-1]
    v2 = v2[-1]
    if v1[0] == v2[0]:
        return v1[-1] - v2[-1]
    return v1[0] - v2[0]


if __name__ == '__main__':
    paths = {'test': "../../../data/conll2003/test.txt",
             'train': "../../../data/conll2003/train.txt",
             'dev': "../../../data/conll2003/testa.txt"}
    pipe = NERPipe("conll03")
    data_bundle = pipe.process_from_file(paths)
    print(data_bundle)
