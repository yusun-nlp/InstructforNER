import argparse
from random import sample

from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import Evaluator
import fitlog

from data.pipe import NERPipe
from model.zero_shot_model import ZeroEnNERModel, ZeroZhNERModel
from metrics import NERSpanMetrics

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="conll03", type=str,
                    choices=['BC5CDR-disease-IOB', 'BC5CDR-chem-IOB', 'BC2GM-IOB', 'conll03', 'ontonotes5',
                             'ontonotes4', 'msra', 'ace04', 'ace05'])
parser.add_argument('--test_sample', default=1000, type=int, help="if 0, use all test examples")
# large language model
parser.add_argument('--metric_type', default='all', type=str, choices=['span', 'all', 'cls'])
parser.add_argument('--prompt_type', default='raw', type=str, choices=['raw', 'struct'])
parser.add_argument('--llm', default="text-davinci-003", type=str, help='large language model',
                    choices=["text-davinci-003"])
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--tag', default='', type=str)

args = parser.parse_args()
dataset = args.dataset
test_sample = args.test_sample
metric_type = args.metric_type
prompt_type = args.prompt_type
llm_model_name = args.llm
temperature = args.temperature
tag = args.tag

# fitlog.debug()  # train的时候注释掉
fitlog.set_log_dir('logs/')  # set the logging directory
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters
fitlog.set_rng_seed(123)


# process data
@cache_results('caches/cache.pkl', _refresh=False)
def get_data(dataset):
    if dataset == 'ontonotes4':
        encoding_type = 'bmeso'
        paths = {'train': '../data/OntoNote4NER/train.char.bmes',
                 'dev': '../data/OntoNote4NER/dev.char.bmes',
                 'test': '../data/OntoNote4NER/test.char.bmes'}
    elif dataset == 'conll03':
        encoding_type = 'bio'
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/testa.txt"}
    elif dataset == 'ontonotes5':
        paths = "../../data/ontonotes/"
    elif dataset == 'ace04':
        paths = '../../data/en_ace04/'
    elif dataset == 'ace05':
        paths = '../../data/en_ace05/'
    else:
        paths = ''
    pipe = NERPipe(dataset=dataset)
    data_bundle = pipe.process_from_file(paths=paths)
    return data_bundle, encoding_type


data_bundle, encoding_type = get_data(dataset)
print(data_bundle)

# 随即拼接一些sample
testset = data_bundle.get_dataset('test')
if test_sample:
    test_sample = min(test_sample, len(testset))
    test_idxs = sample([idx for idx in range(len(testset))], test_sample)
    testset = testset[test_idxs]
test_dl = prepare_torch_dataloader(testset, batch_size=64, num_workers=4)

if dataset == 'ontonotes4' or dataset == 'msra':
    llm_model = ZeroZhNERModel(dataset, data_bundle.get_vocab('tgt_tokens'), prompt_type,
                               encoding_type=encoding_type, llm_model_name=llm_model_name,
                               temperature=temperature)
else:
    llm_model = ZeroEnNERModel(dataset, data_bundle.get_vocab('tgt_tokens'), prompt_type,
                               encoding_type=encoding_type, llm_model_name=llm_model_name, temperature=temperature)
metric = {'all': NERSpanMetrics()}

evaluator = Evaluator(model=llm_model,
                      driver='torch',
                      dataloaders=test_dl,
                      device='cpu',
                      metrics=metric,
                      fp16=False,
                      progress_bar='rich',
                      torch_kwargs={'ddp_kwargs': {'find_unused_parameters': True}})

eval_res = evaluator.run()
fitlog.add_best_metric(eval_res)
llm_model.f_wrong.close()
llm_model.f_invalid.close()
llm_model.f_all.close()

fitlog.finish()  # finish the logging
