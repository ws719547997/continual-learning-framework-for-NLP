import torch
from torch.utils.data import TensorDataset

from tokenizer.tokenizer import single_sentence_token, sentence_pair_token
from dataloader.task_manage import BaseTask


def _load_cls_format_data(path, tokenizer, max_seq_length):
    sentence1 = []
    sentence2 = []
    labels = []
    isTest = False
    with open(path, 'r', encoding='UTF-8') as f:
        # 第一行不要
        header = f.readline().strip().split('\t')
        if 'label' not in header:
            isTest = True
        sentence_len = len(header) if isTest else len(header) - 1
        for line in f:
            lin_sp = line.strip().split('\t')
            if sentence_len == 1:
                sentence1.append(lin_sp[0])
            elif sentence_len == 2:
                sentence1.append(lin_sp[0])
                sentence2.append(lin_sp[1])
            labels.append(0 if isTest else int(lin_sp[-1]))
    return sentence1, sentence2, labels

def loader(name, tokenizer, max_seq_length):
    task = BaseTask()
    [dataset, sub_dataset] = name.split('.')

    if sub_dataset in ['afqmc', 'cluewsc2020', 'cmnli', 'csl', 'tnews', 'iflytek']:
        for data_type in ['train', 'test_nolabel', 'dev']:
            path = f'data/{dataset}/{sub_dataset}/{data_type}.tsv'
            sentence1, sentence2, labels = _load_cls_format_data(path, tokenizer, max_seq_length)
            if len(sentence2) == 0:
                features = single_sentence_token(sentence1, tokenizer, max_seq_length)
            else:
                features = sentence_pair_token(sentence1, sentence2, tokenizer, max_seq_length)

            tensor_label = torch.tensor(labels, dtype=torch.long)
            tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            tensor_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            tensor_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

            tensor_dataset = TensorDataset(tensor_input_ids,
                                           tensor_token_type_ids,
                                           tensor_attention_mask,
                                           tensor_label)

            # split a testset with label from trainng data
            if data_type == 'train':
                tensor_dataset_len = len(tensor_dataset)
                task.get_dataset(TensorDataset(*tensor_dataset[0:int(tensor_dataset_len * 0.8)]), data_type)
                task.get_dataset(TensorDataset(*tensor_dataset[int(tensor_dataset_len * 0.8):]), 'test')
            else:
                task.get_dataset(tensor_dataset, data_type)

        task.name = name
        task.language = "zh"
        if sub_dataset == 'afqmc':
            task.task_output = 2
            task.task_type = "matching"
        elif sub_dataset == 'cluewsc2020':
            task.task_output = 2
            task.label = ['false', 'true']
            task.task_type = 'co-reference resolution'
        elif sub_dataset == 'cmnli':
            task.task_output = 3
            task.label = ['entailment', 'neutral', 'contradiction']
            task.task_type = "nli"
        elif sub_dataset == 'csl':
            task.task_output = 2
            task.task_type = "matching"
        elif sub_dataset == 'tnews':
            import json
            with open("data/clue/tnews/labels.json", 'r', encoding='utf-8') as f:
                task.label = [json.loads(i)['label_desc'] for i in f]
            task.task_output = len(task.label)
            task.task_type = 'cls'
        elif sub_dataset == 'iflytek':
            import json
            with open("data/clue/iflytek/labels.json", 'r', encoding='utf-8') as f:
                task.label = [json.loads(i)['label_des'] for i in f]
            task.task_output = len(task.label)
            task.task_type = 'cls'

    return task
