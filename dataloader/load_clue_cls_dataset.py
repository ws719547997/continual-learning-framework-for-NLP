import torch
from torch.utils.data import TensorDataset

from tokenizer.fineturn_tokenizer import single_sentence_token, sentence_pair_token


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

def clue_loader(task, tokenizer):
    task.name = task.json_args['task_name']
    task.task_type = task.json_args['task_type']
    task.task_output = task.json_args['task_output']
    task.language = task.json_args['language']
    task.target = task.json_args['target']

    [dataset, sub_dataset] = task.name.split('.')

    if sub_dataset in ['afqmc', 'cluewsc2020', 'cmnli', 'csl', 'tnews', 'iflytek']:
        for data_type in ['train', 'test_nolabel', 'dev']:
            path = f'datasets/{dataset}/{sub_dataset}/{data_type}.tsv'
            sentence1, sentence2, labels = _load_cls_format_data(path, tokenizer, task.args.max_seq_length)
            if len(sentence2) == 0:
                features = single_sentence_token(sentence1, tokenizer, task.args.max_seq_length)
            else:
                features = sentence_pair_token(sentence1, sentence2, tokenizer, task.args.max_seq_length)

            tensor_label = torch.tensor(labels, dtype=torch.long)
            tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            tensor_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            tensor_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

            tensor_dataset = TensorDataset(tensor_input_ids,
                                           tensor_token_type_ids,
                                           tensor_attention_mask,
                                           tensor_label)

            # split a testset with label from training data
            if data_type == 'train':
                tensor_dataset_len = len(tensor_dataset)
                task.get_dataset(TensorDataset(*tensor_dataset[0:int(tensor_dataset_len * 0.8)]), data_type)
                task.get_dataset(TensorDataset(*tensor_dataset[int(tensor_dataset_len * 0.8):]), 'test')
            else:
                task.get_dataset(tensor_dataset, data_type)

        if sub_dataset == 'cluewsc2020':
            task.label = ['false', 'true']
            task.task_output = len(task.label)
        elif sub_dataset == 'cmnli':
            task.label = ['entailment', 'neutral', 'contradiction']
            task.task_output = len(task.label)
        elif sub_dataset == 'tnews':
            import json
            with open("datasets/clue/tnews/labels.json", 'r', encoding='utf-8') as f:
                task.label = [json.loads(i)['label_desc'] for i in f]
            task.task_output = len(task.label)
        elif sub_dataset == 'iflytek':
            import json
            with open("datasets/clue/iflytek/labels.json", 'r', encoding='utf-8') as f:
                task.label = [json.loads(i)['label_des'] for i in f]
            task.task_output = len(task.label)

    return task
