import torch
from torch.utils.data import TensorDataset

from tokenizer.fineturn_tokenizer import single_sentence_token


def _load_jd_format_data(path):
    contents = []
    labels = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            if len(lin.split('\t')) < 5:
                continue
            lin_sp = lin.split('\t')
            contents.append(lin_sp[4])
            labels.append(0 if lin_sp[2] == 'NEG' else 1)
    return contents, labels


def dsc_loader(task, tokenizer):
    # 基础信息
    task.name = task.json_args['task_name']
    task.task_type = task.json_args['task_type']
    task.task_output = task.json_args['task_output']
    task.language = task.json_args['language']
    task.target = task.json_args['target']
    task.label = ['POS', 'NEG']

    # 读取数据
    [dataset, sub_dataset] = task.name.split('.')
    for data_type in ['train', 'test', 'dev']:
        path = f'datasets/{dataset}/data/{data_type}/{sub_dataset}.txt'

        contents, labels = _load_jd_format_data(path)
        features = single_sentence_token(contents, tokenizer, task.args.max_seq_length)

        tensor_label = torch.tensor(labels, dtype=torch.long)
        tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        tensor_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        tensor_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

        tensor_dataset = TensorDataset(tensor_input_ids,
                                       tensor_token_type_ids,
                                       tensor_attention_mask,
                                       tensor_label)
        task.get_dataset(tensor_dataset, data_type)
    return task
