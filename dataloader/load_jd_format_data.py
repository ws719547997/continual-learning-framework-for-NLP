import torch
from torch.utils.data import TensorDataset


def _load_jd_txt_data(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            if len(lin.split('\t')) < 5:
                continue
            lin_sp = lin.split('\t')
            content = lin_sp[4]
            label = 0 if lin_sp[2] == 'NEG' else 1
            contents.append((content, label))
    return contents


def _padding(line, max_seq_length):
    """
    把每个list都补长到指定维数
    """
    while len(line) < max_seq_length:
        line.append(0)
    return line


def _process_dsc_data(data, tokenizer, max_seq_length):
    # token 是转化为数字和对齐
    padded_data = []
    for (sentence, label) in data:
        token = tokenizer(sentence, truncation='longest_first', max_length=max_seq_length)
        for key in token:
            token[key] = _padding(token[key], max_seq_length)
        padded_data.append((token, label))
    return padded_data


def data_loader(path, tokenizer, max_seq_length):
    raw_data = _load_jd_txt_data(path)
    data = _process_dsc_data(raw_data, tokenizer, max_seq_length)
    tensor_label = torch.tensor([f[1] for f in data], dtype=torch.long)
    tensor_input_ids = torch.tensor([f[0].input_ids for f in data], dtype=torch.long)
    tensor_token_type_ids = torch.tensor([f[0].token_type_ids for f in data], dtype=torch.long)
    tensor_attention_mask = torch.tensor([f[0].attention_mask for f in data], dtype=torch.long)

    return TensorDataset(tensor_input_ids,
                         tensor_token_type_ids,
                         tensor_attention_mask,
                         tensor_label)
