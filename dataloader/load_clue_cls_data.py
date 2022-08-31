import torch
from torch.utils.data import TensorDataset


def _load_clue_tsv_data(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        # 第一行不要
        header = f.readline()
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_sp = lin.split('\t')
            sent1 = lin_sp[0]
            sent2 = lin_sp[1]
            # 这几个测试集都是没有标签的，那我测试个蛋？？？
            label = 0 if 'test' in path else int(lin_sp[2])
            contents.append((sent1, sent2, label))
    return contents

def _process_clue_cls_data(data, tokenizer, max_seq_length):
    # token 是转化为数字和对齐
    padded_data = []
    for (sentence1, sentence2, label) in data:
        token = tokenizer(sentence1, sentence2, max_length=max_seq_length, truncation="only_second", padding="max_length")
        padded_data.append((token, label))
    return padded_data


def data_loader(path, tokenizer, max_seq_length):
    raw_data = _load_clue_tsv_data(path)
    data = _process_clue_cls_data(raw_data, tokenizer, max_seq_length)
    tensor_label = torch.tensor([f[1] for f in data], dtype=torch.long)
    tensor_input_ids = torch.tensor([f[0].input_ids for f in data], dtype=torch.long)
    tensor_token_type_ids = torch.tensor([f[0].token_type_ids for f in data], dtype=torch.long)
    tensor_attention_mask = torch.tensor([f[0].attention_mask for f in data], dtype=torch.long)

    return TensorDataset(tensor_input_ids,
                         tensor_token_type_ids,
                         tensor_attention_mask,
                         tensor_label)
