import torch
from torch.utils.data import TensorDataset


def _load_clue_tsv_data(path, tokenizer, max_seq_length):
    contents = []
    isTest = False
    sentence_len = 0
    with open(path, 'r', encoding='UTF-8') as f:
        # 第一行不要
        header = f.readline().strip().split('\t')
        if 'label' not in header:
            isTest = True
        sentence_len = len(header) if isTest else len(header)-1
        for line in f:
            lin_sp = line.strip().split('\t')

            if sentence_len == 1:
                token = tokenizer(lin_sp[0], max_length=max_seq_length, truncation='longest_first', padding='max_length')
            elif sentence_len == 2:
                token = tokenizer(lin_sp[0], lin_sp[1], max_length=max_seq_length, truncation="only_second",
                                  padding="max_length")
            label = 0 if isTest else int(lin_sp[-1])
            contents.append((token, label))
            # if len(contents) >= 10:
            #     break
    return contents


def data_loader(path, tokenizer, max_seq_length):
    data = _load_clue_tsv_data(path,tokenizer, max_seq_length)
    tensor_label = torch.tensor([f[1] for f in data], dtype=torch.long)
    tensor_input_ids = torch.tensor([f[0].input_ids for f in data], dtype=torch.long)
    tensor_token_type_ids = torch.tensor([f[0].token_type_ids for f in data], dtype=torch.long)
    tensor_attention_mask = torch.tensor([f[0].attention_mask for f in data], dtype=torch.long)

    return TensorDataset(tensor_input_ids,
                         tensor_token_type_ids,
                         tensor_attention_mask,
                         tensor_label)
