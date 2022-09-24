def single_sentence_token(data, tokenizer, max_seq_length):
    # token 是转化为数字和对齐
    padded_data = []
    for d in data:
        token = tokenizer(d, max_length=max_seq_length, truncation='longest_first', padding='max_length')
        padded_data.append(token)
    return padded_data


def sentence_pair_token(sentence1, sentence2, tokenizer, max_seq_length):
    padded_data = []
    l = len(sentence1)
    for i in range(l):
        token = tokenizer(sentence1[i], sentence2[i],
                          max_length=max_seq_length, truncation="only_second", padding="max_length", )
        padded_data.append(token)
    return padded_data

