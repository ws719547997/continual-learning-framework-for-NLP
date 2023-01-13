from datasets import ClassLabel


def _set_tokenizer(name_or_path: str):
    """
    set tokenizer by name.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return tokenizer


class Tokenizer:
    def __init__(self, args) -> None:
        # tokenizer config
        self.tokenizer = _set_tokenizer(args.bert_name)
        self.max_length: int = args.max_seq_length
        self.truncation: str = 'longest_first'  # "only_second"
        self.padding: str = 'max_length'

        # task classlabel info
        self.class_label: ClassLabel = None

    # 针对不同任务的处理函数，从数据集的格式转化为模型的输入
    def cls_single_sentence(self, example: dict) -> dict:
        """
        input format:{
            'sentence': 'this is a sentence.'
        }
        """
        return self.tokenizer(example['sentence'], max_length=self.max_length, truncation=self.truncation,
                              padding=self.padding, return_tensors='np')

    def nli_two_sentence(self, example: dict) -> dict:
        """
        input format:{
            'sentence1': 'this is first sentence.',
            'sentence2': 'this is second sentence.'
        }
        """
        return self.tokenizer(example['sentence1'], example['sentence2'],
                              max_length=self.max_length, truncation=self.truncation, padding=self.padding,
                              return_tensors='np')

    def ner_tokens_BIO(self, example: dict) -> dict:
        """
        input format:{
            'tokens': ['EU','rejects','German','call','to','boycott','British','lamb','.'],
            'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]
        }
        using BIO tag, seeing conll2003 and msra_ner
        """
        token_dict = self.tokenizer(' '.join(example['tokens']), max_length=self.max_length, truncation=self.truncation,
                                    padding=self.padding, return_tensors='np')

        # first [0] for CLS, second [0] for SEP, rest of zeros for padding.
        if len(example['ner_tags']) <= self.max_length - 2:
            token_dict['ner_tags'] = [0] + example['ner_tags'] + [0] + [0] * (
                        self.max_length - 2 - len(example['ner_tags']))
        else:
            token_dict['ner_tags'] = [0] + example['ner_tags'][:self.max_length - 2] + [0]
        return token_dict

