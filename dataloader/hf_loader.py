from datasets import load_dataset, get_dataset_split_names, load_dataset_builder
from tokenization import Tokenizer

builder = load_dataset_builder("script\jd21.py", 'iPhone')
class_label = builder.info.features['label']
dataset = load_dataset("script\jd21.py", 'iPhone')

tokenizer = Tokenizer(args=None)
tokenizer.class_label = class_label

for split in get_dataset_split_names("script\jd21.py", 'iPhone'):
    dataset[split] = dataset[split].map(tokenizer.cls_single_sentence, batched=True)
    dataset[split].set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask","label"])