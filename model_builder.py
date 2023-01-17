from torch.nn import Linear
from models.network.TextCNN import TextCNN


def build_encoder(args):
    name = args.bert_name.split('/')[-1]

    # bert format
    if name in ['bert-base-chinese', 'bert-base-uncased', 'mengzi-bert-base']:
        from transformers import BertTokenizer as Tokenizer, \
            BertModel as Model, \
            BertConfig as Config

    # roberta format
    if name in ['chinese-roberta-wwm-ext', 'roberta-large-wwm-chinese-cluecorpussmall', 'roberta-base-finetuned-jd-binary-chinese']:
        from transformers import AutoTokenizer as Tokenizer, \
            AutoModel as Model, \
            AutoConfig as Config

    return Tokenizer, Config, Model


def build_top(args):
    top_dict = {
        'textcnn': TextCNN
    }
    name = args.top_name
    return top_dict[name]


def build_target(args, basetask):
    target_dict = {
        'linear': Linear
    }
    # 每个任务可以在json中设定自己的target网络，如果没有的话，就使用在args里设定的默认target。
    name = basetask.json_args.get('target') if basetask.json_args.get('target') is not None else args.target
    return target_dict[name]

