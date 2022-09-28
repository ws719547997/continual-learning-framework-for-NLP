from typing import List

import torch
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from models.encoder.automodel import AutoPTM

from torch.nn import Linear
from models.network.TextCNN import TextCNN

encoders_args_dict = {
    'bert': (BertTokenizer, BertConfig, BertModel),
    'auto': (AutoTokenizer, AutoConfig, AutoModel)
}

encoders_dict = {
    'auto':AutoPTM
}

targets_dict = {
    'linear': Linear,
    'textcnn': TextCNN
}



class Net(torch.nn.Module):
    def __init__(self, args, encoder, top, target):
        super(Net, self).__init__()
        self.encoder = encoder
        self.top = top
        self.target = torch.nn.ModuleList()
        for t in target:
            self.target.append(t)

    def forward(self, input_ids, segment_ids, input_mask, t=1):
        h = self.encoder(input_ids, segment_ids, input_mask)
        h = self.top(h)
        y = []
        for target in self.target:
            y.append(target(h))
        return y


def build_models(task_manager, args):
    Tokenizer, Config, Model = encoders_args_dict[args.bert_type]
    encoder = encoders_dict[args.bert_type](args, Config, Model)
    top = targets_dict[args.top_type](args)
    targets = [targets_dict[t.target](300, 2) for t in task_manager.tasklist]

    model = Net(args, encoder, top, targets)
    return model

