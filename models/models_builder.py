from typing import List

import torch
from models._init__ import *
from models.encoder.automodel_fixed import AutoPTM


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
        for i in range(t):
            y.append(self.target[i](h))
        return y


def build_models(args, targets_name_list: List):
    Tokenizer, Config, Model = encoders_dict[args.bert_type]
    encoder = AutoPTM(args, Config, Model)
    top = targets_dict['textcnn'](args)
    targets = [targets_dict[t](300,2) for t in targets_name_list]

    model = Net(args, encoder, top, targets)
    return model
