from typing import List

import torch
from models._init__ import *


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


def build_models(args, task_manager):
    Tokenizer, Config, Model = encoders_args_dict[args.bert_type]
    encoder = encoders_dict[args.bert_type](args, Config, Model)
    top = targets_dict[args.top_type](args)


    targets = [targets_dict[t.target](300, 2) for t in task_manager.tasklist]

    model = Net(args, encoder, top, targets)
    return model

