# coding: utf-8
import sys
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):

    def __init__(self, args, config, model):
        super(Encoder, self).__init__()
        config = config.from_pretrained(args.bert_name)
        config.return_dict = True
        self.args = args
        self.bert = model.from_pretrained(args.bert_name, config=config)

        for param in self.bert.parameters():
            param.requires_grad = args.train_bert

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        return

    def forward(self, input_ids, segment_ids, input_mask):
        bert_dict = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        return bert_dict
