# coding: utf-8
import sys
import torch
from torch import nn
import torch.nn.functional as F


class AutoPTM(torch.nn.Module):

    def __init__(self, args, BertConfig, BertModel):
        super(AutoPTM, self).__init__()
        config = BertConfig.from_pretrained(args.bert_name)
        config.return_dict = False
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_name, config=config)

        for param in self.bert.parameters():
            param.requires_grad = args.train_bert

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        return

    def forward(self, input_ids, segment_ids, input_mask):
        # TODO：把参数放入字典
        output_dict = {}

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        output_dict['sequence_output'] = self.dropout(sequence_output)

        return sequence_output
