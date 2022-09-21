import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, taskcla, args):
        super(TextCNN, self).__init__()

        self.taskcla = taskcla
        self.args = args

        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.WORD_DIM = args.bert_hidden_size

        self.relu = torch.nn.ReLU()

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)
        self.c2 = torch.nn.Conv1d(1, self.FILTER_NUM[1], self.WORD_DIM * self.FILTERS[1], stride=self.WORD_DIM)
        self.c3 = torch.nn.Conv1d(1, self.FILTER_NUM[2], self.WORD_DIM * self.FILTERS[2], stride=self.WORD_DIM)

        self.dropout = nn.Dropout(0.5)
        return

    def forward(self, sequence_output):

        h = sequence_output.view(-1, 1, self.WORD_DIM * self.args.max_seq_length)

        h1 = F.max_pool1d(F.relu(self.c1(h)), self.args.max_seq_length - self.FILTERS[0] + 1).view(-1,
                                                                                                   self.FILTER_NUM[0],
                                                                                                   1)
        h2 = F.max_pool1d(F.relu(self.c2(h)), self.args.max_seq_length - self.FILTERS[1] + 1).view(-1,
                                                                                                   self.FILTER_NUM[1],
                                                                                                   1)
        h3 = F.max_pool1d(F.relu(self.c3(h)), self.args.max_seq_length - self.FILTERS[2] + 1).view(-1,
                                                                                                   self.FILTER_NUM[2],
                                                                                                   1)

        h = torch.cat([h1, h2, h3], 1)
        h = h.view(sequence_output.size(0), -1)
        h = self.dropout(h)

        return h
