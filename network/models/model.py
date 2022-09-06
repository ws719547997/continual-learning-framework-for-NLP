import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of two (five) parts for now:

        - encoders


        - targets
    """

    def __init__(self, args, encoder, target):
        super(Model, self).__init__()
        self.encoder = encoder
        self.target = target
        self.args = args

    def forward(self, t, input_ids, segment_ids, input_mask):
        encoder_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        output = self.top[t](encoder_output)
        return output
