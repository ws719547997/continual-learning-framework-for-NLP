from torch import nn
from model_builder import *
from models.encoder.base_encoder import Encoder


class Net(nn.Module):
    def __init__(self, args, task_manage, logger):
        super(Net, self).__init__()
        self.args = args

        tokenizer, config, model = build_encoder(self.args)
        self.encoder = Encoder(self.args, config, model)

        top = build_top(self.args)
        self.top = top(self.args)

        self.targets = nn.ModuleList()
        for task in task_manage.tasklist:
            target = build_target(args, task)
            # 输入是textcnn的维度，300，输出是几分类，根据任务决定，这个信息必须写在json文件里
            self.targets.append(target(300, task.json_args['task_output']))

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids, segment_ids, input_mask, t=1):
        bert_dict = self.encoder(input_ids, segment_ids, input_mask)
        h = bert_dict.last_hidden_state
        h = self.top(h)
        y = []
        for target in self.targets:
            y.append(target(h))
        return y
