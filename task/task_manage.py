"""
wangsong2

"""
import time
import json
from typing import List

from dataloader.load_clue_cls_dataset import clue_loader
from dataloader.load_dsc_dataset import dsc_loader
from models.models_builder import encoders_args_dict
from task.BaseTask import BaseTask


class TaskManage:
    def __init__(self, args):
        self.args = args
        self.tasklist: List[BaseTask] = []
        self.tokenizer = self._set_tokenizer()

        self.get_task_args()
        self.build_task()

    def get_task_args(self):
        with open(self.args.task_list, 'r', encoding='utf-8') as f:
            j = json.load(f)
            for tj in j['task_list']:
                self.tasklist.append(BaseTask(self.args, tj))

    def set_task_args(self, args, task: BaseTask):
        for k, v in task.json_args.items():
            args.__dict__[k] = v
        self.args = args
        return args

    def _set_tokenizer(self):
        """
        目前任务中的样本有下列形式：
        1. [CLS] sent1 [SEP]
        2. [CLS] sent1 [SEP] sent2 [SEP]
        """
        print(f"Chose {self.args.bert_type} as tokenizer.")
        tokenizer, _, _ = encoders_args_dict[self.args.bert_type]
        return tokenizer.from_pretrained(self.args.bert_name)

    def build_task(self):
        """
        利用名字来获取数据集合,我真是个天才
        eg. jd21.修复霜, amz20.Baby
        通过自己实现数据读取方式，将任务数据和信息写入BaseTask
        :param name_list: 通过namelist来定义任务序列
        :return: List[task]
        """

        for task in self.tasklist:
            time_start = time.time()
            """
            根据每一个任务的名称选择对应的数据读取步骤
            在对应分支里实现自己的读取逻辑就可以，只要最后输出符合模型输入格式的数据就行（train，test，dev）
            """
            name = task.json_args['task_name']
            if name.split('.')[0] in ['jd21', 'stock', 'jd7k', 'amz20', 'snap10k']:
                task = dsc_loader(task, self.tokenizer)

            elif 'clue' in name:
                task = clue_loader(task, self.tokenizer)

            self.print_task_info(task)
            print(f'Load all data in {time.time() - time_start:.2f}s.')

    def build_args(self):
        for j in self.tasklist_args:
            epochs = j['epochs'] if j.get('epochs') else self.args.epochs
            lr = j['lr'] if j.get('lr') else self.args.lr
            optimizer = j['optimizer'] if j.get('optimizer') else self.args.optimizer

            self.argslist.append({
                'epochs': epochs,
                'lr': lr,
                'optimizer': optimizer
            })

    def __len__(self):
        return len(self.tasklist)

    def print_task_info(self, task: BaseTask):
        print('-' * 70)
        print(f'{self.tasklist.index(task)}. name:{task.name} | type:{task.task_type} | language:{task.language}')
        print(
            f'train:{len(task.train_data)} | dev:{len(task.dev_data)} | test:{len(task.test_data)} | test nolabel:{len(task.test_nolabel_data)} | output:{task.task_output}')
