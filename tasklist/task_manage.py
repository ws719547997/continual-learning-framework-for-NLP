"""
wangsong2

"""
import time
import json
from dataloader.load_clue_cls_dataset import clue_loader
from dataloader.load_dsc_dataset import dsc_loader
from models._init__ import encoders_args
from tasklist.BaseTask import BaseTask


class TaskManage:
    def __init__(self, args):
        self.args = args

        self.task_number = 0
        self.tasklist = []
        self.tasklist_args = []

        self.max_seq_length = args.max_seq_length
        self.tokenizer = self._set_tokenizer()

        self.get_tasklist(self.args.task_list)

    def _set_tokenizer(self):
        """
        目前任务中的样本有下列形式：
        1. [CLS] sent1 [SEP]
        2. [CLS] sent1 [SEP] sent2 [SEP]
        """
        print(f"chose {self.args.bert_type} as tokenizer.")
        tokenizer, _, _ = encoders_args[self.args.bert_type]
        return tokenizer.from_pretrained(self.args.bert_name)

    def get_tasklist(self, task_json):
        with open(task_json, 'r', encoding='utf-8') as f:
            j = json.load(f)
            for t in j['task_list']:
                self.tasklist_args.append(t)

    def load_data(self):
        """
        利用名字来获取数据集合,我真是个天才
        eg. jd21.修复霜, amz20.Baby
        通过自己实现数据读取方式，将任务数据和信息写入BaseTask
        :param name_list: 通过namelist来定义任务序列
        :return: List[task]
        """

        for taskargs in self.tasklist_args:
            task = BaseTask()
            time_start = time.time()
            """
            根据每一个任务的名称选择对应的数据读取步骤
            在对应分支里实现自己的读取逻辑就可以，只要最后输出符合模型输入格式的数据就行（train，test，dev）
            """
            name = taskargs['task_name']
            if name.split('.')[0] in ['jd21', 'stock', 'jd7k', 'amz20', 'snap10k']:
                task = dsc_loader(task, taskargs, self.tokenizer, self.max_seq_length)

            elif 'clue' in name:
                task = clue_loader(task, taskargs, self.tokenizer, self.max_seq_length)

            task.print_task_info()
            print(f'load {task.len_train + task.len_test + task.len_dev} data in {time.time() - time_start:.2f}s.')

            self.tasklist.append(task)
            self.task_number += 1
