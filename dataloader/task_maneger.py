"""
wangsong2

"""

import abc
import torch
from torch.utils.data import TensorDataset
from typing import List


class BaseTask:
    def __init__(self):
        self.task_type = None
        self.eval_method = None
        self.name = None

        self.train = None
        self.test = None
        self.eval = None
        self.language = None

    def get_dataset(self, data, datatype):
        if datatype == "train":
            self.train = data
        elif datatype == "test":
            self.test = data
        elif datatype == "eval":
            self.eval = data


class TaskList:
    def __init__(self, args):
        self.task_number = 0
        self.task_path = []
        self.tasklist = []
        self.max_seq_length = args.max_seq_length

    def load_data_by_name(self, name_list, tokenizer):
        """
        利用名字来获取数据集合,我真是个天才
        eg. jd21.修复霜, amz20.Baby
        通过自己实现数据读取方式，将任务数据和信息写入BaseTask
        :param name_list:
        :param tokenizer:
        :return:
        """

        for name in name_list:
            task = BaseTask()

            if 'jd21' in name:
                from load_jd_format_data import data_loader as data_loader_jd
                [dataset, sub_dataset] = name.split('.')
                for data_type in ['train', 'test', 'dev']:
                    path = f'data/{dataset}/data/{data_type}/{sub_dataset}.txt'
                    task.get_dataset(data_loader_jd(path, self.max_seq_length),
                                     data_type)

                task.name = dataset
                task.task_type = "dsc"
                task.eval_method = 2
                task.language = "zh"

            elif 'snap10k' in name or 'amz' in name:
                from load_jd_format_data import data_loader as data_loader_jd
                [dataset, sub_dataset] = name.split('.')
                for data_type in ['train', 'test', 'dev']:
                    path = f'data/{dataset}/data/{data_type}/{sub_dataset}.txt'
                    task.get_dataset(data_loader_jd(path, self.max_seq_length),
                                     data_type)

                task.name = dataset
                task.task_type = "dsc"
                task.eval_method = 2
                task.language = "en"

            self.tasklist.append(task)
