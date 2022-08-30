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
        elif datatype == "dev":
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
        :param name_list: 通过namelist来定义任务序列
        :param tokenizer: 通常我们用一个模型进行实验对应该模型得tokenizer
        :return: List[task]
        """

        for name in name_list:
            task = BaseTask()
            """
            根据每一个任务的名称选择对应的数据读取步骤
            在对应分支里实现自己的读取逻辑就可以，只要最后输出符合模型输入格式的数据就行（train，test，dev）
            """

            if 'jd21' in name or 'stock' in name:
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

            elif 'your_dataset' in name:
                # 实现你自己的读取逻辑
                pass

            self.tasklist.append(task)
