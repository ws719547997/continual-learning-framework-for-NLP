"""
wangsong2

"""
import time


class BaseTask:
    def __init__(self):
        self.task_type = None
        self.task_output = None
        self.name = None
        self.language = None
        self.label = None

        self.train = None
        self.test = None
        self.dev = None
        self.len_train = 0
        self.len_test = 0
        self.len_dev = 0

    def get_dataset(self, data, datatype):
        if "train" in datatype:
            self.train = data
            self.len_train = len(data)
        elif "test" in datatype:
            self.test = data
            self.len_test = len(data)
        elif "dev" in datatype:
            self.dev = data
            self.len_dev = len(data)

    def print_task_info(self):
        print('-'*50)
        print(f'name:{self.name} | type:{self.task_type} | language:{self.language}')
        print(f'train:{self.len_train} | dev:{self.len_dev} | test:{self.len_test} | output:{self.task_output}')


class TaskManage:
    def __init__(self, args):
        self.args = args

        self.task_number = 0
        self.task_path = []
        self.tasklist = []

        self.max_seq_length = args.max_seq_length
        self.tokenizer = self._set_tokenizer()

    def _set_tokenizer(self):
        """
        目前任务中的样本有下列形式：
        1. [CLS] sent1 [SEP]
        2. [CLS] sent1 [SEP] sent2 [SEP]
        """
        if "bert" in self.args.model_name:
            print(f"chose BertTokenizer as tokenizer.")
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained(self.args.model_name)

    def load_data_by_name(self, name_list):
        """
        利用名字来获取数据集合,我真是个天才
        eg. jd21.修复霜, amz20.Baby
        通过自己实现数据读取方式，将任务数据和信息写入BaseTask
        :param name_list: 通过namelist来定义任务序列
        :return: List[task]
        """

        for name in name_list:
            task = BaseTask()
            time_start = time.time()
            """
            根据每一个任务的名称选择对应的数据读取步骤
            在对应分支里实现自己的读取逻辑就可以，只要最后输出符合模型输入格式的数据就行（train，test，dev）
            """

            if 'jd21' in name or \
                    'stock' in name or \
                    'jd7k' in name:
                from dataloader.load_jd_format_data import data_loader as data_loader_jd
                [dataset, sub_dataset] = name.split('.')
                for data_type in ['train', 'test', 'dev']:
                    path = f'data/{dataset}/data/{data_type}/{sub_dataset}.txt'
                    task.get_dataset(data_loader_jd(path, self.tokenizer, self.max_seq_length),
                                     data_type)

                task.name = name
                task.task_type = "dsc"
                task.task_output = 2
                task.language = "zh"

            elif 'snap10k' in name or 'amz' in name:
                from dataloader.load_jd_format_data import data_loader as data_loader_jd
                [dataset, sub_dataset] = name.split('.')
                for data_type in ['train', 'test', 'dev']:
                    path = f'data/{dataset}/data/{data_type}/{sub_dataset}.txt'
                    task.get_dataset(data_loader_jd(path, self.tokenizer, self.max_seq_length),
                                     data_type)

                task.name = name
                task.task_type = "dsc"
                task.task_output = 2
                task.language = "en"

            elif 'clue' in name:
                [dataset, sub_dataset] = name.split('.')
                if sub_dataset in ['afqmc', 'cluewsc2020', 'cmnli', 'csl', 'tnews', 'iflytek']:
                    from dataloader.load_clue_cls_data import data_loader as data_loader_clue_cls
                    for data_type in ['train', 'test_nolabel', 'dev']:
                        path = f'data/{dataset}/{sub_dataset}/{data_type}.tsv'
                        task.get_dataset(data_loader_clue_cls(path, self.tokenizer, self.max_seq_length),
                                         data_type)
                    task.name = name
                    task.language = "zh"
                    if sub_dataset == 'afqmc':
                        task.task_output = 2
                        task.task_type = "matching"
                    elif sub_dataset == 'cluewsc2020':
                        task.task_output = 2
                        task.label = ['false', 'true']
                        task.task_type = 'co-reference resolution'
                    elif sub_dataset == 'cmnli':
                        task.task_output = 3
                        task.label = ['entailment', 'neutral', 'contradiction']
                        task.task_type = "nli"
                    elif sub_dataset == 'csl':
                        task.task_output = 2
                        task.task_type = "matching"
                    elif sub_dataset == 'tnews':
                        import json
                        with open("data/clue/tnews/labels.json", 'r', encoding='utf-8') as f:
                            task.label = [json.loads(i)['label_desc'] for i in f]
                        task.task_output = len(task.label)
                        task.task_type = 'cls'
                    elif sub_dataset == 'iflytek':
                        import json
                        with open("data/clue/iflytek/labels.json", 'r', encoding='utf-8') as f:
                            task.label = [json.loads(i)['label_des'] for i in f]
                        task.task_output = len(task.label)
                        task.task_type = 'cls'

            task.print_task_info()
            print(f'load {task.len_train+task.len_test+task.len_dev} data in {time.time() - time_start:.2f}s.')

            self.tasklist.append(task)
            self.task_number += 1
