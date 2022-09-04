"""
wangsong2

"""
import time

class BaseTask:
    """
    定义了任务，里面应该包含什么
    test_nolabel是glue和clue里面的评测数据集，没标签的，所以我从训练集里划分了20%补充到测试集里面。
    """
    def __init__(self):
        self.task_type = None
        self.task_output = None
        self.name = None
        self.language = None
        self.label = None

        self.train = None
        self.test = None
        self.test_nolabel = None
        self.dev = None
        self.len_train = 0
        self.len_test = 0
        self.len_dev = 0
        self.len_test_nolabel = 0

    def get_dataset(self, data, datatype):
        if "train" in datatype:
            self.train = data
            self.len_train = len(data)
        elif "nolabel" in datatype:
            self.test_nolabel = data
            self.len_test_nolabel = len(data)
        elif "test" in datatype:
            self.test = data
            self.len_test = len(data)
        elif "dev" in datatype:
            self.dev = data
            self.len_dev = len(data)

    def print_task_info(self):
        print('-' * 50)
        print(f'name:{self.name} | type:{self.task_type} | language:{self.language}')
        print(f'train:{self.len_train} | dev:{self.len_dev} | test:{self.len_test} | test nolabel:{self.len_test_nolabel} | output:{self.task_output}')


class TaskManage:
    def __init__(self, args):
        self.args = args

        self.task_number = 0
        self.task_path = []
        self.tasklist = []
        self.tasklist_name = []

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

    def get_tasklist(self, path_or_list):
        if path_or_list.split('.')[-1] == "txt":
            with open(path_or_list, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.tasklist_name.append(line.strip())
        else:
            self.tasklist_name = path_or_list.split(' ')
        print(f'task list: {self.tasklist_name}')

    def load_data(self):
        """
        利用名字来获取数据集合,我真是个天才
        eg. jd21.修复霜, amz20.Baby
        通过自己实现数据读取方式，将任务数据和信息写入BaseTask
        :param name_list: 通过namelist来定义任务序列
        :return: List[task]
        """

        for name in self.tasklist_name:
            task = BaseTask()
            time_start = time.time()
            """
            根据每一个任务的名称选择对应的数据读取步骤
            在对应分支里实现自己的读取逻辑就可以，只要最后输出符合模型输入格式的数据就行（train，test，dev）
            """

            if name.split('.')[0] in ['jd21', 'stock', 'jd7k', 'amz20', 'snap10k']:
                from dataloader.load_dsc_dataset import loader
                task = loader(name, self.tokenizer, self.max_seq_length)

            elif 'clue' in name:
                from dataloader.load_clue_cls_dataset import loader
                task = loader(name, self.tokenizer, self.max_seq_length)

            task.print_task_info()
            print(f'load {task.len_train + task.len_test + task.len_dev} data in {time.time() - time_start:.2f}s.')

            self.tasklist.append(task)
            self.task_number += 1
