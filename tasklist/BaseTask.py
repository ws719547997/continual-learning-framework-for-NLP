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
        self.target = None

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
