class BaseTask:
    """
    定义了任务，里面应该包含什么
    test_nolabel是glue和clue里面的评测数据集，没标签的，所以我从训练集里划分了20%补充到测试集里面。
    """
    def __init__(self, args, json_args=None):
        self.json_args = json_args
        self.args = args
        self.task_type = None
        self.task_output = None
        self.name = None
        self.language = None
        self.label = None
        self.target = None

        self.train_data = ''
        self.test_data = ''
        self.test_nolabel_data = ''
        self.dev_data = ''

    def get_dataset(self, data, datatype):
        if "train" in datatype:
            self.train_data = data
        elif "nolabel" in datatype:
            self.test_nolabel_data = data
        elif "test" in datatype:
            self.test_data = data
        elif "dev" in datatype:
            self.dev_data = data