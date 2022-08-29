"""
wangsong2

"""

import abc


class BaseTask(metaclass=abc.ABCMeta):
    def __init__(self):
        self.task_type = None
        self.eval_method = None
        self.data = None

    @abc.abstractmethod
    def get_task_info(self):
        pass

    @abc.abstractmethod
    def get_task_length(self):
        pass

    @abc.abstractmethod
    def get_line(self):
        pass

    @abc.abstractmethod
    def load(self, **kwargs):
        pass


class DataSet(metaclass=abc.ABCMeta):
    def __init__(self):
        self.task_type = None
        self.eval_method = None
        self.task = None


