import json
from contextlib import contextmanager
import random
import os
import numpy as np
import torch
import GPUtil
from threading import Thread, Lock
import time
from torchinfo import summary


def set_seeds(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f'{name} during {time.time() - start:.2f} s')


def get_available_gpus(order='first', limit=1, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[],
                       excludeUUID=[]):
    """
    :param order:first,last,random,load,memory
    :param limit:返回GPU的数量
    :param maxLoad:最大负载比（0~1）
    :param maxMemory:最大内存占用（0~1）
    :param memoryFree:空闲内存
    :param includeNan:GPU负载和内存占用为nan的情况
    :param excludeID:需要排除的gpu id
    :param excludeUUID:需要排除的gpu uuid
    :return:List of available gpu ids.
    """
    return GPUtil.getAvailable(order=order, limit=limit, maxLoad=maxLoad, maxMemory=maxMemory, memoryFree=memoryFree,
                               includeNan=includeNan, excludeID=excludeID, excludeUUID=excludeUUID)


class GPUMonitor(Thread):
    def __init__(self, delay, gpu_ids, task_status=None):
        super(GPUMonitor, self).__init__()
        self.ids = gpu_ids
        self.gpu_status = GPUstatus(self.ids, task_status)
        self._stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self._value_lock = Lock()
        self.start()

    def run(self):
        while not self._stopped:
            # see detail at https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py
            self._value_lock.acquire()
            self.gpu_status.add_gpu_status()
            self._value_lock.release()
            time.sleep(self.delay)

    def add_log(self, task_status=None):
        self._value_lock.acquire()
        self.gpu_status.add_gpu_status(task_status)
        self._value_lock.release()

    def stop(self, path):
        self._value_lock.acquire()
        self._stopped = True
        self._value_lock.release()
        with open(path, 'w') as fp:
            fp.write(json.dumps(self.gpu_status.status))


class GPUstatus:
    def __init__(self, gpu_ids, task_status='init'):
        self._start_time = time.time()
        self.status = {}
        self.task_status = task_status
        self.ids = gpu_ids

        gpus = GPUtil.getGPUs()
        for gpu_id in self.ids:
            self.status.update({f'GPU {gpu_id}':
                                    {'memory': [],
                                     'load': [],
                                     'status': [],
                                     'time': [],
                                     'name': gpus[gpu_id].name,
                                     'memoryTotal': gpus[gpu_id].memoryTotal}
                                })

    def add_gpu_status(self, task_status=None):

        time_step = time.time() - self._start_time
        gpus = GPUtil.getGPUs()
        if task_status:
            self.task_status = task_status
        for gpu_id in self.ids:
            self.status[f'GPU {gpu_id}']['time'].append(time_step)
            self.status[f'GPU {gpu_id}']['memory'].append(gpus[gpu_id].memoryUsed)
            self.status[f'GPU {gpu_id}']['load'].append(gpus[gpu_id].load)
            self.status[f'GPU {gpu_id}']['status'].append(self.task_status)


def get_model_summary(model, input_size,):
    return summary(model, input_size)


if __name__ == '__main__':
    output_path = 'output/gpu_status.json'
    ids = get_available_gpus()
    gpu_monitor = GPUMonitor(5, ids, 'task1')
    time.sleep(10)
    gpu_monitor.add_log('stage1')
    time.sleep(10)
    gpu_monitor.add_log('stage2')
    time.sleep(10)
    gpu_monitor.stop(output_path)
