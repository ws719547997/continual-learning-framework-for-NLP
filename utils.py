from contextlib import contextmanager
import random
import os
import numpy as np
import torch
import GPUtil
from threading import Thread
import time


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
    def __init__(self, delay, gpu_ids):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()
        self.status = {}
        self.ids = gpu_ids
        self.start_time = time.time()
        gpus = GPUtil.getGPUs()

        for gpu_id in self.ids:
            self.status.update({f'gpu:{gpu_id}':
                                    {'memory': [],
                                     'load': [],
                                     'name':gpus[gpu_id].name,
                                     'memoryTotal':gpus[gpu_id].memoryTotal}
                                })
            self.status[f'gpu:{gpu_id}']['memory'].append(gpus[gpu_id].menoryUsed)
            self.status[f'gpu:{gpu_id}']['load'].append(gpus[gpu_id].load)

    def run(self):
        while not self.stopped:
            # see detail at https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py
            time.sleep(self.delay)
            gpus = GPUtil.getGPUs()
            for gpu_id in self.ids:
                self.status[f'gpu:{gpu_id}']['memory'].append(gpus[gpu_id].menoryUsed)
                self.status[f'gpu:{gpu_id}']['load'].append(gpus[gpu_id].load)

    def stop(self):
        self.stopped = True
        self.status.update({'time':f"{time.time()-self.start_time:.1f}"})
        return self.status

if __name__ =='__main__':
    ids = get_available_gpus()
    gpu_monitor = GPUMonitor(2,ids)
    time.sleep(20)
    print(gpu_monitor.stop())
