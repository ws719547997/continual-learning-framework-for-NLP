"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
import numpy as np
import torch

from config import set_args
from task_manage import TaskManage

from utils import *

args = set_args()
set_seeds(args.seed)
# Seed

gpu_ranks = get_available_gpus(order='load', memoryFree=8000, limit=1)
gpu_monitor = GPUMonitor(150, gpu_ranks)

task_manage = TaskManage(args)
task_manage.get_tasklist(args.task_list)
with timer('Load task list'):
    task_manage.load_data()

print('Inits...')





print('done')
