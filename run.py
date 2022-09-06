"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
import numpy as np
import torch

from config import set_args
from task_manage import TaskManage

from utils import timer

args = set_args()

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); exit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

task_manage = TaskManage(args)
task_manage.get_tasklist(args.task_list)
with timer('Load task list'):
    task_manage.load_data()

print('Inits...')





print('done')
