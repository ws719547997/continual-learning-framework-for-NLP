"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
from torch.utils.tensorboard import SummaryWriter

from models.models_builder import build_models
from config import set_args
from tasklist.task_manage import TaskManage
from utils import *

args = set_args()
set_seeds(args.seed)
gpu_ranks = get_available_gpus(order='load', memoryFree=8000, limit=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(gpu_ranks[0])
# gpu_monitor = GPUMonitor(150, gpu_ranks)

task_manage = TaskManage(args)
with timer('Load task list'):
    task_manage.load_data()

model = build_models(args, task_manage)

summary(model,
        ((32, 128), (32, 128), (32, 128)),
        dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'],
        device='cpu')

print('done')
