"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from config import set_args
from models.models_builder import build_models
from approaches.approches_builder import build_approaches
from task_manage import TaskManage
from torchinfo import summary
from utils import *

print('0. init.....')
args = set_args()
set_seeds(args.seed)
gpu_ranks = get_available_gpus(order='load', memoryFree=8000, limit=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(gpu_ranks[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_monitor = GPUMonitor(150, gpu_ranks)

print('1. Load task, model and approach.....')
task_manage = TaskManage(args)
with timer('1.1 Load task list'):
    task_manage.build_task()

model = build_models(task_manage, args)
summary(model,((32, 128), (32, 128), (32, 128)),
        dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'],
        device='cpu')

model = model.to(device)
appr = build_approaches(model, task_manage, args, device)

print('2. Start training.....')
acc = np.zeros((len(task_manage), len(task_manage)), dtype=np.float32)
lss = np.zeros((len(task_manage), len(task_manage)), dtype=np.float32)
f1 = np.zeros((len(task_manage), len(task_manage)), dtype=np.float32)

for task_id, task in enumerate(task_manage.tasklist):
    if 'mtl' in args.approach:
        # Get data. We do not put it to GPU
        if task_id == 0:
            train = task.train
            valid = task.dev
            num_train_steps = int(math.ceil(task.len_train / args.train_batch_size)) * args.epochs
        else:
            train = ConcatDataset([train, task.train])
            valid = ConcatDataset([valid, task.dev])
            num_train_steps += int(math.ceil(task.len_train / args.train_batch_size)) * args.epochs
        if task_id < len(task_manage) - 1: continue  # only want the last one

    else:
        train = task.train
        valid = task.dev
        num_train_steps = int(math.ceil(task.len_train / args.train_batch_size)) * args.epochs

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    appr.train(task_id, train_dataloader, valid_dataloader, num_train_steps=num_train_steps, train_data=train,
               valid_data=valid)

    for test_id, test_task in enumerate(task_manage.tasklist):
        test = test_task.test
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        test_loss, test_acc, test_f1 = appr.eval(test_id, test_dataloader)

        acc[task_id, test_id] = test_acc
        lss[task_id, test_id] = test_loss
        f1[task_id, test_id] = test_f1

print('4. Logging')

print('done')
