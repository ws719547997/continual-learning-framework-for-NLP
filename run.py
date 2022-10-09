"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from config import set_args
from task.task_manage import TaskManage
from torchinfo import summary
from utils import *
from method_builder import build_method
from log_bulider import Log

print('0. init.....')
args = set_args()
# args.few_shot = True
set_seeds(args.seed)
logger = Log(args)
gpu_ranks = get_available_gpus(order='load', memoryFree=8000, limit=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(gpu_ranks[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.set_gpu_monitor(200, gpu_ranks)

print('1. Load task, model and approach.....')
task_manage = TaskManage(args)

Appr, Net = build_method(args)
model = Net(args, task_manage, logger)
logger.add_model_summary(model, ((32, 128), (32, 128), (32, 128)),
        dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'],
        device='cpu')
model = model.to(device)
appr = Appr(model, args, device, logger)

print('2. Start training.....')
logger.add_metric('acc', (len(task_manage), len(task_manage)))
logger.add_metric('loss', (len(task_manage), len(task_manage)))
logger.add_metric('f1', (len(task_manage), len(task_manage)))

for task_id, task in enumerate(task_manage.tasklist):
    # 为每个任务设定自己的参数
    args = task_manage.set_task_args(args, task)

    if args.mutli_task:
        # Get data. We do not put it to GPU
        if task_id == 0:
            train = task.train_data
            valid = task.dev_data
            num_train_steps = int(math.ceil(task.len_train / args.train_batch_size)) * args.epochs
        else:
            train = ConcatDataset([train, task.train_data])
            valid = ConcatDataset([valid, task.dev_data])
            num_train_steps += int(math.ceil(task.len_train / args.train_batch_size)) * args.epochs
        if task_id < len(task_manage) - 1:
            continue  # only want the last one

    elif args.few_shot and task.json_args.get('train_samples') is not None:
        train = TensorDataset(*task.train_data[:task.json_args.get('train_samples')])
        valid = task.dev_data
        num_train_steps = \
            int(math.ceil(task.json_args.get('train_samples') / args.train_batch_size)) * args.epochs
    else:
        train = task.train_data
        valid = task.dev_data
        num_train_steps = int(math.ceil(len(task.train_data) / args.train_batch_size)) * args.epochs

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    logger.add_gpu_point('training')
    appr.train(args, task_id, train_dataloader, valid_dataloader, num_train_steps=num_train_steps, train_data=train,
               valid_data=valid)

    logger.add_gpu_point('testing')
    for test_id, test_task in enumerate(task_manage.tasklist):
        test = test_task.test_data
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        test_loss, test_acc, test_f1 = appr.eval(test_id, test_dataloader)

        logger.set_metric('acc', (task_id, test_id), test_acc)
        logger.set_metric('loss', (task_id, test_id), test_loss)
        logger.set_metric('f1', (task_id, test_id), test_f1)

print('4. Logging')
logger.end()
print('done')
