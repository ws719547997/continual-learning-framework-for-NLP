"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from config import set_args
from task_builder import TaskManage
from utils import *
from method_builder import build_method
from log_bulider import Log

args = set_args()
# args.few_shot = True
# args.sgd_momentum = True
set_seeds(args.seed)
logger = Log(args)
logger.logger.info(f'Start exp.')
gpu_ranks = get_available_gpus(order='load', memoryFree=8000, limit=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(gpu_ranks[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.set_gpu_monitor(1000, gpu_ranks)

logger.logger.info(f'Load task, model and approach.....')
task_manage = TaskManage(args, logger)

Appr, Net = build_method(args)
model = Net(args, task_manage, logger)
logger.add_model_summary(model,
                         ((args.train_batch_size, args.max_seq_length), (args.train_batch_size, args.max_seq_length),
                          (args.train_batch_size, args.max_seq_length)),
                         dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'],
                         device='cpu')
model = model.to(device)
appr = Appr(model, args, device, logger)

logger.logger.info(f'Start training.....')
logger.add_metric('acc', (len(task_manage)+1, len(task_manage)))
# logger.add_metric('loss', (len(task_manage)+1, len(task_manage)))
logger.add_metric('f1', (len(task_manage)+1, len(task_manage)))

# 先在开始测试一下空白模型在每个任务上的性能
logger.logger.info(f'Test with init model...')
for test_id, test_task in enumerate(task_manage.tasklist):
    test = test_task.test_data
    test_sampler = SequentialSampler(test)
    test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

    test_loss, test_acc, test_f1 = appr.eval(test_id, test_dataloader)

    logger.set_metric('acc', (0, test_id), test_acc)
    # logger.set_metric('loss', (0, test_id), test_loss)
    logger.set_metric('f1', (0, test_id), test_f1)


for task_id, task in enumerate(task_manage.tasklist):
    # 为每个任务设定自己的参数
    task_manage.print_task_info(task)
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

    logger.add_gpu_point(f'training {task.name}')
    appr.train(args, task_id, train_dataloader, valid_dataloader, num_train_steps=num_train_steps, task=task)

    for test_id, test_task in enumerate(task_manage.tasklist):
        test = test_task.test_data
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        test_loss, test_acc, test_f1 = appr.eval(test_id, test_dataloader)

        logger.set_metric('acc', (task_id+1, test_id), test_acc)
        # logger.set_metric('loss', (task_id+1, test_id), test_loss)
        logger.set_metric('f1', (task_id+1, test_id), test_f1)
    logger.print_result()

logger.end()

