"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""

from config import set_args
from dataloader.task_manage import TaskManage

args = set_args()
task_list = args.task_list.split(' ')

task_manage = TaskManage(args)
task_manage.load_data_by_name(task_list)
print('done')
