"""
主文件。入口
wangsong2 2022.8.31 (8月过得不错！）
"""

from config import set_args
from dataloader.task_manage import TaskManage

args = set_args()

task_manage = TaskManage(args)
task_manage.get_tasklist(args.task_list)
task_manage.load_data()
print('done')
