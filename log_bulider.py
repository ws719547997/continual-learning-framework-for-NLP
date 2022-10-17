import time
import os
import json
import logging
from shutil import copyfile
import numpy as np
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from utils import GPUMonitor


class Log:
    def __init__(self, args):
        self.exp_name = f'_{args.approach}_{args.task_list.split("/")[-1].split(".")[0]}_{args.bert_name.split("/")[-1]}_{args.comment}'
        self.date = time.strftime("%m%d%H%M", time.localtime())
        self.dir = args.output_dir + self.date + self.exp_name + '/'
        self.gpu_monitor = None
        self.metric = {}
        logging.basicConfig(filename='my.log', level=logging.INFO)
        print(os.getcwd())
        # 创建目录
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        # 把tasklist配置复制过来
        copyfile(args.task_list, self.dir + 'tasklist.json')
        # 把args的参数保存起来
        json.dump(args.__dict__, open(self.dir + 'args.json', 'w'), sort_keys=True, indent=2)
        # 创建Tensorboard
        self.writer = SummaryWriter(self.dir+'runs')
        # 创建logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(self.dir+'info.log', encoding='utf8'))
        self.logger.info(f'Log init. save at {self.dir}.')

    def set_gpu_monitor(self, interval, rank):
        self.logger.info(f'Log set_gpu_monitor. interval={interval}, rank={interval}.')
        self.gpu_monitor = GPUMonitor(interval, rank)

    def add_metric(self, name, shape, dtype=np.float32):
        self.logger.info(f'Log add_metric: {name}, shape:{shape}.')
        self.metric.update({name: np.zeros(shape, dtype=dtype)})

    def add_gpu_point(self, comment=None):
        self.gpu_monitor.add_log(task_status=comment)

    def set_metric(self, name, pos, value):
        (curr, test) = pos
        self.metric[name][curr][test] = value

    def add_model_summary(self, model, input_shape, dtypes, device='cpu'):
        with open(self.dir + 'model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(str(summary(model, input_shape, dtypes=dtypes, device=device)))

    def get_bwt(self, metric):
        pass

    def get_fwt(self, metric):
        pass

    def end(self):
        self.gpu_monitor.stop(f'{self.dir}/gpu_status.json')
        for k, v in self.metric.items():
            np.savetxt(self.dir+k+'.txt', v*100, '%.3f', delimiter='\t')
        self.writer.close()
        self.logger.info(f'All done!')
