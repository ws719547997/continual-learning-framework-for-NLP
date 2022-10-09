import time


class Log:
    def __init__(self, args):
        self.exp_name = f'_{args.approach}_B{args.train_batch_size}_E{args.epochs}_{args.comment}'
        self.date = time.strftime("%m%d_%H:%M", time.localtime())
        self.dir = args.output_dir + self.date + self.exp_name
        self.gpu = 0


