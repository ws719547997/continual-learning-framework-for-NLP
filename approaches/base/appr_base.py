import torch


class Appr(object):
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_min = args.lr_min
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience
        self.clipgrad = args.clipgrad
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.optimizer = self._get_optimizer()
        self.ce = torch.nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_args(self, args):
        self.args = args
        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_min = args.lr_min
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience
        self.clipgrad = args.clipgrad
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.optimizer = self._get_optimizer()

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.args.optimizer == 'sgd' and self.args.sgd_momentum:
            print('sgd+momentum')
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9,nesterov=True)
        elif self.args.optimizer == 'sgd':
            print('sgd')
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        elif self.args.optimizer == 'adam':
            print('adam')
            return torch.optim.Adam(self.model.parameters(),lr=lr)

    def f1_compute_fn(self,y_true, y_pred, average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        if self.args.f1_macro:
            return f1_score(y_true, y_pred, average=average)
        else:
            return f1_score(y_true, y_pred, pos_label=0)