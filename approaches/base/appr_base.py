import torch
from log_bulider import Log


class Appr(object):
    def __init__(self, model, args, device, logger):
        self.model = model
        self.logger: Log = logger
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

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        if self.args.optimizer == 'sgd' and self.args.sgd_momentum:
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def f1_compute_fn(self, y_true, y_pred, average):
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

    def train_epoch(self, t, data, iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets = batch
            # Forward
            outputs = self.model.forward(input_ids, segment_ids, input_mask, t)
            output = outputs[t]

            loss = self.ce(output, targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, t, data, test=None, trained_task=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets = batch
                real_b = input_ids.size(0)

                outputs = self.model.forward(input_ids, segment_ids, input_mask, t)

                output = outputs[t]

                loss = self.ce(output, targets)

                _, pred = output.max(1)
                hits = (pred == targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss += loss.data.cpu().numpy().item() * real_b
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += real_b

            f1 = self.f1_compute_fn(y_pred=torch.cat(pred_list, 0), y_true=torch.cat(target_list, 0), average='macro')

        return total_loss / total_num, total_acc / total_num, f1

    def print_info(self):
        self.logger.logger.info(
            f'total_epoch:{self.args.epochs} | optimizer:{self.args.optimizer} | lr:{self.args.lr}. \n'
            f'target:{self.args.target} | fewshot:{self.args.few_shot} | mutli-task:{self.args.mutli_task}.')
