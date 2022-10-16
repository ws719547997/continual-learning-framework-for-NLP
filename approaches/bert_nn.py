import sys, time
import numpy as np
import torch

from models import models_utils
from tqdm import tqdm
from approaches.base.appr_base import Appr as ApprBase


class Appr(ApprBase):
    def __init__(self, model, args, device, logger):
        super(Appr, self).__init__(model=model, args=args, device=device, logger=logger)
        return

    def train(self, args, t, train, valid, num_train_steps=None, task=None):
        self.set_args(args)
        best_loss = np.inf

        # Loop epochs
        for e in range(self.epochs):
            # Train
            clock0 = time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            self.train_epoch(t, train, iter_bar)
            clock1 = time.time()
            train_loss, train_acc, train_f1_macro = self.eval(t, train)
            clock2 = time.time()
            print(f'Epoch {e + 1}, '
                  f'time {self.train_batch_size * (clock1 - clock0) / len(train):4.2f}s / '
                  f'{1000 * self.train_batch_size * (clock2 - clock1) / len(train):4.2f}s | '
                  f'Train: loss={train_loss:.3f}, acc={100*train_acc:5.1f}',
                  end='')
            # Valid
            valid_loss, valid_acc, valid_f1_macro = self.eval(t, valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            self.logger.writer.add_scalar(f'{task.name}/loss', train_loss, e)
            self.logger.writer.add_scalar(f'{task.name}/acc', train_acc * 100, e)
            self.logger.writer.add_scalar(f'{task.name}/loss', valid_loss, e)
            self.logger.writer.add_scalar(f'{task.name}/acc', valid_acc * 100, e)

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = models_utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    self.lr /= self.lr_factor
                    print(' lr={:.1e}'.format(self.lr), end='')
                    if self.lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(self.lr)
            print()

        # Restore best
        models_utils.set_model_(self.model, best_model)
        return

