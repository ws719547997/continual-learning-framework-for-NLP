import sys
import os
import json
import random
import numpy as np
import torch

from approaches.base.contrastive_loss import SupConLoss
from copy import deepcopy

class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model, taskcla,args=None):


        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.output_dir = f'res/{args.approach}'
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(
            args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)

        self.model=model
        self.args = args

        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        self.ce=torch.nn.CrossEntropyLoss()

        if 'one' in args.approach:
            self.initial_model=deepcopy(model)

        print('BERT NCL')

        return

    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets,t):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        loss = self.sup_con(outputs, targets,args=self.args)
        return loss


    def f1_compute_fn(self,y_true, y_pred,average):
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
