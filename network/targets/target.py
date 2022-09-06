import torch.nn as nn


class Target(nn.Module):
    def __init__(self):
        self.target_list = nn.ModuleList()
        self.target_name_list = []

    def update(self, target, target_name):
        self.target_list.append(target)
        self.target_name_list.append(target_name)

