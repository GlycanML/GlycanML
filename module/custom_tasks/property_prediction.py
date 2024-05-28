import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm
from torchdrug.layers import functional

@R.register("tasks.GlycanPropertyPrediction")
@utils.copy_args(tasks.PropertyPrediction)
class GlycanPropertyPrediction(tasks.PropertyPrediction):
    """
    Glycan prediction task compatible with multi-task learning.

    Parameters:
        adjust: (str, optional): loss adjustment method for multi-task learning. Available methods are ``ts``,  ``uw``, 
            ``dwa``, ``dtp``, ``gn`` and ``norm``. Initial values of (hyper-)parameters should be specified inside square brackets.
        **kwargs
    """

    def __init__(self, adjust="default", **kwargs):
        super(GlycanPropertyPrediction, self).__init__(**kwargs)
        self.iteration = 0

        if adjust.find("[") != -1:
            adjust_val = eval(adjust[adjust.find("[")+1: adjust.find("]")])
            adjust = adjust[: adjust.find("[")].replace(" ", "")
        else:
            adjust_val = None
        self.adjust = adjust if len(self.task) > 1 else "default"
        self.adjust_val = adjust_val
        self.mtl_param_group1 = []      # same lr as the backbone network
        self.mtl_param_group2 = []      # exclusive lr

        if self.adjust == "uw" or self.adjust == "ts":
            self.logvar = nn.Parameter(torch.zeros(len(self.task)) + adjust_val)
            self.mtl_param_group1.append(self.logvar)
        elif self.adjust == "dwa":
            self.register_buffer("dwa_l1", torch.ones(len(self.task), dtype=torch.float))
            self.register_buffer("dwa_l2", torch.ones(len(self.task), dtype=torch.float))
        elif self.adjust == "dtp":
            # enforcing alpha=1
            self.gamma = self.adjust_val
            self.register_buffer("dtp_k", torch.zeros(len(self.task), dtype=torch.float) + 0.8)
        elif self.adjust == "gn":
            self.gn_alpha = self.adjust_val
            self.gn_w = nn.Parameter(torch.ones(len(self.task)))
            self.mtl_param_group2.append(self.gn_w)

    def forward(self, batch):
        self.iteration += 1
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    if self.adjust == "ts":
                        if self.iteration % 100 == 0:
                            print(self.logvar)
                        _pred = _pred * torch.exp(-self.logvar[i])
                        _loss = F.cross_entropy(_pred, target[:, i].long(), reduction="none").unsqueeze(-1)
                    elif self.adjust == "uw":
                        if self.iteration % 100 == 0:
                            print(self.logvar)
                        _loss = F.cross_entropy(_pred, target[:, i].long(), reduction="none").unsqueeze(-1)
                        _loss = _loss * torch.exp(-self.logvar[i]) + self.logvar[i] * 0.5
                    else:
                        _loss = F.cross_entropy(_pred, target[:, i].long(), reduction="none").unsqueeze(-1)
                    loss.append(_loss)
                    num_class += cur_num_class
                loss = torch.cat(loss, dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            if self.adjust == "norm":
                loss = loss / (loss.detach() + 1e-9)
            elif self.adjust == "dwa":
                self.dwa_l2 = self.dwa_l1
                self.dwa_l1 = loss.detach()
                if comm.get_world_size() > 1:
                    self.dwa_l1 = comm.reduce(self.dwa_l1, op="mean")
                if self.iteration > 2:
                    dwa_w = torch.exp(torch.clip((self.dwa_l1 / (self.dwa_l2 + 1e-9)) / self.adjust_val, max=50))
                    loss = loss * dwa_w / dwa_w.mean()
                if self.iteration % 100 == 0:
                    print(dwa_w)
            elif self.adjust == "dtp":
                dtp_w = -((1-self.dtp_k)**self.gamma) * torch.log(self.dtp_k)
                loss = loss * dtp_w / dtp_w.mean()
                if self.iteration % 100 == 0:
                    print(dtp_w)
            elif self.adjust == "gn":
                self.gn_w.data.div_(self.gn_w.detach().mean())
                gnorm = []
                for i in range(len(self.task)):
                    grads = torch.autograd.grad(loss[i], self.mlp.layers[-2].parameters(), retain_graph=True, create_graph=True)
                    gnorm.append(torch.norm(torch.cat([x.reshape(-1) for x in grads]) * self.gn_w[i]))
                gnorm = torch.stack(gnorm)   
                gnorm_avg = gnorm.detach().mean() 
                
                if not hasattr(self, "gn_l0"):
                    self.register_buffer("gn_l0", loss.detach())
                    if comm.get_world_size() > 1:
                        self.gn_l0 = comm.reduce(self.gn_l0, op="mean")
                l_t = (loss/self.gn_l0).detach()
                r_t = l_t / l_t.mean()
                grad_loss = F.l1_loss(gnorm, gnorm_avg * (r_t ** self.gn_alpha))
                self.gn_w.grad = torch.autograd.grad(grad_loss, self.gn_w)[0]
                
                loss = loss * (self.gn_w.detach())
                if self.iteration % 100 == 0:
                    print(self.gn_w)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric
    
    def evaluate(self, pred, target):
        metric = super(GlycanPropertyPrediction, self).evaluate(pred, target)
        for _metric in self.metric:
            name = tasks._get_metric_name(_metric)
            metric["%s [average]" % (name)] = sum([metric["%s [%s]" % (name, t)] for t in self.task])/len(self.task)
           
            if name == "accuracy" and self.adjust == "dtp":
                acc = torch.tensor([metric["%s [%s]" % (name, t)] for t in self.task], device=self.device)
                if comm.get_world_size() > 1:
                    acc = comm.reduce(acc, op="mean")
                self.dtp_k = acc

        return metric