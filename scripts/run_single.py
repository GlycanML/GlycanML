import os
import sys
import math
import random
import pprint
import numpy as np
import shutil
from matplotlib import pyplot as plt

import torch
from torch.optim import lr_scheduler

from torchdrug import core, datasets, layers, models, tasks, transforms
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module import custom_data, custom_datasets, custom_models, custom_tasks, util


def lower_better(metric, metric_list=["root mean squared error", "mean absolute error"]):
    lower_better = False
    for _metric in metric_list:
        if metric.startswith(_metric):
            lower_better = True

    return lower_better


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return solver

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("inf") if lower_better(cfg.metric) else float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if lower_better(cfg.metric):
            if result < best_result:
                best_result = result
                best_epoch = solver.epoch
        else:
            if result > best_result:
                best_result = result
                best_epoch = solver.epoch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)
        torch.cuda.empty_cache()

    if comm.get_rank() == 0:
        shutil.move("model_epoch_%d.pth" % best_epoch, "best_valid_epoch_%d.pth" % best_epoch)
        for fname in os.listdir("./"):
            if fname.startswith("model_epoch_"):
                os.remove(fname)
    comm.synchronize()
    torch.cuda.empty_cache()
    solver.load("best_valid_epoch_%d.pth" % best_epoch, load_optimizer=False)

    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_solver(cfg, _dataset)

    solver = train_and_validate(cfg, solver, scheduler)
    test_metric = test(cfg, solver)
