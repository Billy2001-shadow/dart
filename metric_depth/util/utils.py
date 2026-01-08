import os
import re
import numpy as np
import logging

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# 计算并输出模型参数量
def count_parameters(model):
    pretrained_params = sum(p.numel() for name, p in model.named_parameters() if 'pretrained' in name)
    head_params = sum(p.numel() for name, p in model.named_parameters() if 'depth_head' in name)
    total_params = pretrained_params + head_params
    return pretrained_params, head_params, total_params
