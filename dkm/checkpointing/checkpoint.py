import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger


class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

    def __call__(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
    ):
        assert model is not None
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        states = model.state_dict()
        state_dict_path = os.path.join(self.dir, f"latest_epoch.pth")
        torch.save(states, state_dict_path)
        logger.info(f"Saved states to %s, at step %d" % (state_dict_path, n))
