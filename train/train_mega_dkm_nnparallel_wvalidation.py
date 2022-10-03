from __future__ import print_function, division
import os, sys, inspect
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import numpy as np
import PIL.Image as Image
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dkm.train.train import train_k_steps
from dkm.checkpointing.checkpoint import CheckPoint
from dkm import DKMv2
from dkm.datasets.megadepth import MegadepthBuilder
from dkm.losses import DepthRegressionLoss
from dkm.benchmarks.validate_scannet import Validation

def set_frozen_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

def run(gpus=1, args=None):
    # Initialize parallel computing
    experiment_name = args.experiment_name
    checkpoint_dir = os.path.join(project_root, 'checkpoints', args.experiment_name)
    h, w = 384, 512
    model = DKMv2(
        pretrained=args.pretrained, version="outdoor"
    )

    # Load ckpt
    if args.restore_ckpt:
        state_dict = torch.load(args.restore_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("Successfully Load from %s" % args.restore_ckpt)

    # Num steps
    n0 = 0
    batch_size = args.batch_size
    N = 250000  # 250k steps of batch size 32
    # checkpoint every
    k = 10000

    # Data
    mega = MegadepthBuilder(data_root=args.data_root)
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, ht=h, wt=w, shake_t=32
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    # Loss and optimizer
    depth_loss = DepthRegressionLoss(ce_weight=0.01)

    parameters = [
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.decoder.parameters(), "lr": 1e-6},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[N // 3, (2 * N) // 3], gamma=0.2
    )
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)

    dp_model = nn.DataParallel(model)

    # Set Misc
    writer = SummaryWriter(checkpoint_dir, flush_secs=30)
    # Set Validation
    validator = Validation(project_root=project_root)

    if args.eval_only:
        validator.apply_eval(model=dp_model, writer=writer, steps=0, args=args, save=True)
        torch.cuda.synchronize()
        return

    # Train
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples=batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size=batch_size,
                sampler=mega_sampler,
                num_workers=args.num_workers
            )
        )
        train_k_steps(
            n, k, mega_dataloader, dp_model, depth_loss, optimizer, lr_scheduler, writer=writer
        )
        validator.apply_eval(model=dp_model, writer=writer, steps=n, args=args, save=True)
        checkpointer(model, optimizer, lr_scheduler, n)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--gpus", default=3, type=int)
    parser.add_argument("--data_root", default="data/megadepth", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--restore_ckpt", type=str)
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument('--relfrmin_eval', type=int, nargs="+", required=True)
    parser.add_argument('--downscale', action="store_true")
    parser.add_argument('--scannetroot', type=str)

    args, _ = parser.parse_known_args()
    os.makedirs(os.path.join(project_root, 'checkpoints', args.experiment_name), exist_ok=True)
    run(gpus=args.gpus, args=args)