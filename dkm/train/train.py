import copy
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tqdm import tqdm
from dkm.utils.flow_viz import flow_to_image
from dkm.utils.utils import tensor2rgb, tensor2disp
from dkm.utils.utils import to_cuda, InputPadder

def train_step(train_batch, model, objective, optimizer, **kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)
    l.backward()
    optimizer.step()
    return {"train_out": out, "train_loss": l.item()}

def postprocess(batch, outputs):
    support_points = outputs['train_out'][1]['dense_flow']

    b, c, h, w = batch['query'].shape
    device = batch['query'].device
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    xx = torch.from_numpy(xx).float().to(device).view([1, 1, h, w])
    yy = torch.from_numpy(yy).float().to(device).view([1, 1, h, w])
    coord = torch.cat([xx, yy], dim=1)

    support_points = (support_points + 1) / 2
    support_points_x, support_points_y = torch.split(support_points, 1, dim=1)
    support_points_x = support_points_x * w - 0.5
    support_points_y = support_points_y * h - 0.5
    flow_prediciton = torch.cat([support_points_x, support_points_y], dim=1) - coord
    return flow_prediciton

def set_frozen_bn_trainmode(model):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, progress_bar=True, writer=None
):
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        outputs = train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n=n,
        )
        lr_scheduler.step()

        if writer is not None:
            writer.add_scalar('loss', outputs["train_loss"], n)
            writer.add_scalar('lr', lr_scheduler.optimizer.param_groups[0]['lr'], n)

            if np.mod(n, 1000) == 0:
                mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).view([1, 3, 1, 1]).to(batch['query'].device)
                std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).view([1, 3, 1, 1]).to(batch['query'].device)

                reconstructed_rgb = F.grid_sample(batch['support'], outputs['train_out'][1]['dense_flow'].permute([0, 2, 3, 1]), mode='bilinear', align_corners=False)

                flow_prediciton = postprocess(batch, outputs)
                flow_visualization = flow_prediciton.detach().permute([0, 2, 3, 1])[0].cpu().numpy()

                visualization1 = tensor2rgb((batch['query'] * std + mean) * 255.0,  viewind=0)
                visualization2 = tensor2rgb((batch['support'] * std + mean) * 255.0, viewind=0)
                visualization3 = tensor2rgb((reconstructed_rgb * std + mean) * 255.0, viewind=0)
                visualization5 = flow_to_image(flow_visualization)

                if 'query_depth' in batch:
                    query_depth_visualization = copy.deepcopy(batch['query_depth'])
                    query_depth_visualization[query_depth_visualization == 0] = np.inf
                    visualization4 = tensor2disp(1 / query_depth_visualization.unsqueeze(1), viewind=0, vmax=1.0)

                    visualization = np.concatenate([np.array(visualization1),
                                                    np.array(visualization2),
                                                    np.array(visualization3),
                                                    np.array(visualization4),
                                                    np.array(visualization5)], axis=0)
                else:
                    visualization = np.concatenate([np.array(visualization1),
                                                    np.array(visualization2),
                                                    np.array(visualization3),
                                                    np.array(visualization5)], axis=0)

                writer.add_image('visualization', (torch.from_numpy(visualization).float() / 255).permute([2, 0, 1]), n)

def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
