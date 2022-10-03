import sys, os
sys.path.append('core')
import copy
import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from dkm.datasets.scannet import Scannet
from dkm.utils.utils import DistributedSamplerNoEvenlyDivisible
from dkm.utils.utils import get_connection_graph, to_cuda_scannet, stack_imgs, preprocess, postprocess

@torch.no_grad()
def validate_scannet(model, args, steps, writer=None):
    """ In Validation, random sample N Images to mimic a real test set situation  """
    model.eval()
    relfrmin, orgfrm, suppconnections = get_connection_graph(args.relfrmin_eval)
    val_dataset = Scannet(root=args.scannetroot, relfrmin=args.relfrmin_eval)
    val_loader = DataLoader(val_dataset, batch_size=args.gpus, pin_memory=False, shuffle=False, num_workers=2, drop_last=False)
    measurements_val = dict()

    for i in relfrmin:
        if i != orgfrm:
            frmgap = int(np.abs(i - orgfrm))
            if frmgap not in measurements_val:
                measurements_val[frmgap] = torch.zeros(5)

    width = 640
    maxepe = int(0.0125 * width)
    finest_scale = 1

    for val_id, data_blob in enumerate(tqdm.tqdm(val_loader)):

        data_blob = copy.deepcopy(data_blob)
        data_blob = to_cuda_scannet(data_blob)

        cbz = data_blob['rgbs'][list(data_blob['rgbs'].keys())[0]].shape[0]

        if args.downscale:
            for key in data_blob['rgbs'].keys():
                images = data_blob['rgbs'][key]
                images = F.interpolate(images, (int(images.shape[2] / 2), int(images.shape[3] / 2)), mode='bilinear', align_corners=True)
                data_blob['rgbs'][key] = images

        data_blob = preprocess(data_blob, relfrmin=relfrmin, orgfrm=orgfrm)

        flow_predictions = model(data_blob['batch'])
        _flow_predictions = postprocess(data_blob, flow_predictions, relfrmin=relfrmin, orgfrm=orgfrm)
        flow_pr = _flow_predictions[finest_scale]['dense_flow']

        if args.downscale:
            flow_pr = F.interpolate(flow_pr, (int(flow_pr.shape[2] * 2), int(flow_pr.shape[3] * 2)), mode='bilinear', align_corners=True) * 2

        flow_pr = flow_pr.view(cbz, -1, *flow_pr.shape[1::])

        cnt = 0
        for i in relfrmin:
            if i != orgfrm:
                flow_pr_ = flow_pr[:, cnt, :, :, :]
                flowgt, valid = data_blob['flowgts'][(orgfrm, i)]

                epe = torch.sqrt(torch.sum((flow_pr_ - flowgt) ** 2, dim=1, keepdim=True) + 1e-10)
                flowmag = torch.sqrt(torch.sum(flowgt ** 2, dim=1, keepdim=True) + 1e-10)

                valid = (torch.isnan(epe) == 0) * valid
                valid = (valid == 1)

                totval = torch.sum(valid)
                epe_sum = torch.sum(epe[valid])
                px1 = torch.sum((epe < 1)[valid])
                px3 = torch.sum((epe < 3)[valid])
                outliers = torch.sum((((epe > 3) * (epe > 0.05 * flowmag) + (epe > maxepe)) > 0).float() * valid)

                measurements_val[int(np.abs(i - orgfrm))] += torch.stack([epe_sum, px1, px3, outliers, totval]).cpu()
                cnt += 1

    pxl_sum = 0
    out_sum = 0

    for k in measurements_val.keys():
        pxl_sum += measurements_val[k][4]
        out_sum += measurements_val[k][3]
        measurements_val[k][0:4] = measurements_val[k][0:4] / measurements_val[k][4]
        measurements_val[k] = {'epe': measurements_val[k][0].item(), 'px1': measurements_val[k][1].item(), 'px3': measurements_val[k][2].item(), 'out': measurements_val[k][3].item()}

    out_overall = out_sum / pxl_sum

    for kk in measurements_val[1].keys():
        if writer is not None:
            writer.add_scalar('Eval_scannet_gap_{}/{}'.format(k, kk), measurements_val[1][kk], steps)
    return measurements_val, out_overall


class Validation():
    def __init__(self, project_root):
        self.project_root = project_root
        self.min_outlier = 1

    def apply_eval(self, model, writer, steps, args, save=True):
        results, outlier = validate_scannet(model, args, writer=writer, steps=steps)

        torch.cuda.synchronize()

        if outlier < self.min_outlier:
            self.min_outlier = outlier
            if save:
                PATH = os.path.join(self.project_root, 'checkpoints', args.experiment_name, 'minimal_outlier_scannet.pth')
                if isinstance(model, (DataParallel, DistributedDataParallel)):
                    model = model.module
                torch.save(model.state_dict(), PATH)
                print("saving checkpoints to %s" % PATH)

        for k in results.keys():
            result = results[k]
            print("Scannet: Gap_%s, Metric Epe: %.4f, px1: %.4f, px3: %.4f, out: %.4f" % (k, result['epe'], result['px1'], result['px3'], result['out']))

        model.train()
