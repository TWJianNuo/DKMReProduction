import numpy as np
import torch
import torch.utils.data as data

import os
import glob
import random
import PIL.Image as Image

from dkm.utils.utils import get_connection_graph

class Scannet(data.Dataset):
    def __init__(self, root, relfrmin):
        self.root = root
        self.init_seed = False
        self.relfrmin = relfrmin

        self.orgh = 480
        self.orgw = 640
        self.sampleskip = 10
        self.generate_entries()

        xx, yy = np.meshgrid(range(self.orgw), range(self.orgh), indexing='xy')
        self.xx = xx
        self.yy = yy
        self.ones = np.ones_like(xx)

    def generate_entries(self):
        minrel = np.min(np.array(self.relfrmin))
        maxrel = np.max(np.array(self.relfrmin))

        self.entries = list()
        scenes_paths = glob.glob(os.path.join(self.root, '*'))
        for scene_path in scenes_paths:
            jpgs_paths = glob.glob(os.path.join(scene_path, 'color', '*.jpg'))
            imgnum = len(jpgs_paths)
            scene_name = scene_path.split('/')[-1]
            for jpg_path in jpgs_paths:
                jpg_idx = int(jpg_path.split('/')[-1].split('.')[0])
                if int(jpg_idx / self.sampleskip) + minrel >= 0 and int(jpg_idx / self.sampleskip) + maxrel < imgnum:
                    extrinsinc = load_matrix_from_txt(os.path.join(self.root, scene_name, 'pose', '{}.txt'.format(str(jpg_idx))))
                    if np.sum(np.isinf(extrinsinc)) == 0:
                        self.entries.append("{} {}".format(scene_name, jpg_idx))

        random.seed(2022)
        random.shuffle(self.entries)
        maxevalnum = 800
        self.entries = self.entries[0:maxevalnum]

    def read_rgbs(self, scene, frmidx, relfrmin):
        rgbs = dict()
        for key in relfrmin:
            rgb_path = os.path.join(self.root, scene, 'color', '{}.jpg'.format(str(frmidx + key * self.sampleskip)))
            rgb = Image.open(rgb_path)

            scalex = self.orgw / rgb.size[0]
            scaley = self.orgh / rgb.size[1]

            rgb = Image.open(rgb_path).resize((self.orgw, self.orgh))
            rgbs[key] = np.array(rgb)
        return rgbs, scalex, scaley

    def read_extrinsincs(self, scene, frmidx, relfrmin):
        extrinsincs = dict()
        for key in relfrmin:
            extrinsinc_path = os.path.join(self.root, scene, 'pose', '{}.txt'.format(str(frmidx + key * self.sampleskip)))
            extrinsincs[key] = load_matrix_from_txt(extrinsinc_path)
        return extrinsincs

    def read_depths(self, scene, frmidx, relfrmin):
        depths = dict()
        for key in relfrmin:
            depth_path = os.path.join(self.root, scene, 'depth', '{}.png'.format(str(frmidx + key * self.sampleskip)))
            depths[key] = np.array(Image.open(depth_path)).astype(np.uint16).astype(np.float32) / 1000
        return depths

    def generate_flowgt(self, depths, poses, intrinsic_rgb, intrinsic_depth, relfrmin, orgfrm):
        flowgts = dict()
        posegts = dict()

        pts_rgb = np.expand_dims(np.stack([self.xx, self.yy, np.ones_like(self.xx)], axis=2), axis=3)
        prjM_rgb2depth = intrinsic_depth @ np.linalg.inv(intrinsic_rgb)
        prjM_rgb2depth = np.expand_dims(np.expand_dims(prjM_rgb2depth[0:3, 0:3], axis=0), axis=0)
        pts_depth = prjM_rgb2depth @ pts_rgb

        pts_depth_sx = (pts_depth[:, :, 0, 0] / (self.orgw - 1) - 0.5) * 2
        pts_depth_sy = (pts_depth[:, :, 1, 0] / (self.orgh - 1) - 0.5) * 2
        pts_depth_sample = torch.from_numpy(np.stack([pts_depth_sx, pts_depth_sy], axis=2)).float().unsqueeze(0)

        depth_resampled = dict()
        val_resampled = dict()
        for k in depths.keys():
            depth = depths[k]
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            depth_sampled = torch.nn.functional.grid_sample(depth, pts_depth_sample, mode='bilinear', align_corners=True)
            val_sampled = torch.nn.functional.grid_sample((depth > 0).float(), pts_depth_sample, mode='bilinear', align_corners=True) > 0.9

            depth_resampled[k] = depth_sampled.squeeze().numpy()
            val_resampled[k] = val_sampled.squeeze().float().numpy()

        for i in relfrmin:
            for j in relfrmin:
                if i != j and i == orgfrm:
                    depth = depth_resampled[i]
                    posei = poses[i]
                    posej = poses[j]

                    prjM = intrinsic_rgb @ np.linalg.inv(posej) @ posei @ np.linalg.inv(intrinsic_rgb)

                    pts3d = np.expand_dims(np.stack([self.xx * depth, self.yy * depth, depth, self.ones], axis=-1), axis=-1)
                    pts3d_prj = prjM @ pts3d

                    pts3d_prj_x = pts3d_prj[:, :, 0, 0] / (pts3d_prj[:, :, 2, 0] + 1e-10)
                    pts3d_prj_y = pts3d_prj[:, :, 1, 0] / (pts3d_prj[:, :, 2, 0] + 1e-10)

                    flowx = pts3d_prj_x - self.xx
                    flowy = pts3d_prj_y - self.yy

                    flow = np.stack([flowx, flowy], axis=-1)

                    val = (depth > 0) * (pts3d_prj_x > 0) * (pts3d_prj_x < self.orgw) * (pts3d_prj_y > 0) * (pts3d_prj_y < self.orgh) * val_resampled[i]

                    flowgts[(i, j)] = [flow, val]

                    posegts[(i, j)] = np.linalg.inv(posej) @ posei
        return flowgts, posegts

    def formatting(self, rgbs, flowgts, depths, posegts, intrinsic, tag):
        outputs = dict()
        for k in rgbs.keys():
            rgbs[k] = np.transpose(rgbs[k].astype(np.float32), (2, 0, 1))

        for k in flowgts.keys():
            flow, val = flowgts[k]
            flowgts[k] = [np.transpose(flow.astype(np.float32), (2, 0, 1)), np.expand_dims(val.astype(np.float32), axis=0)]

        for k in posegts.keys():
            posegts[k] = posegts[k].astype(np.float32)

        for k in depths.keys():
            depths[k] = np.expand_dims(depths[k].astype(np.float32), axis=0)

        outputs['rgbs'] = rgbs
        outputs['depths'] = depths
        outputs['flowgts'] = flowgts
        outputs['posegts'] = posegts
        outputs['intrinsic'] = intrinsic.astype(np.float32)
        outputs['tag'] = tag
        return outputs

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        scene, frmidx = self.entries[index].split(' ')
        frmidx = int(frmidx)
        relfrmin, orgfrm, suppconnections = get_connection_graph(self.relfrmin)

        rgbs, scalex, scaley = self.read_rgbs(scene, frmidx, relfrmin)
        depths = self.read_depths(scene, frmidx, relfrmin)
        poses = self.read_extrinsincs(scene, frmidx, relfrmin)

        intrinsic_rgb, intrinsic_depth = self.get_intrinsic(scene=scene, scalex=scalex, scaley=scaley)

        flowgts, posegts = self.generate_flowgt(depths, poses, intrinsic_rgb, intrinsic_depth, relfrmin, orgfrm)

        outputs = self.formatting(rgbs, flowgts, depths, posegts, intrinsic_rgb, self.entries[index])
        return outputs

    def __len__(self):
        return len(self.entries)

    def get_intrinsic(self, scene, scalex, scaley):

        intrinsic_rgb = load_matrix_from_txt(os.path.join(self.root, scene, 'intrinsic', 'intrinsic_color.txt'))
        intrinsic_depth = load_matrix_from_txt(os.path.join(self.root, scene, 'intrinsic', 'intrinsic_depth.txt'))

        scaleM = np.eye(4)
        scaleM[0, 0] = scalex
        scaleM[1, 1] = scaley

        intrinsic_rgb = scaleM @ intrinsic_rgb
        return intrinsic_rgb, intrinsic_depth

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)