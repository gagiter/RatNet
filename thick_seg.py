

import torch
import util
import math
from datetime import datetime
import numpy as np
import segmentation_models_pytorch as smp
import os
import pandas as pd
from ignite.metrics import ConfusionMatrix
import ignite
import random


class ThickSeg3D(torch.nn.Module):
    def __init__(self, num_planes=3, classes=9, width=512, height=512):
        super(ThickSeg3D, self).__init__()
        self.num_planes = num_planes
        self.classes = classes
        self.num_features = 8
        self.net = smp.Unet('mobilenet_v2', in_channels=num_planes * self.num_features, classes=classes * num_planes)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.plane_height = height
        self.plane_width = width
        self.device = torch.device('cpu')
        self.planes = None
        self.indices2d = None
        self.labels2d = None
        self.predict2d = None
        self.indices = None
        self.unseen_mask = None
        self.project_mask = None
        self.predict = None
        self.hit_rate = 0.0
        self.num_points = 0
        self.training = True
        self.loss = torch.FloatTensor([0.0])
        self.acc = 0.0
        self.occupancy = 0.0
        self.loss_array = []
        self.acc_array = []
        self.hit_rate_array = []
        self.occupancy_array = []
        self.batch_size = 0
        self.oa = 0.0
        self.iou = []
        self.mean_iou = 0.0
        self.crop_config = {'v': 0.2 * math.pi, 'w0': math.pi / 4.0, 'w1': math.pi,
                            's0': 0.5, 's1': 2.0}
        self.crop_regions = []
        self.flip = False

    def forward(self, data):
        self.prepare(data)
        condition = 0.01
        while torch.any(self.unseen_mask):
            self.project(data)
            self.segment_2d()
            self.count_loss()
            if self.labels2d is not None and self.training:
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.accumulate()
            if self.occupancy < condition:
                self.direct_guide(data)
                break

        self.metric(data)

    def train(self, mode=True):
        self.net.train(mode)
        self.training = mode

    def eval(self):
        self.net.eval()
        self.training = False
        pass

    def to(self, device=None, dtype=None, non_blocking=False):
        self.device = device
        self.net.to(device, dtype, non_blocking)

    def metric(self, data):
        if 'labels' in data:
            mask = (data['labels'].squeeze(-1) > 0) & (self.predict > 0)
            y = data['labels'].squeeze(-1)[mask] - 1
            y_pred = self.predict[mask] - 1

            num_classes = self.classes - 1
            indices = num_classes * y + y_pred
            cm = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)

            def compute(cm):
                sum_over_row = cm.sum(0)
                sum_over_col = cm.sum(1)
                cm_diag = cm.diag().double()
                denominator = sum_over_row + sum_over_col - cm_diag
                iou = torch.zeros_like(cm_diag)
                deno_mask = denominator > 0
                iou[deno_mask] = cm_diag[deno_mask] / denominator[deno_mask]
                mean_iou = iou[deno_mask].float().mean().item()
                oa = (cm_diag.sum() / cm.sum()).item()
                return iou, mean_iou, oa

            self.iou, self.mean_iou, self.oa = compute(cm)
            print('mIoU: %.5f oa: %.5f' % (self.mean_iou, self.oa))

    def save_points(self, data, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for batch in range(self.batch_size):
            points = data['points'][batch].cpu().detach().numpy()
            predict = self.predict[batch].cpu().detach().numpy()
            predict_out = np.concatenate([points[:, 0:3], points[:, 4:7], np.expand_dims(predict, axis=-1)], axis=-1)
            path = os.path.join(out_dir, data['name'][batch] + '-predict.txt')
            np.savetxt(path, predict_out, fmt='%.5f %.5f %.5f %d %d %d %d')

            if 'labels' in data:
                target = data['labels'][batch].cpu().detach().numpy()
                target_out = np.concatenate([points[:, 0:3], points[:, 4:7], target], axis=-1)
                path = os.path.join(out_dir, data['name'][batch] + '-target.txt')
                np.savetxt(path, target_out, fmt='%.5f %.5f %.5f %d %d %d %d')

    def save_labels(self, data, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for batch in range(self.batch_size):
            predict = self.predict[batch].cpu().detach().numpy()
            path = os.path.join(out_dir, data['name'][batch] + '.labels')
            np.savetxt(path, predict, fmt='%d')

    def count_loss(self):
        if self.labels2d is not None:
            mask = self.labels2d > 0
            predict = torch.nn.functional.log_softmax(self.predict2d, dim=-1)
            predict = predict[mask]
            target = self.labels2d[mask]
            self.loss = torch.nn.functional.nll_loss(predict, target)
            self.loss_array.append(self.loss.item())
            predict = torch.argmax(self.predict2d, dim=-1)[mask]
            self.acc = (predict == target).float().mean().item()
            self.acc_array.append(self.acc)

            print('hit_rate: %.5f occupancy: %.5f loss: %.5f acc: %.5f' %
                  (self.hit_rate, self.occupancy, self.loss.item(), self.acc))


    def accumulate(self):
        semantic2d = torch.argmax(self.predict2d, dim=-1)
        for batch in range(self.batch_size):
            self.predict[batch, self.indices2d[batch, self.project_mask[batch]]] = \
                semantic2d[batch, self.project_mask[batch]]
        features = self.planes[self.project_mask]
        features[:, 7] = semantic2d[self.project_mask].float() / self.classes
        self.planes[self.project_mask] = features


    def prepare(self, data):

        batch_size = data['points'].shape[0]
        self.planes = torch.zeros([batch_size, self.plane_height, self.plane_width, self.num_planes, self.num_features],
                                  dtype=torch.float32, device=self.device)
        self.project_mask = torch.zeros([batch_size, self.plane_height, self.plane_width, self.num_planes],
                                  dtype=torch.bool, device=self.device)
        self.labels2d = None
        if 'labels' in data:
            self.labels2d = torch.zeros([batch_size, self.plane_height, self.plane_width, self.num_planes],
                                        dtype=torch.long, device=self.device)
        data['points'] = data['points'].to(self.device)
        if 'labels' in data:
            data['labels'] = data['labels'].to(self.device)

        self.batch_size = batch_size
        self.num_points = data['points'].shape[1]
        self.indices = torch.arange(self.num_points, device=self.device)
        self.indices = self.indices.expand(self.batch_size, self.num_points)
        self.unseen_mask = torch.ones_like(self.indices, dtype=torch.bool)
        self.predict = torch.zeros([self.batch_size, self.num_points], dtype=torch.long, device=self.device)
        self.crop_regions = self.make_crop_region()
        self.loss_array = []
        self.acc_array = []
        self.hit_rate_array = []

    def direct_guide(self, data):

        semantic2d = (self.planes[..., -1] * self.classes).long()
        for batch in range(self.batch_size):
            num_unseen = self.unseen_mask[batch].sum().item()
            local_indices = torch.arange(num_unseen, device=self.device)
            xyz = data['points'][batch, :, 0:3][self.unseen_mask[batch]]
            sphere = util.project_sphere(xyz)
            u = sphere[:, 0]
            v = sphere[:, 1]
            u = (u + math.pi + self.crop_regions[batch]['r']).fmod(math.pi * 2.0)
            u = ((u - self.crop_regions[batch]['u']) / self.crop_regions[batch]['w'] + 0.5) * self.plane_width
            v = ((v - self.crop_regions[batch]['v']) / self.crop_regions[batch]['h'] + 0.5) * self.plane_height
            crop_mask = (u >= 0) & (u < self.plane_width) & (v >= 0) & (v < self.plane_height)
            u = u[crop_mask].long()
            v = v[crop_mask].long()
            w = torch.rand_like(u, dtype=torch.float32)
            w = (w * self.num_planes - 0.0001).long()
            local_indices = local_indices[crop_mask]

            global_indices = self.indices[batch, self.unseen_mask[batch]][local_indices]
            self.predict[batch, global_indices] = semantic2d[batch, v, u, w]
            self.unseen_mask[batch, global_indices] = False

    def make_crop_region(self):
        crop_regions = []
        for b in range(self.batch_size):
            region = dict()
            if self.training:
                region['w'] = random.uniform(self.crop_config['w0'], self.crop_config['w1'])
                region['u'] = random.uniform(region['w'] * 0.5,
                                             math.pi * 2 - region['w'] * 0.5)
                region['s'] = random.uniform(self.crop_config['s0'], self.crop_config['s1'])
                region['v'] = random.uniform(-self.crop_config['v'], self.crop_config['v'])
                region['h'] = region['w'] / self.plane_width * self.plane_height
                region['r'] = random.uniform(0.0, math.pi * 2.0)
                self.flip = random.random() > 0.5
            else:
                region['w'] = math.pi * 2
                region['u'] = math.pi
                region['s'] = 1.0
                region['v'] = 0.0
                region['h'] = region['w'] / self.plane_width * self.plane_height
                region['r'] = 0.0
                self.flip = False
            crop_regions.append(region)
        return crop_regions

    def project(self, data):

        self.indices2d = torch.zeros(
            [self.batch_size, self.plane_height, self.plane_width, self.num_planes],
            dtype=torch.long, device=self.device) - 1
        hit_rate = 0.0
        for batch in range(self.batch_size):
            num_unseen = self.unseen_mask[batch].sum().item()
            local_indices = torch.arange(num_unseen, device=self.device)
            xyz = data['points'][batch, :, 0:3][self.unseen_mask[batch]]
            irgb = data['points'][batch, :, 3:7][self.unseen_mask[batch]] / 255.0
            sphere = util.project_sphere(xyz)
            u = sphere[:, 0]
            v = sphere[:, 1]
            rho = sphere[:, 2] * self.crop_regions[batch]['s']
            rho = rho.clamp_min(1.0)
            rho = 1.0 / rho
            u = (u + math.pi + self.crop_regions[batch]['r']).fmod(math.pi * 2.0)
            u = ((u - self.crop_regions[batch]['u']) / self.crop_regions[batch]['w'] + 0.5) * self.plane_width
            v = ((v - self.crop_regions[batch]['v']) / self.crop_regions[batch]['h'] + 0.5) * self.plane_height
            offset_u = u - torch.floor(u)
            offset_v = v - torch.floor(v)
            offset = torch.stack([offset_u, offset_v], dim=-1)
            crop_mask = (u >= 0) & (u < self.plane_width) & (v >= 0) & (v < self.plane_height)
            u = u[crop_mask]
            v = v[crop_mask]
            local_indices = local_indices[crop_mask]

            u = u.long()
            v = v.long()

            w = torch.rand_like(u, dtype=torch.float32)
            w = (w * self.num_planes - 0.0001).long()

            self.indices2d[batch, v, u, w] = local_indices
            self.project_mask[batch] = self.indices2d[batch] >= 0
            local_indices = self.indices2d[batch, self.project_mask[batch]]
            global_indices = self.indices[batch, self.unseen_mask[batch]][local_indices]
            self.unseen_mask[batch, global_indices] = False
            features = self.planes[batch, self.project_mask[batch]]
            features[:, 0:4] = irgb[local_indices]
            features[:, 4:6] = offset[local_indices]
            features[:, 6] = rho[local_indices]
            self.planes[batch, self.project_mask[batch]] = features
            self.indices2d[batch, self.project_mask[batch]] = global_indices
            if 'labels' in data:
                self.labels2d[batch, self.project_mask[batch]] = data['labels'][batch, global_indices].squeeze(-1)
            if num_unseen > 0:
                hit_rate += local_indices.shape[0] / num_unseen

        self.hit_rate = hit_rate / self.batch_size
        self.hit_rate_array.append(self.hit_rate)
        self.occupancy = self.project_mask.float().mean().item()
        self.occupancy_array.append(self.occupancy)

    def pre_process(self, data):
        self.project(data)

    def segment_2d(self):
        self.planes = self.planes.permute(0, 3, 4, 1, 2).reshape(
            [self.batch_size, self.num_planes * self.num_features, self.plane_height, self.plane_width])
        if self.flip:
            self.predict2d = self.net(self.planes.flip(dims=[-1]))
            self.predict2d = self.predict2d.flip(dims=[-1])
        else:
            self.predict2d = self.net(self.planes)
        self.planes = self.planes.permute(0, 2, 3, 1).reshape(
            [self.batch_size, self.plane_height, self.plane_width, self.num_planes, self.num_features])
        self.predict2d = self.predict2d.permute(0, 2, 3, 1).reshape(
            [self.batch_size, self.plane_height, self.plane_width, self.num_planes, self.classes])


