"""Metrics
calsulate ious and mean iou
"""
import numpy as np
import torch

class Metrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.initialize()

    def initialize(self):
        self.cmx = np.zeros((self.n_classes, self.n_classes))
        self.loss_list = []
        self.intersection = torch.zeros(self.n_classes)
        self.union = torch.zeros(self.n_classes)

    def update(self, preds, targets, loss):
        
        self.loss_list.append(loss)
        
        pred = preds.view(-1)
        target = targets.view(-1)

        for c in range(self.n_classes):
            pred_inds = pred == c
            target_inds = target == c
            intersection = (pred_inds[target_inds]).long().sum().data.cpu()
            union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
            self.intersection[c] += intersection
            self.union[c] += union

    def calc_metrics(self):

        for c in range(self.n_classes):
            if self.union[c] == 0:
                self.union[c] = float('nan')
        
        loss = np.mean(self.loss_list)

        ious = np.array(self.intersection) / np.array(self.union)
        mean_iou = np.nanmean(ious)

        return loss, mean_iou