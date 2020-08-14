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
        self.pred_list = []
        self.target_list = []
        self.loss_list = []
        self.kldivloss_list = []
        self.intersection = torch.zeros(self.n_classes)
        self.union = torch.zeros(self.n_classes)

    def update(self, preds, targets, loss, kldivloss):
        
        self.loss_list.append(loss)
        self.kldivloss_list.append(kldivloss)
        
        pred = preds.view(-1)
        target = targets.view(-1)

        self.pred_list.append(pred)
        self.target_list.append(target)

    def calc_metrics(self):

        pred = torch.cat([p for p in self.pred_list], axis=0)
        target = torch.cat([t for t in self.target_list], axis=0)

        pred = pred.numpy()
        target = target.numpy()

        # calc histgram and make confusion matrix
        cmx = np.bincount(self.n_classes * target.astype(int) 
                         + pred, minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        
        with np.errstate(invalid='ignore'):
            ious = np.diag(cmx) / (cmx.sum(axis=1) + cmx.sum(axis=0) - np.diag(cmx))
        
        loss = np.mean(self.loss_list)
        kldivloss = np.mean(self.kldivloss_list)

        mean_iou = np.nanmean(ious)

        return loss, kldivloss, mean_iou

    # def calc_metrics(self):

    #     pred = torch.cat([p for p in self.pred_list], axis=0)
    #     target = torch.cat([t for t in self.target_list], axis=0)

    #     for c in range(self.n_classes):
    #         pred_inds = pred == c
    #         target_inds = target == c
    #         intersection = (pred_inds[target_inds]).long().sum().data.cpu()
    #         union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
    #         self.intersection[c] += intersection
    #         self.union[c] += union

    #     for c in range(self.n_classes):
    #         if self.union[c] == 0:
    #             self.union[c] = float('nan')
        
    #     loss = np.mean(self.loss_list)
    #     kldivloss = np.mean(self.kldivloss_list)

    #     ious = np.array(self.intersection) / np.array(self.union)
    #     mean_iou = np.nanmean(ious)

    #     return loss, kldivloss, mean_iou