"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
criterion_l1 = nn.L1Loss()


class CriterionCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''embedding : embedding network output (N, 32, 512, 512)
           prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        for b in range(0, batch_size):

            # 3.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + ce_loss

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

        return loss


class CriterionMatching(nn.Module):
    def __init__(self, num_classes=3, ths_prob=0.6, ths_sim=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.ths_prob = ths_prob
        self.ths_sim = ths_sim

    def forward(self, outputs, embeddings, class_labels,
                outputs_aug, embeddings_aug, class_labels_aug,
                iou=False, meter_plant=None, meter_disease=None):

        batch_size, height, width = outputs.size(
            0), outputs.size(2), outputs.size(3)

        loss_ce_total = 0
        loss_matching_total = torch.tensor(0.0).cuda()

        # calculate confidence map(probability) & segmentation prediction
        confidence_map, seg_prediction = torch.max(torch.softmax(outputs, dim=1),
                                                   dim=1)  # (N, 512, 512), (N, 512, 512)
        confidence_map_aug, seg_prediction_aug = torch.max(torch.softmax(outputs_aug, dim=1), dim=1)

        # ------------------------ resize to embedding size ------------------------
        # 1. resize labels to embedding resolution
        size = (embeddings.shape[2], embeddings.shape[3])  # (h, w)
        labels_down = nn.functional.interpolate(class_labels.float().unsqueeze(1),
                                                size=size, mode='nearest').squeeze(1)  # (N, h, w)
        labels_down_aug = nn.functional.interpolate(class_labels_aug.float().unsqueeze(1),
                                                    size=size, mode='nearest').squeeze(1)

        # 2. resize seg predictions to embedding resolution
        seg_prediction_down = nn.functional.interpolate(seg_prediction.float().unsqueeze(1), size=size,
                                                        mode='nearest').squeeze(1)
        seg_prediction_down_aug = nn.functional.interpolate(seg_prediction_aug.float().unsqueeze(1), size=size,
                                                            mode='nearest').squeeze(1)

        # 3. resize confidence map to embedding resolution
        confidence_map_down = nn.functional.interpolate(confidence_map.float().unsqueeze(1), size=size,
                                                        mode='nearest').squeeze(1)
        confidence_map_down_aug = nn.functional.interpolate(confidence_map_aug.float().unsqueeze(1), size=size,
                                                            mode='nearest').squeeze(1)

        cnt = 0
        for b in range(0, batch_size):

            # ----------------- cross entropy loss -----------------
            pred = outputs[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_labels[b].unsqueeze(0)  # (1, 512, 512)
            gt_label = gt_label.type(torch.long).cuda()

            loss_ce_total += criterion_ce(pred, gt_label)

            pred = outputs_aug[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_labels_aug[b].unsqueeze(0)  # (1, 512, 512)
            gt_label = gt_label.type(torch.long).cuda()

            # loss_ce_total += criterion_ce(pred, gt_label)

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_labels[b].unsqueeze(0) == 1)
                gt_disease = (class_labels[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

            # ----------------- feature matching loss for self-consistency (top K x top K) -----------------
            class_idx = 1
            top_k = 400

            emb = embeddings[b]
            emb_size, emb_dim = (emb.shape[1], emb.shape[2]), emb.shape[0]
            emb_flat = emb.view(emb_dim, -1)  # (32, 512*512)

            emb_aug = embeddings_aug[b]
            emb_flat_aug = emb_aug.view(emb_dim, -1)  # (32, 512*512)

            ths = 0.8
            mask_class = ((seg_prediction_down[b] == class_idx).float() * (confidence_map_down[b] > ths).float()).bool()
            mask_class = (mask_class == True).flatten()  # (256*256)

            mask_class_aug = ((seg_prediction_down_aug[b] == class_idx).float() * (
                         confidence_map_down_aug[b] > ths).float()).bool()
            mask_class_aug = (mask_class_aug == True).flatten()  # (256*256)

            emb_mask = emb_flat[:, mask_class]  # (16, m pixels)
            emb_mask_aug = emb_flat_aug[:, mask_class_aug]  # (16, j pixels)

            # Top-K sampling (original)
            feat = emb_mask.transpose(1, 0)  # (m pixels, 16)
            feat = torch.mean(feat, dim=1).unsqueeze(1)  # (m pixels, 1)

            # sort mask_emb_query (ascending order)
            _, indices = torch.sort(feat[:, 0], dim=0)
            emb_mask = emb_mask[:, indices]
            emb_mask = emb_mask[:, :top_k]  # (16, top_k pixels)

            # Top-K sampling (aug)
            feat = emb_mask_aug.transpose(1, 0)  # (j pixels, 16)
            feat = torch.mean(feat, dim=1).unsqueeze(1)  # (j pixels, 1)

            # sort mask_emb_query (ascending order)
            _, indices = torch.sort(feat[:, 0], dim=0)
            emb_mask_aug = emb_mask_aug[:, indices]
            emb_mask_aug = emb_mask_aug[:, :top_k]  # (16, top_k pixels)

            if mask_class.sum() > top_k and mask_class_aug.sum() > top_k:
                cnt += 1
                similarities = pairwise_cosine_similarity(emb_mask.T, emb_mask_aug.T)  # (top_k pixels x top_k pixels)
                distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
                print('matching-score(1) :', distances.mean())
                loss_matching_total += distances.mean()

            # ----------------- feature matching loss for self-consistency (pairwise) -----------------
            class_idx = 2
            emb = embeddings[b]
            emb_size, emb_dim = (emb.shape[1], emb.shape[2]), emb.shape[0]
            emb_flat = emb.view(emb_dim, -1)  # (32, 512*512)

            emb_aug = embeddings_aug[b]
            emb_flat_aug = emb_aug.view(emb_dim, -1)  # (32, 512*512)

            ths = 0.6
            mask_class = ((seg_prediction_down[b] == class_idx).float() * (confidence_map_down[b] > ths).float()).bool()
            mask_class_aug = ((seg_prediction_down_aug[b] == class_idx).float() * (confidence_map_down_aug[b] > ths).float()).bool()

            mask_class = (mask_class == True).flatten()  # (256*256)
            mask_class_aug = (mask_class_aug == True).flatten()  # (256*256)

            emb_mask = emb_flat[:, mask_class]  # (16, m pixels)
            emb_mask_aug = emb_flat_aug[:, mask_class_aug]  # (16, j pixels)

            if mask_class.sum() != 0 and mask_class_aug.sum() != 0:
                cnt += 1
                # do not need to normalize
                similarities = pairwise_cosine_similarity(emb_mask.T, emb_mask_aug.T)  # (m pixels x j pixels)
                distances = 1 - similarities  # values between [0, 2] where 0 means same vectors
                print('matching-score(2) :', distances.mean())
                loss_matching_total += distances.mean()

        if cnt != 0:
            loss_matching_total = loss_matching_total / cnt
        loss_ce_total = loss_ce_total / (1*batch_size)

        loss = (1 * loss_ce_total) + (2 * loss_matching_total)
        #print('CE :', loss_ce_total)
        #print('CE :', loss_ce_total.dtype)
        #print('Matching :', loss_matching_total)

        return loss, loss_ce_total, loss_matching_total


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
