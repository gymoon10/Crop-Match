"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import train_crop_config

# from criterions.loss import CriterionCE, CriterionMatching
from criterions.loss_crop import CriterionCE, CriterionMatching

from datasets import get_dataset
from models import get_model, ERFNet_Semantic_Original
from utils.utils import AverageMeter, Logger, Visualizer  # for CVPPP


torch.backends.cudnn.benchmark = True

args = train_crop_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    if not os.path.exists(args['save_dir1']):
        os.makedirs(args['save_dir1'])
    if not os.path.exists(args['save_dir2']):
        os.makedirs(args['save_dir2'])
    if not os.path.exists(args['save_dir_aux']):
        os.makedirs(args['save_dir_aux'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader (student)
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader (student)
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set criterion
criterion_val = CriterionCE()
criterion = CriterionMatching(num_classes=3, ths_prob=0.6, ths_sim=0.9)

criterion_val = torch.nn.DataParallel(criterion_val).to(device)
criterion = torch.nn.DataParallel(criterion).to(device)

# Logger
logger = Logger(('train', 'train_ce_loss', 'train_matching_loss',
                 'val', 'val_iou_plant', 'val_iou_disease'), 'loss')


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def save_checkpoint(epoch, state, recon_best1, recon_best2, recon_best3, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if recon_best1:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_plant_model_%d.pth' % (epoch)))

    if recon_best2:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_disease_model_%d.pth' % (epoch)))

    if recon_best3:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_both_model_%d.pth' % (epoch)))

def main():
    # init
    start_epoch = 0
    best_iou_plant = 0
    best_iou_disease = 0
    best_iou_both = 0

    # set model (student)
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)
    if args['pretrained_path']:
        state = torch.load(args['pretrained_path'])
        model.load_state_dict(state['model_state_dict'], strict=False)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=1e-4)

    def lambda_(epoch):
        return pow((1 - ((epoch) / args['n_epochs'])), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_, )

    # resume (student)
    if args['resume_path'] is not None and os.path.exists(args['resume_path']):
        print('Resuming model-student from {}'.format(args['resume_path']))
        state = torch.load(args['resume_path'])
        start_epoch = state['epoch'] + 1
        best_iou_plant = state['best_iou_plant']
        best_iou_disease = state['best_iou_disease']
        best_iou_both = state['best_iou_both']
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']

    for epoch in range(start_epoch, args['n_epochs']):
        print('Starting epoch {}'.format(epoch))

        loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_matching_meter = AverageMeter()

        # Training (Student)
        for i, sample in enumerate(tqdm(train_dataset_it)):
            image = sample['image']  # (N, 3, 512, 512)
            label = sample['label_all'].squeeze(1)  # (N, 512, 512)

            # augmented set
            image_aug = sample['image_crop']
            label_aug = sample['label_crop'].squeeze(1)  # (N, 512, 512)

            # ------------------------ forward ------------------------
            model.train()
            outputs, embeddings = model(image)  # (N, num_classes=3, 512, 512), # (N, c, h, w)
            outputs_aug, embeddings_aug = model(image_aug)

            # ------------------------ resize to embedding size ------------------------
            # 1. resize labels to embedding resolution
            size = (embeddings.shape[2], embeddings.shape[3])  # (h, w)
            label_down = nn.functional.interpolate(label.float().unsqueeze(1),
                                                   size=size, mode='nearest').squeeze(1)  # (N, h, w)
            label_aug_down = nn.functional.interpolate(label_aug.float().unsqueeze(1),
                                                       size=size, mode='nearest').squeeze(1)

            # ------------------------ calculate loss ------------------------
            loss, loss_ce, loss_matching = \
                criterion(outputs=outputs,  embeddings=embeddings, class_labels=label,
                          outputs_aug=outputs_aug, embeddings_aug=embeddings_aug, class_labels_aug=label_aug)

            loss = loss.mean()
            loss_ce = loss_ce.mean()
            loss_matching = loss_matching.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            loss_ce_meter.update(loss_ce.item())
            loss_matching_meter.update(loss_matching.item())

        train_loss, train_ce, train_matching= \
            loss_meter.avg, loss_ce_meter.avg, loss_matching_meter.avg
        scheduler.step()

        print('===> train loss: {:.5f}, train-ce: {:.5f}, train-matching: {:.5f}' \
              .format(train_loss, train_ce, train_matching))
        logger.add('train', train_loss)
        logger.add('train_ce_loss', train_ce)
        logger.add('train_matching_loss', train_matching)

        # validation
        loss_val_meter = AverageMeter()
        iou1_meter, iou2_meter = AverageMeter(), AverageMeter()

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                image = sample['image']  # (N, 3, 512, 512)
                label = sample['label_all'].squeeze(1)  # (N, 512, 512)

                output, _ = model(image)  # (N, 4, h, w)

                loss = criterion_val(output, label,
                                     iou=True, meter_plant=iou1_meter, meter_disease=iou2_meter)
                loss = loss.mean()
                loss_val_meter.update(loss.item())

        val_loss, val_iou_plant, val_iou_disease = loss_val_meter.avg, iou1_meter.avg, iou2_meter.avg
        print('===> val loss: {:.5f}, val iou-plant: {:.5f}, val iou-disease: {:.5f}'.format(val_loss, val_iou_plant,
                                                                                             val_iou_disease))

        logger.add('val', val_loss)
        logger.add('val_iou_plant', val_iou_plant)
        logger.add('val_iou_disease', val_iou_disease)
        logger.plot(save=args['save'], save_dir=args['save_dir'])

        # save
        is_best_plant = val_iou_plant > best_iou_plant
        best_iou_plant = max(val_iou_plant, best_iou_plant)

        is_best_disease = val_iou_disease > best_iou_disease
        best_iou_disease = max(val_iou_disease, best_iou_disease)

        val_iou_both = (val_iou_plant + val_iou_disease) / 2

        is_best_both = val_iou_both > best_iou_both
        best_iou_both = max(val_iou_both, best_iou_both)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou_plant': best_iou_plant,
                'best_iou_disease': best_iou_disease,
                'best_iou_both': best_iou_both,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data
            }
            save_checkpoint(epoch, state, is_best_plant, is_best_disease, is_best_both)


if __name__ == '__main__':
    main()






