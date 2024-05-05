import argparse,math,time,warnings,copy, numpy as np, os.path as path
from utils.assymetric_loss_opt import AsymmetricLossOptimized 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random
import os
import csv
import uuid
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from utils.poly_loss import BCEPolyLoss, FLPolyLoss


def get_class_weights(y_true):
    y_pos = np.sum(y_true, axis=0)
    weights = y_pos.max() / y_pos

    return torch.Tensor(weights)


def run_epoch(args,model,data,optimizer,epoch,desc,device,train=False,warmup_scheduler=None):

    output_folder = "saved_images"
    os.makedirs(output_folder, exist_ok=True)
    csv_file = "image_labels.csv"
    mode = 'a' if os.path.exists(csv_file) else 'w'
    with open(csv_file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':
            writer.writerow(['image_filename', 'label', 'mask_filename'])

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    if args.loss == 'asl':
        criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip, disable_torch_grad_focal_loss=args.disable_torch_fl)
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss == 'wbce':
        criterion = nn.BCEWithLogitsLoss(weight=get_class_weights(data.dataset.get_labels()).to(device), reduction='none')
    elif args.loss == 'bce_poly':
        criterion = BCEPolyLoss(eps=args.poly_eps)
    elif args.loss == 'fl_poly':
        criterion = FLPolyLoss(eps=args.poly_eps, gamma=args.poly_gamma)

    output_folder = "saved_images"
    os.makedirs(output_folder, exist_ok=True)
    csv_file = "image_labels_predictions.csv"
    mode = 'a' if os.path.exists(csv_file) else 'w'
    with open(csv_file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
    if mode == 'w':
        writer.writerow(['image_filename', 'label', 'mask_filename', 'prediction'])
            
    for batch in tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=50):
        if batch_idx == max_samples:
            break

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        unk_mask = custom_replace(mask,1,0,0)
        all_image_ids += batch['imageIDs']
        mask_in = mask.clone()

        if train:
            pred,int_pred,attns = model(images.to(device),mask_in.to(device))
        else:
            with torch.no_grad():
                pred,int_pred,attns = model(images.to(device),mask_in.to(device))

        for i in range(len(images)):
            unique_id           = uuid.uuid4()
            image               = TF.to_pil_image(images[i])
            image_filename      = f"{labels[i]}_{unique_id}.png"
            image.save(os.path.join(output_folder, image_filename))
            with open(csv_file, mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                prediction = pred[i].tolist()
                # prediction = ','.join(map(str, pred[i].tolist()))  # Convert tensor to string
                writer.writerow([image_filename, labels[i], mask_in[i], prediction])



        
        if args.dataset == 'cub':
            class_label = batch['class_label'].float()
            concept_certainty = batch['concept_certainty'].float()

            class_label_onehot = torch.zeros(class_label.size(0),200)
            class_label_onehot.scatter_(1,class_label.long(),1)

            labels = torch.cat((labels,class_label_onehot),1)
            loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.to(device),reduction='none')
            loss = (unk_mask.to(device)*loss).sum()/unk_mask.detach().sum().item()

            aux_loss =  F.binary_cross_entropy_with_logits(int_pred.view(labels.size(0),-1),labels.to(device),reduction='none')
            aux_loss = (unk_mask.to(device)*aux_loss).sum()/unk_mask.detach().sum().item()

            loss_out = 1.0*loss + float(args.aux_loss)*aux_loss
            loss = loss_out
        else:
            loss = criterion(pred.view(labels.size(0),-1), labels.to(device))
            
            if args.loss_labels == 'unk': 
                # only use unknown labels for loss
                loss_out = (unk_mask.to(device)*loss).sum()
            else: 
                # use all labels for loss
                loss_out = loss.sum() 

            # loss_out = loss_out/unk_mask.to(device).sum()
        if train:
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx +=1

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))

    return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total


