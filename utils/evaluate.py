
import numpy as np
import logging
from collections import OrderedDict
import torch
import math
from pdb import set_trace as stop
import os
from models.utils import custom_replace
from utils.metrics import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(args,all_predictions,all_targets,all_masks,loss,loss_unk,elapsed,known_labels=0, metrics_per_class=False,verbose=True):

    all_predictions = F.sigmoid(all_predictions)

    # Separate multilabel and binary predictions
    bin_targets = all_targets[:, 1] #NORMAL column
    bin_predictions = all_predictions[:, 1]
    bin_predictions_thr = (bin_predictions > 0.5)

    multilabel_targets = np.delete(all_targets.numpy(), 1, axis=1)
    multilabel_predictions = np.delete(all_predictions.numpy(), 1, axis=1)
    multilabel_predictions_thr = (multilabel_predictions > 0.5)

    unknown_label_mask = custom_replace(all_masks,1,0,0)

    if known_labels > 0:
        ml_meanAP = ml_auc_score = ml_f1 = bin_auc = bin_f1 = ml_score = all_mAP = all_auc = all_f1 = all_kappa = odir_score = 0
    else:
        ml_meanAP = metrics.average_precision_score(multilabel_targets,multilabel_predictions, average='macro', pos_label=1)
        ml_auc_score = metrics.roc_auc_score(multilabel_targets, multilabel_predictions, average='macro')
        ml_f1 = metrics.f1_score(multilabel_targets, multilabel_predictions_thr, average='macro')

        bin_auc = metrics.roc_auc_score(bin_targets, bin_predictions)
        bin_f1 = metrics.f1_score(bin_targets, bin_predictions_thr)

        ml_score = (ml_meanAP + ml_auc_score) / 2.0

        all_mAP = metrics.average_precision_score(all_targets, all_predictions, average='macro')
        all_auc = metrics.roc_auc_score(all_targets, all_predictions, average='macro')
        all_f1 = metrics.f1_score(all_targets, (all_predictions>0.5), average='macro')

        all_kappa = 0.0
        for col in range(0, all_targets.size(dim=1)):
            all_kappa += metrics.cohen_kappa_score(all_targets[:, col], (all_predictions[:, col]>0.5))

        all_kappa /= (all_targets.size(dim=1) * 1.0)

        odir_score = (all_auc + all_f1 + all_kappa) / 3.0

    if metrics_per_class:
        scores = np.zeros((5, all_targets.size(dim=1)))
        for idx in range(all_targets.size(dim=1)):
            scores[0, idx] = metrics.accuracy_score(all_targets[:, idx], (all_predictions[:, idx] > 0.5))
            scores[1, idx] = metrics.precision_score(all_targets[:, idx], (all_predictions[:, idx] > 0.5))
            scores[2, idx] = metrics.recall_score(all_targets[:, idx], (all_predictions[:, idx] > 0.5))
            scores[3, idx] = metrics.f1_score(all_targets[:, idx], (all_predictions[:, idx] > 0.5))
            scores[4, idx] = metrics.roc_auc_score(all_targets[:, idx], all_predictions[:, idx])

        for idx in range(all_targets.size(dim=1)):
            for idx_metric in range(0, 5):
                print(scores[idx_metric, idx], end=',')
            print('')

        np.savetxt('all_predictions.csv', all_predictions.numpy(), delimiter=',')


    optimal_threshold = 0.5

    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()

    top_3rd = np.sort(all_predictions)[:,-3].reshape(-1,1)
    all_predictions_top3 = all_predictions.copy()
    all_predictions_top3[all_predictions_top3<top_3rd] = 0
    all_predictions_top3[all_predictions_top3<optimal_threshold] = 0
    all_predictions_top3[all_predictions_top3>=optimal_threshold] = 1

    CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
    CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
    CF1_top3 = (2*CP_top3*CR_top3)/(CP_top3+CR_top3)
    OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
    OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
    OF1_top3 = (2*OP_top3*OR_top3)/(OP_top3+OR_top3)


    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', OF1),
                        ('Label-based Macro F1', CF1)])


    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('loss:  {:0.3f}'.format(loss))
        print('lossu: {:0.3f}'.format(loss_unk))
        print('----')
        print('ml_f1:    {:0.5f}'.format(ml_f1))
        print('ml_mAP:   {:0.5f}'.format(ml_meanAP))
        print('ml_auc:   {:0.5f}'.format(ml_auc_score))
        print('ml_score: {:0.5f}'.format(ml_score))
        print('bin_auc:  {:0.5f}'.format(bin_auc))
        print('bin_f1:   {:0.5f}'.format(bin_f1))
        print('model_score: {:0.5f}'.format((ml_score + bin_auc)/2.0))
        print('----')
        print('all_mAP: {:0.5f}'.format(all_mAP))
        print('all_f1: {:0.5f}'.format(all_f1))
        print('all_auc: {:0.5f}'.format(all_auc))
        print('all_kappa: {:0.5f}'.format(all_kappa))
        print('odir_score: {:0.5f}'.format(odir_score))
        print('---')
        print('CP:    {:0.1f}'.format(CP*100))
        print('CR:    {:0.1f}'.format(CR*100))
        print('CF1:   {:0.1f}'.format(CF1*100))
        print('OP:    {:0.1f}'.format(OP*100))
        print('OR:    {:0.1f}'.format(OR*100))
        print('OF1:   {:0.1f}'.format(OF1*100))

    metrics_dict = {}
    metrics_dict['ml_mAP'] = ml_meanAP
    metrics_dict['ml_auc'] = ml_auc_score
    metrics_dict['ml_score'] = ml_score
    metrics_dict['bin_auc'] = bin_auc
    metrics_dict['model_score'] = (ml_score + bin_auc)/2.0
    metrics_dict['odir_score'] = odir_score

    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    print('')

    return metrics_dict