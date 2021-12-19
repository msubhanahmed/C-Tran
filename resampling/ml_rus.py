import numpy as np
import random
import copy

import resampling.metrics as mld_metrics


def ML_RUS(y, p):
    y_new = copy.deepcopy(y)

    samples_to_delete = int(y.shape[0] * p / 100)
    samples_per_label = {}
    samples = np.arange(y.shape[0])

    for label in range(y.shape[1]):
        label_samples = y[:, label] == 1
        samples_per_label[label] = samples[label_samples]

    mean_ir = mld_metrics.mean_ir(y)
    majority_bag = []

    for i in range(y.shape[1]):
        if mld_metrics.ir_per_label(i, y) < mean_ir:
            majority_bag.append(i)

    delete_samples = []

    while samples_to_delete > 0 and len(majority_bag) > 0:
        for label in majority_bag:
            x = random.randint(0, len(samples_per_label[label]) - 1)
            y_new[samples_per_label[label][x]] = np.zeros(y.shape[1])

            if mld_metrics.ir_per_label(label, y_new) >= mean_ir:
                majority_bag.remove(label)
            
            delete_samples.append(samples_per_label[label][x])
            samples_to_delete -= 1
            samples_per_label[label] = np.delete(samples_per_label[label], x)

    return delete_samples