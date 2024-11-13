from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
from tqdm import tqdm

# Assuming `train_dataset` is already created with all data
def balanced_loader(dataset, batch_size=16, ratio=1/2, shuffle=True, sampler = None):
    # WeightedRandomSampler for training
    labels = torch.tensor(dataset.labels)
    class_counts = torch.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_counts
    weights = class_weights[labels.long()]
    sampler = sampler

    if sampler:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size)

    return loader

## WeightedRandomSampler(weights, len(weights))