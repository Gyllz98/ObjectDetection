from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

# Assuming `train_dataset` is already created with all data
def balanced_loader(dataset, batch_size=16, ratio=1/2, shuffle=True, weight_sample = False):
    # WeightedRandomSampler for training
    labels = torch.tensor(dataset.labels)
    class_counts = torch.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_counts
    weights = class_weights[labels.long()]
    sampler = WeightedRandomSampler(weights, len(weights))

    if weight_sample:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size)

    return loader

## WeightedRandomSampler(weights, len(weights))