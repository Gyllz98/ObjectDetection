from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
from tqdm import tqdm

# Assuming `train_dataset` is already created with all data
def balanced_loader(dataset, batch_size=16, ratio=1/2, shuffle=True):
    potholes = []
    backgrounds = []

    # Separate potholes and backgrounds
    for idx in tqdm(range(len(dataset)),"separation of potholes"):
        _, label = dataset[idx]
        if label.item() == 1:
            potholes.append(idx)
        else:
            backgrounds.append(idx)

    # Ensure we have all potholes and sample backgrounds at the specified ratio
    desired_background_count = int(len(potholes) / ratio)
    selected_backgrounds = random.sample(backgrounds, min(desired_background_count, len(backgrounds)))

    # Combine the indices
    balanced_indices = potholes + selected_backgrounds

    if shuffle:
        random.shuffle(balanced_indices)

    # Create a subset DataLoader with the balanced data
    balanced_subset = torch.utils.data.Subset(dataset, balanced_indices)
    loader = DataLoader(balanced_subset, batch_size=batch_size, shuffle=shuffle)

    return loader