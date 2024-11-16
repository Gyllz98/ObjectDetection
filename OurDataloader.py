from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

# Assuming `train_dataset` is already created with all data
def balanced_loader(dataset, batch_size=16, shuffle=True, weight_sample = True):
    # WeightedRandomSampler for training
    labels = torch.tensor(dataset.labels)
    class_counts = torch.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_counts
    weights = class_weights[labels.long()]
    sampler = WeightedRandomSampler(weights, len(weights))

    if weight_sample:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)

    return loader

def balanced_loader2(dataset, batch_size=16, target_ratio=None, shuffle=True):
    """
    Creates a DataLoader with a WeightedRandomSampler to balance classes according to a target ratio.
    
    Args:
        dataset (Dataset): The dataset to load from.
        batch_size (int): Number of samples per batch.
        target_ratio (float): Desired ratio of potholes to backgrounds (e.g., 1/2).
        shuffle (bool): Whether to shuffle within the DataLoader. Only applicable if sampler is not used.
        
    Returns:
        DataLoader: DataLoader object with balanced sampling.
    """
    if target_ratio:
        labels = torch.tensor(dataset.labels)
        
        # Count the number of instances in each class
        num_potholes = torch.sum(labels == 1).item()
        num_backgrounds = torch.sum(labels == 0).item()
        
        # Calculate the current ratio
        current_ratio = num_potholes / num_backgrounds
        
        # Adjust weights to achieve target ratio
        if current_ratio < target_ratio:
            # More weight to potholes
            weight_pothole = target_ratio / current_ratio
            weight_background = 1
        else:
            # More weight to backgrounds
            weight_pothole = 1
            weight_background = current_ratio / target_ratio

        # Assign weights
        weights = torch.zeros_like(labels, dtype=torch.float32)
        weights[labels == 1] = weight_pothole
        weights[labels == 0] = weight_background
        
        # Create sampler
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader