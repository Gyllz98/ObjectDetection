import json
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from Dataset import PotholeDataset
from tqdm import tqdm
from OurDataloader import balanced_loader
import torch
from PotholeCNN import PotholeCNN
from TrainLoop import train

json_path = r"/zhome/33/9/203501/Projects/IDLCV/ObjectDetection/Potholes/splits.json"
img_dir = r"/zhome/33/9/203501/Projects/IDLCV/ObjectDetection/Potholes/annotated-images"
annotations_dir = r"/zhome/33/9/203501/Projects/IDLCV/ObjectDetection/Potholes/labeled_proposals"

if __name__ == "__main__":
    # Load splits.json
    with open(json_path, 'r') as f:
        splits = json.load(f)

    # Extract train and test splits
    train_files = splits['train']
    test_files = splits['test']

    # Further split the train set into train and validation
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    # Initialize datasets
    train_dataset = PotholeDataset(img_dir, annotations_dir, train_files, transform=transform)
    print(len(train_dataset))
    val_dataset = PotholeDataset(img_dir, annotations_dir, val_files, transform=transform)
    test_dataset = PotholeDataset(img_dir, annotations_dir, test_files, transform=transform)

    # # WeightedRandomSampler for training
    # labels = [label for _, label in tqdm(train_dataset,desc = "sampler for split")]
    
    class_bgd_ratio = 1/3
    train_loader = balanced_loader(train_dataset, batch_size=16, ratio=class_bgd_ratio)  # DataLoader for training
    val_loader = balanced_loader(val_dataset, batch_size=16, ratio=class_bgd_ratio)    # DataLoader for validation
    # test_loader = balanced_loader(test_dataset, batch_size=16, ratio=class_bgd_ratio, shuffle=False)    # DataLoader for testing
    # Initialize model, optimizer, and load data
    model = PotholeCNN()  # Your model class
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train(model, optimizer, epochs=10, train_loader=train_loader, test_loader=val_loader, device=device)