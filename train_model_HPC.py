import json
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from Dataset import PotholeDataset
from OurDataloader import balanced_loader, balanced_loader2
import torch
from PotholeCNN import PotholeCNN, PretrainedPothole
from TrainLoop import train

json_path = r"/zhome/33/9/203501/Projects/IDLCV/ObjectDetection/Potholes/splits.json"
img_dir = r"/dtu/blackhole/0d/203501/Data/IDLCV/annotated-images"
annotations_dir = r"/dtu/blackhole/0d/203501/Data/IDLCV/labeled_proposals"

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
        transforms.Resize((250, 250)),  # Ensure images are at least slightly larger than crop size
        transforms.RandomResizedCrop(224,scale=(0.8, 1.0), ratio=(0.75, 1.33)), # Better size for ResNet101
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = PotholeDataset(img_dir, annotations_dir, train_files, transform=transform)
    # print(len(train_dataset))
    val_dataset = PotholeDataset(img_dir, annotations_dir, val_files, transform=transform_test)
    test_dataset = PotholeDataset(img_dir, annotations_dir, test_files, transform=transform_test)

    # # WeightedRandomSampler for training
    # labels = [label for _, label in tqdm(train_dataset,desc = "sampler for split")]
    
    class_bgd_ratio = 1/3
    batch_size = 16
    train_loader = balanced_loader2(train_dataset, batch_size=batch_size, target_ratio=class_bgd_ratio)  # DataLoader for training
    val_loader = balanced_loader2(val_dataset, batch_size=batch_size, target_ratio=class_bgd_ratio)    # DataLoader for validation
    test_loader = balanced_loader2(test_dataset, batch_size=batch_size, shuffle=True)    # DataLoader for testing
    
    # Initialize model, optimizer, and load data
    #model = PretrainedPothole("ResNet101",pretrained=True, freeze_backbone=True)  # Your model class
    model = PretrainedPothole("VGG16",pretrained=True, freeze_backbone=True)  # Your model class

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4,weight_decay = 1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming you have a DataLoader `dataloader`
    # import matplotlib
    # matplotlib.use('Agg')  # Non-interactive backend
    # import matplotlib.pyplot as plt
    # from torchvision.transforms import ToPILImage

    # for batch in train_loader:
    #     images, labels = batch[0], batch[1]
    #     img = ToPILImage()(images[0])
    #     plt.imshow(img)
    #     plt.title(f"Label: {labels[0]}")
    #     plt.savefig('output_image.png')  # Save the plot as an image
    #     print("Image saved as output_image.png")
    #     break
    best_model_path = r"/dtu/blackhole/0d/203501/Data/IDLCV/best_model_VGG16.pth"

    # Train the model
    train(model, optimizer, scheduler, epochs=25, train_loader=train_loader, test_loader=val_loader, device=device, save_path=best_model_path)

