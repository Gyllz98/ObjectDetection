import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, optimizer, epochs, train_loader, test_loader, device):
    # Move model to the specified device
    model.to(device)

    # Initialize lists to track metrics over epochs
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device).float()

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = F.binary_cross_entropy_with_logits(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy for this batch
            predicted = torch.sigmoid(outputs) >= 0.5  # Convert logits to binary predictions
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation phase
        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch['input'].to(device), batch['label'].to(device).float()
                outputs = model(inputs).squeeze()
                loss = F.binary_cross_entropy_with_logits(outputs, labels)

                # Accumulate test loss
                running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.sigmoid(outputs) >= 0.5
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        # Calculate average test loss and accuracy for the epoch
        avg_test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        # Print metrics for the current epoch
        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    # Plot the loss and accuracy over epochs
    epochs_range = range(1, epochs + 1)

    # Plot for Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Training Loss", color='blue')
    plt.plot(epochs_range, test_losses, label="Test Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Plot for Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy", color='blue')
    plt.plot(epochs_range, test_accuracies, label="Test Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()
