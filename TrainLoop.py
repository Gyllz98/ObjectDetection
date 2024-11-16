import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb

def train(model, optimizer, scheduler, epochs, train_loader, test_loader, device, save_path="model.pth"):
    # Move model to the specified device
    model.to(device)

    # Create save folder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set up wandb for reporting
    wandb.init(
        project="IDLCV",
        config={
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": "GT_BS",
            "dataset": "Potholes",
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "transform": "transform",
            "optimizer": optimizer.__class__.__name__,
            "loss_fn": "BCE",
            "nfft": ""
        }
    )

    best_test_loss = float('inf')  # Track the best test loss to save the model

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # Forward pass
            outputs = model(inputs)
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)  # Squeeze only the channel dimension
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

        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%")

        # Evaluation phase
        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)  # Squeeze only the channel dimension
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
        print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

        # Update the learning rate using the scheduler
        scheduler.step(avg_test_loss)  # Adjust based on test loss (for ReduceLROnPlateau)

        # Save the best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Test Loss: {avg_test_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    wandb.finish()