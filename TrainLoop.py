import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, optimizer, epochs, train_loader, test_loader, device):
    # Move model to the specified device
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader):
            inputs, labels =  batch[0].to(device), batch[1].to(device)

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
