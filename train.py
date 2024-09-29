import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import UNet
from dataloader import get_train_val_loaders
from utils import save_model, load_model, calculate_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 2
CHECKPOINT_PATH = "checkpoints/unet_checkpoint.pth"

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(loader)

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    metrics = {"accuracy": 0, "iou": 0}

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Calculate metrics (accuracy, IoU, etc.)
            batch_metrics = calculate_metrics(predictions, targets)
            metrics = {k: metrics[k] + batch_metrics[k] for k in metrics}

    metrics = {k: metrics[k] / len(loader) for k in metrics}
    return total_loss / len(loader), metrics

def plot_and_save_graph(train_losses, val_losses, accuracies, ious, epochs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='blue')
    plt.plot(epochs, ious, label='IoU', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Accuracy and IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

def main():
    model = UNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader, test_loader = get_train_val_loaders(BATCH_SIZE)

    if load_model(model, optimizer, CHECKPOINT_PATH):
        print("Checkpoint loaded.")

    train_losses, val_losses, accuracies, ious = [], [], [], []
    epochs = []

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, metrics = validate(model, val_loader, loss_fn)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, IoU: {metrics['iou']:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(metrics['accuracy'])
        ious.append(metrics['iou'])
        epochs.append(epoch + 1)

        save_model(model, optimizer, epoch, CHECKPOINT_PATH)

    plot_and_save_graph(train_losses, val_losses, accuracies, ious, epochs)

if __name__ == "__main__":
    main()
