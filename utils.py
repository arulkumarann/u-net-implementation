import torch
import os
def save_model(model, optimizer, epoch, file_path):
    """Saves the model state, optimizer state, and current epoch."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, file_path)
    print(f"Model saved at {file_path}")

def load_model(model, optimizer, file_path):
    """Loads the model and optimizer states from a checkpoint."""
    try:
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}")
        return True
    except FileNotFoundError:
        print(f"No checkpoint found at {file_path}. Starting from scratch.")
        return False

def calculate_metrics(predictions, targets):
    """Calculate accuracy, IoU, or other metrics."""
    _, preds = torch.max(predictions, 1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    intersection = (preds & targets).sum((1, 2))
    union = (preds | targets).sum((1, 2))
    iou = (intersection / union).mean().item()

    return {"accuracy": accuracy, "iou": iou}
