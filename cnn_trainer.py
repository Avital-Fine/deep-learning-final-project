import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class CNNImageClassifier(nn.Module):
    """
    A generic convolutional neural network (CNN) image classifier.

    Args:
        num_classes (int): Number of output classes for classification.
        norm_layer (nn.Module, optional): Normalization layer to use after each convolution. 
            Defaults to nn.BatchNorm2d.

    Architecture:
        - Three convolutional layers with increasing channel sizes (64, 128, 256), each followed by:
            - Normalization (default: BatchNorm2d)
            - ReLU activation
            - Max pooling (after first and second conv layers)
        - Adaptive average pooling to reduce spatial dimensions to 1x1.
        - Classifier head:
            - Flatten
            - Linear layer (256 -> 128)
            - ReLU activation
            - Dropout (p=0.5)
            - Linear layer (128 -> num_classes)

    Forward Pass:
        Input image tensor is passed through feature extractor and classifier to produce logits for each class.

    Example:
        model = CNNImageClassifier(num_classes=10)
        output = model(torch.randn(8, 3, 32, 32))  # batch of 8 RGB images, 32x32 pixels
    """
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d):
        super(CNNImageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, early_stop_acc=None):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.2%}")
        
        epoch_end = time.time()
        print(f"Epoch Time: {(epoch_end - epoch_start):.2f} sec")

        if early_stop_acc and epoch_val_accuracy >= early_stop_acc:
            print("Early stopping criteria met. Stopping training.")
            break

    return train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {model.__class__.__name__}")
    print(f"#Params: {num_params}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


# ------------------------ Main Training ------------------------
def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    data_dir = 'chest_xray'

    # Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    # Model, criterion, optimizer
    num_classes = len(train_dataset.classes)
    model = CNNImageClassifier(num_classes=num_classes)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    start_time = time.time()

    # Train
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs
    )

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes")

    # Evaluate on test set
    evaluate_model(model, test_loader, device)

    # Plot results
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()