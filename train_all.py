import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------
# Configuration
# ----------------------
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = "dataset_fire"
SAVE_PATH = "model_"

models_to_evaluate = {"resnet_18", "mobilenet_v2", "efficientnet_b0", "squeezenet1_1"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Train and Save Model
# ----------------------
def train_model(model_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------- Model Selection ----------------------
    if model_name == "resnet_18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------- Training Loop ----------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"[{model_name}] Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # ---------------------- Save Trained Model ----------------------
    final_save_path = SAVE_PATH + model_name + "_test.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_save_path)
    print(f"Model saved to {final_save_path}\n")


# ----------------------
# Main Execution
# ----------------------
for model_name in models_to_evaluate:
    train_model(model_name)
