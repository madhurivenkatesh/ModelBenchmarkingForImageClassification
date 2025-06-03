import time
import psutil
import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import os
from torch.utils.data import DataLoader

NUM_CLASSES = 2
DATA_DIR = "dataset_fire"
LEARNING_RATE = 0.001
MODEL_DIR = "."

def setup_model_and_benchmark(model_path):

    model_name = os.path.basename(model_path)

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Identify model architecture from filename
    if "resnet_18" in model_name:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif "mobilenet_v2" in model_name:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    elif "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif "squeezenet1_1" in model_name:
        model = models.squeezenet1_1(pretrained=False)
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model.to(device)

    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Benchmarking
    start_time = time.time()
    inference_times = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_start = time.time()
            outputs = model(inputs)
            batch_end = time.time()

            inference_times.append(batch_end - batch_start)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()

    # Metrics
    val_accuracy = correct / total
    total_inference_time = end_time - start_time
    avg_batch_inference_time = sum(inference_times) / len(inference_times)
    memory_used = psutil.virtual_memory().percent
    cpu_used = psutil.cpu_percent(interval=1)

    # Results
    print("Performance for model:", model_name)
    print(f"Size on disk: {os.path.getsize(model_path)/(1024*1024):.2f} MB")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Total Inference Time: {total_inference_time:.2f} seconds")
    print(f"Average Batch Inference Time: {avg_batch_inference_time:.4f} seconds")
    print(f"Memory Used: {memory_used}%")
    print(f"CPU Used: {cpu_used}%\n")


# Run benchmarking on all .pth models in directory
for filename in os.listdir(MODEL_DIR):
    if filename.lower().endswith(".pth"):
        full_path = os.path.join(MODEL_DIR, filename)
        setup_model_and_benchmark(full_path)
