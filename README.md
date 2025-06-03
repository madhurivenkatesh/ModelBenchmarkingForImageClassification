# ModelBenchmarkingForImageClassification
Benchmarking models in PyTorch, YOLO etc based on Image classification
This repository contains benchmarking results and scripts for evaluating the performance of various deep learning image classification models on a custom binary fire dataset.

## 🔍 Objective

To compare classification models across:
- **Validation Accuracy**
- **Inference Time (per batch)**
- **Model Size on Disk**
- **System Resource Usage (CPU & Memory)**

The goal is to evaluate trade-offs between model size, speed, and accuracy — particularly in resource-constrained environments.

---

## 🧪 Models Benchmarked

| Model             | Size (MB) | Accuracy | Avg Batch Inference Time (s) | Params (M) | Remarks               |
|------------------|-----------|----------|-------------------------------|------------|------------------------|
| ResNet-18        | 128.05    | 98.00%   | 0.0210                        | ~11.7      | Large, accurate        |
| EfficientNet-B0  | 46.37     | 98.50%   | 0.0136                        | ~5.3       | Great balance          |
| MobileNet-V2     | 25.85     | 98.50%   | 0.0062                        | ~3.4       | Fast & accurate        |
| SqueezeNet 1.1   | 8.35      | 89.00%   | 0.0058                        | ~1.2       | Small but less accurate|

> 📝 YOLOv8-classification models will be added soon.

---

## 🗂 Dataset

- Custom fire classification dataset
- Two classes: `fire`, `no_fire`
- Structure follows ImageFolder format (`train/`, `val/` directories)
- It should work just as well for any other binary classifier

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python benchmark.py
