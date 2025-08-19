### Enhancing Maritime Surveillance: A Multi-Model Analysis on a Curated Multi-Dataset Benchmark

This repository contains the source code for the paper **"Enhancing Maritime Search and Rescue: A Comparative Analysis of CNNs, ViTs, and YOLO on a Curated Multi-Dataset Benchmark,"** submitted to the **24th IEEE Latin American Robotics Competition & 22nd IEEE Latin American Robotics Symposium (2025).**

The project provides a comparative analysis of three major deep learning architectures for binary classification in maritime search and rescue operations. It includes code to train and evaluate each model both from scratch and using a pre-trained approach.

---

### 📂 Repository Structure

The repository is organized into three main directories, one for each model architecture evaluated in the paper.

```
.
├── CNN/
│   ├── main.py
│   ├── train.py
│   ├── model.py
│   └── ...
├── ViT/
│   ├── main.py
│   ├── train.py
│   ├── vit_model.py
│   └── ...
├── YOLO/
│   ├── main.py
│   └── ...
└── README.md
```

Each folder (`CNN`, `ViT`, and `YOLO`) contains the necessary scripts to:

- **Train a model from scratch:** Trains the model on the curated benchmark dataset without using pre-trained weights.
    
- **Train a pre-trained model:** Fine-tunes a pre-existing model (e.g., ResNet-18, ViT-B/16, YOLOv8n) on the curated dataset.
    
- **Evaluate a trained model:** Assesses a model's performance on the five distinct test subsets used in the paper.
    

---

### 📦 Dataset

The curated dataset used for this project is a combination of images from three public datasets: AFO, Singapore, and a portion of Seagull.

- The full dataset, excluding images from the Seagull dataset, can be downloaded from: **[https://huggingface.co/datasets/paperlars/Dataset_LARS_paper]**
    

**Note on the Seagull Dataset:** Due to licensing and redistribution restrictions, the complete Seagull dataset is not included in this repository. The images are only available upon a specific request to the dataset authors, and their licensing agreement does not permit us to share the data.

The models in this repository were trained and evaluated on the full curated benchmark dataset, which includes the complete AFO, Singapore, and Seagull datasets. Consequently, readers who do not have access to the full Seagull dataset may not be able to reproduce the exact performance metrics reported in our paper.

---

### 🚀 Getting Started

To replicate the experiments from the paper, follow these steps:

1. Clone this repository: `git clone [repository URL]`
    
2. Download and place the dataset in the `data/` directory.
    
3. Install the required dependencies listed in each model's respective directory.
    
4. Navigate to the desired model's folder (e.g., `ViT/`).
    
5. Run the training and evaluation scripts. Instructions for running each script can be found in the comments within the code.
