import os
import torch
from vit_model import ViT, create_pretrained_vit
from data_utils import get_train_val_loaders, get_benchmark_paths
from train_eval import train_model, evaluate_benchmarks_with_resampling

# --- Configuration for the entire project ---
CONFIG = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "num_layers": 12,
    "num_heads": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "dropout_rate": 0.148,
    "lr": 4.104e-5,
    "num_epochs": 500,
    "batch_size": 32,
    "dataset_path": "./datasetRGB/",
    "model_path": "./vit_best_model.pth",
    "class_names": ["NIL", "POD"],
    "num_resamples": 5,
    "resample_size": 0.5
}

# Dynamically calculate num_patches
CONFIG["num_patches"] = (CONFIG["image_size"] // CONFIG["patch_size"]) ** 2
CONFIG["projection_dim"] = CONFIG["patch_size"] * CONFIG["patch_size"] * CONFIG["num_channels"]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader = get_train_val_loaders(CONFIG["dataset_path"], CONFIG)
    benchmark_paths = get_benchmark_paths(CONFIG["dataset_path"])
    
    #model = create_pretrained_vit(CONFIG, model_name="vit_base_patch16_224")
    model = ViT(CONFIG)
    model.to(device)
    train_model(model, train_loader, valid_loader, CONFIG, device)

    evaluate_benchmarks_with_resampling(
        model, 
        benchmark_paths, 
        CONFIG, 
        device, 
        num_resamples=CONFIG["num_resamples"],
    )