import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler

def test_benchmark(test_loader, model):
    """
    Performs a single test run and returns true labels and predicted probabilities.
    """
    device = torch.device('cpu')
    model = model.to(device)
    y_pred_probs = []
    y_true = []
    model.eval() 
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1] # Get probability of the positive class
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(probabilities.cpu().numpy())
    return y_true, y_pred_probs

def evaluate_and_save_benchmark(y_true, y_pred_probs, output_file, threshold=0.5, class_names=None):
    """Calculates, prints, and saves evaluation metrics to a text file."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)

    y_pred_binary = [1 if p > threshold else 0 for p in y_pred_probs]
    cm = confusion_matrix(y_true, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision_class_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_class_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision_class_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0

    metrics = {
        'AUC': roc_auc,
        'Accuracy': accuracy,
        f'Precision (Class {class_names[0] if class_names else 0})': precision_class_0,
        f'Recall    (Class {class_names[0] if class_names else 0})': recall_class_0,
        f'Precision (Class {class_names[1] if class_names else 1})': precision_class_1,
        f'Recall    (Class {class_names[1] if class_names else 1})': recall_class_1,
        'Threshold': threshold
    }

    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics

def test_benchmark_repeated(test_dataset, model, thr, num_repetitions=5, split_percentage=0.5, batch_size=32):
    """
    Performs repeated testing on random subsets of the test dataset.
    """
    device = torch.device('cpu')
    model = model.to(device).eval()
    all_metrics = defaultdict(list)

    num_samples = len(test_dataset)
    subset_size = int(split_percentage * num_samples)

    for i in range(num_repetitions):
        #print(f"\n--- Repetition {i+1} ---")
        indices = torch.randperm(num_samples).tolist()
        subset_indices = indices[:subset_size]
        subset_sampler = SubsetRandomSampler(subset_indices)
        subset_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=subset_sampler)

        y_pred_probs = []
        y_true = []

        with torch.no_grad():
            for images, labels in subset_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(probabilities.cpu().numpy())

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = roc_auc_score(y_true, y_pred_probs)

        y_pred_binary = [1 if p > thr else 0 for p in y_pred_probs]
        cm = confusion_matrix(y_true, y_pred_binary)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision_class_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        recall_class_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision_class_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0

        metrics = {
            'AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision (Class 0)': precision_class_0,
            'Recall (Class 0)': recall_class_0,
            'Precision (Class 1)': precision_class_1,
            'Recall (Class 1)': recall_class_1,
        }

        for metric, value in metrics.items():
            all_metrics[metric].append(value)

        #print("Repetition Metrics:")
        #for metric, value in metrics.items():
        #    print(f"  {metric}: {value:.4f}")

    return all_metrics