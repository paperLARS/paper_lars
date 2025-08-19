import os
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT_PATH = 'datasetRGB/'
MODEL_PATH = 'runs/classify/train/weights/best.pt'
RESULTS_DIR = 'yolov8_benchmark_results'

def load_model():
    return YOLO('yolov8s-cls.pt')
    
def evaluate_and_save_benchmark(y_true, y_pred_probs, output_file, threshold=0.5, class_names=None):
    """Calculates, prints, and saves evaluation metrics to a text file."""
    # Ensure there are samples to evaluate
    if not y_true:
        print(f"Skipping evaluation due to no samples in {output_file}")
        return {}

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)

    y_pred_binary = [1 if p > threshold else 0 for p in y_pred_probs]
    cm = confusion_matrix(y_true, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
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

def test_benchmark_repeated(model, benchmark_folder, thr, num_repetitions=5, split_percentage=0.5):
    """
    Performs repeated testing on random subsets of the benchmark folder.
    """
    all_metrics = defaultdict(list)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    # Get all image paths from the benchmark folder, filtering by valid extensions
    img_paths = [
        os.path.join(dp, f) 
        for dp, _, fn in os.walk(benchmark_folder) 
        for f in fn 
        if f.lower().endswith(valid_extensions) and os.path.exists(os.path.join(dp, f))
    ]

    num_samples = len(img_paths)
    subset_size = int(split_percentage * num_samples)

    if subset_size == 0:
        print(f"Warning: No images found or subset size is zero in {benchmark_folder}. Skipping.")
        return all_metrics

    # Get class names from the model
    class_names = model.names
    
    for i in range(num_repetitions):
        #print(f"\n--- Repetition {i+1} ---")
        # Randomly select a subset of images
        subset_paths = np.random.choice(img_paths, size=subset_size, replace=False)
        results = model.predict(list(subset_paths), verbose=False)
        
        y_true = []
        y_pred_probs = []

        for r in results:
            original_path = r.path
            true_label = os.path.basename(os.path.dirname(original_path))
            try:
                true_label_id = list(class_names.keys())[list(class_names.values()).index(true_label)]
            except ValueError:
                print(f"Warning: Class label '{true_label}' not found in the model. Skipping the image {original_path}.")
                continue
                
            y_true.append(true_label_id)

            if r.probs.data.shape[0] > 1:
                y_pred_probs.append(r.probs.data[1].item())
            else:
                y_pred_probs.append(0.0 if r.probs.data[0].item() > 0.5 else 1.0)
                print(f"Warning: Only one class was predicted for the image {original_path}.")
            
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        y_pred_binary = (y_pred_probs > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        metrics = {
            'AUC': roc_auc,
            'Accuracy': accuracy,
        }

        # Handle potential division by zero for precision/recall
        if (TP + FP) > 0: metrics['Precision (Class 1)'] = TP / (TP + FP)
        if (TP + FN) > 0: metrics['Recall (Class 1)'] = TP / (TP + FN)
        if (TN + FN) > 0: metrics['Precision (Class 0)'] = TN / (TN + FN)
        if (TN + FP) > 0: metrics['Recall (Class 0)'] = TN / (TN + FP)

        for metric, value in metrics.items():
            all_metrics[metric].append(value)
            
        #print("Repetition Metrics:")
        #for metric, value in metrics.items():
        #    print(f"  {metric}: {value:.4f}")

    return all_metrics
    
def main():
    model = load_model()
    print("Starting training...")
    start_time = time.time()
    
    model.train(
        data='../paper_larc/datasetRGB/',
        epochs=500,
        patience=20,
        imgsz=224,
        batch=32
    )
    model = YOLO('/data/user/runs/classify/train/weights/best.pt')

    end_time = time.time()
    duration = end_time - start_time
    print("Training complete.")
    print(f"Total training time: {duration:.2f} seconds")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)  
    print("🚀 Starting repeated benchmark evaluation with YOLOv8...")
    
    all_benchmark_repeated_results = {}
    benchmark_test_path = os.path.join(ROOT_PATH, 'test')
    
    # Iterate over each subfolder in the 'test' directory
    for benchmark_name in sorted(os.listdir(benchmark_test_path)):
        benchmark_folder = os.path.join(benchmark_test_path, benchmark_name)
        if os.path.isdir(benchmark_folder):
            print(f"\nEvaluating on benchmark: {benchmark_name}")
            
            repeated_metrics = test_benchmark_repeated(
                model, benchmark_folder, thr=0.5, num_repetitions=5
            )
            all_benchmark_repeated_results[benchmark_name] = repeated_metrics

            output_file = os.path.join(RESULTS_DIR, f"{benchmark_name}_repeated_results.txt")
            with open(output_file, 'w') as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                for metric, values in repeated_metrics.items():
                    mean_val = np.mean(values)
                    std_dev = np.std(values)
                    f.write(f"Mean {metric}: {mean_val:.4f}\n")
                    f.write(f"Std Dev {metric}: {std_dev:.4f}\n")
                f.write("\n")
                
    print("\n✅ Overall Repeated Benchmark Results Summary:")
    for benchmark, metrics in all_benchmark_repeated_results.items():
        print(f"Benchmark: {benchmark}")
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_dev = np.std(values)
            print(f"  Mean {metric}: {mean_val:.4f}")
            print(f"  Std Dev {metric}: {std_dev:.4f}")

if __name__ == '__main__':
    main()