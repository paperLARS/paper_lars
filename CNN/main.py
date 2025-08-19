import os
import numpy as np
from train import tuning_model
from data_utils import train_dataloader, val_dataloader, test_dataloader
from test import test_benchmark_repeated, test_benchmark, evaluate_and_save_benchmark

def main():
    """
    Main function to orchestrate training and testing.
    """
    ROOT_PATH  = 'datasetRGB/'
    BATCH_SIZE = 32
    IMG_SIZE   = 224
    threshold  = 0.5
    class_names = ['NIL', 'POD']
    
    # Load data loaders for training and validation
    train_loader = train_dataloader(data_dir=ROOT_PATH+'train', batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    val_loader   = val_dataloader(data_dir=ROOT_PATH+'val', batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    
    # Train the model
    model = tuning_model(train_loader, val_loader, IMG_SIZE)
    print("Training completed!")

    results_dir = 'resnet_benchmark_results'
    os.makedirs(results_dir, exist_ok=True)

    all_benchmark_repeated_results = {}
    for benchmark_name in sorted(os.listdir(ROOT_PATH+'test')):
        benchmark_folder = os.path.join(ROOT_PATH+'test', benchmark_name)
        if os.path.isdir(benchmark_folder):
            print(f"\nEvaluating on benchmark: {benchmark_name} (Repeated Testing)")
            test_dataset = test_dataloader(benchmark_folder, BATCH_SIZE, IMG_SIZE)
            repeated_metrics = test_benchmark_repeated(
                test_dataset, model, threshold, num_repetitions=5,
                split_percentage=0.5, batch_size=BATCH_SIZE
            )
            all_benchmark_repeated_results[benchmark_name] = repeated_metrics

            output_file = os.path.join(results_dir, f"{benchmark_name}_repeated_results.txt")
            with open(output_file, 'w') as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                for metric, values in repeated_metrics.items():
                    mean_val = np.mean(values)
                    std_dev = np.std(values)
                    f.write(f"Mean {metric}: {mean_val:.4f}\n")
                    f.write(f"Std Dev {metric}: {std_dev:.4f}\n")
                f.write("\n")
                
    print("\nOverall Repeated Benchmark Results Summary:")
    for benchmark, metrics in all_benchmark_repeated_results.items():
        print(f"Benchmark: {benchmark}")
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_dev = np.std(values)
            print(f"  Mean {metric}: {mean_val:.4f}")
            print(f"  Std Dev {metric}: {std_dev:.4f}")

    
if __name__ == '__main__':
    main()