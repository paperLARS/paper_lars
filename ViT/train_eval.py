import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_utils import CustomDataset, get_eval_transforms
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import time 

def train_model(model, train_loader, valid_loader, config, device):
    """
    Handles the training loop for the Vision Transformer model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    
    best_valid_loss = float('inf')
    #best_valid_acc = 0.0
    early_stopping_patience = 20
    patience_counter = 0

    print("Starting training...")
    
    # Start the timer
    start_time = time.time()
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Training]"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        valid_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Validation]"):
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                total_predictions += labels.size(0)
                correct_predictions += (predictions.squeeze() == labels.squeeze()).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = correct_predictions / total_predictions
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}")
        
        if valid_loss < best_valid_loss:
        #if valid_accuracy > best_valid_acc:
            best_valid_loss = valid_loss
            #best_valid_acc = valid_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), config["model_path"])
            print(f"Saved best model to {config['model_path']}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # End the timer
    end_time = time.time()
    duration = end_time - start_time
    
    print("Training complete.")
    print(f"Total training time: {duration:.2f} seconds")

def evaluate_benchmarks_with_resampling(model, benchmark_paths, config, device, num_resamples=5, test_size=0.5):
    """
    Evaluates the model on multiple random resamples of each benchmark,
    calculates various metrics, and saves the results to a text file.
    """
    print("\nStarting evaluation on test benchmarks with resampling...")
    
    model.load_state_dict(torch.load(config["model_path"]))
    model.to(device)
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    eval_transforms = get_eval_transforms(config)

    results_dir = os.path.join(config["dataset_path"], "results")
    os.makedirs(results_dir, exist_ok=True)

    for benchmark_name, all_images in benchmark_paths.items():
        print(f"\n--- Evaluating benchmark: {benchmark_name} ({num_resamples} resamples) ---")
        
        if not all_images:
            print(f"Benchmark {benchmark_name} has no images. Skipping.")
            continue

        class_images = {}
        for path in all_images:
            label = os.path.basename(os.path.dirname(path))
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(path)

        resample_accuracies = []
        resample_losses = []
        resample_aucs = []
        resample_precisions_0 = []
        resample_recalls_0 = []
        resample_precisions_1 = []
        resample_recalls_1 = []

        for i in range(num_resamples):
            print(f"  > Resampling run {i+1}/{num_resamples}")
            
            resampled_paths = []
            for label in class_images:
                num_to_sample = int(len(class_images[label]) * test_size)
                resampled_paths.extend(np.random.choice(class_images[label], num_to_sample, replace=False))
            
            resample_dataset = CustomDataset(resampled_paths, config, transform=eval_transforms)
            resample_loader = DataLoader(resample_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

            true_labels = []
            predicted_labels = []
            predicted_logits = []

            with torch.no_grad():
                for images, labels in resample_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    
                    true_labels.extend(labels.cpu().numpy())
                    predicted_logits.extend(outputs.cpu().squeeze().numpy())
                    predicted = (torch.sigmoid(outputs) > 0.5).cpu().squeeze().numpy().astype(int)
                    predicted_labels.extend(predicted)

            if len(true_labels) > 0:
                # Calculate metrics for this resample
                accuracy = accuracy_score(true_labels, predicted_labels)
                auc = roc_auc_score(true_labels, predicted_logits)
                
                precision_0 = precision_score(true_labels, predicted_labels, pos_label=0)
                recall_0 = recall_score(true_labels, predicted_labels, pos_label=0)
                precision_1 = precision_score(true_labels, predicted_labels, pos_label=1)
                recall_1 = recall_score(true_labels, predicted_labels, pos_label=1)

                resample_accuracies.append(accuracy)
                resample_aucs.append(auc)
                resample_precisions_0.append(precision_0)
                resample_recalls_0.append(recall_0)
                resample_precisions_1.append(precision_1)
                resample_recalls_1.append(recall_1)

            else:
                print(f"Resample {i+1} has no data. Skipping metric calculation.")
        
        if resample_accuracies:
            mean_accuracy = np.mean(resample_accuracies)
            std_accuracy = np.std(resample_accuracies)
            mean_auc = np.mean(resample_aucs)
            std_auc = np.std(resample_aucs)
            
            mean_precision_0 = np.mean(resample_precisions_0)
            std_precision_0 = np.std(resample_precisions_0)
            mean_recall_0 = np.mean(resample_recalls_0)
            std_recall_0 = np.std(resample_recalls_0)

            mean_precision_1 = np.mean(resample_precisions_1)
            std_precision_1 = np.std(resample_precisions_1)
            mean_recall_1 = np.mean(resample_recalls_1)
            std_recall_1 = np.std(resample_recalls_1)
            
            file_path = os.path.join(results_dir, f"{benchmark_name}_results.txt")
            with open(file_path, "w") as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Mean AUC: {mean_auc:.4f}\n")
                f.write(f"Std Dev AUC: {std_auc:.4f}\n")
                f.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
                f.write(f"Std Dev Accuracy: {std_accuracy:.4f}\n")
                f.write(f"Mean Precision ({config['class_names'][0]}): {mean_precision_0:.4f}\n")
                f.write(f"Std Dev Precision ({config['class_names'][0]}): {std_precision_0:.4f}\n")
                f.write(f"Mean Recall ({config['class_names'][0]}): {mean_recall_0:.4f}\n")
                f.write(f"Std Dev Recall ({config['class_names'][0]}): {std_recall_0:.4f}\n")
                f.write(f"Mean Precision ({config['class_names'][1]}): {mean_precision_1:.4f}\n")
                f.write(f"Std Dev Precision ({config['class_names'][1]}): {std_precision_1:.4f}\n")
                f.write(f"Mean Recall ({config['class_names'][1]}): {mean_recall_1:.4f}\n")
                f.write(f"Std Dev Recall ({config['class_names'][1]}): {std_recall_1:.4f}\n")
            
            print(f"Benchmark {benchmark_name} results saved to {file_path}")

        else:
            print(f"Benchmark {benchmark_name} - No data to evaluate.")
