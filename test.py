import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import models
from utils.getdata import Data 
from utils.getdata import Data
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(model, data_loader):
    models.mobilenet_v2()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            outputs = model(src)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Debugging untuk memeriksa unique labels
    unique_labels = np.unique(all_labels)
    print(f"Unique labels in all_labels: {unique_labels}")

    # Validasi label, harus berada dalam range [0, NUM_CLASSES-1]
    NUM_CLASSES = 6
    if np.any(all_labels >= NUM_CLASSES) or np.any(all_labels < 0):
        raise ValueError(
            f"Invalid labels detected. Labels must be in the range [0, {NUM_CLASSES-1}]. Found: {unique_labels}"
        )

    # One-hot encoding untuk AUC
    all_labels_onehot = np.eye(NUM_CLASSES)[all_labels]
    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc, cm

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    aug_path = "C:/Users/HP/OneDrive/Dokumen/C++/IPSD/tugasipsd/Dataset/Augmented_Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/HP/OneDrive/Dokumen/C++/IPSD/tugasipsd/Dataset/Original Images/Original Images/FOLDS/"

    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v2(pretrained=True)
    model.load_state_dict(torch.load("trained_model4.pth"))
    model.eval()

    # Evaluasi model pada data test
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, test_loader)

    # Hasil evaluasi
    print("Evaluasi pada data test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualisasi Confusion Matrix sebagai Heatmap
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]
    plot_confusion_matrix(cm, class_names, save_path="./confusion_matrix.png")

if __name__ == "__main__":
    main()