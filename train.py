import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import models 
from utils.getdata import Data 

def main():
    BATCH_SIZE = 8
    EPOCH = 15
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    aug_path = "C:/Users/HP/OneDrive/Dokumen/C++/IPSD/tugasipsd/Dataset/Augmented_Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/HP/OneDrive/Dokumen/C++/IPSD/tugasipsd/Dataset/Original Images/Original Images/FOLDS/"

    # inisialisasi dataset
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    # menggabungkan data tambahan dan data asli
    full_data = dataset.dataset_train + dataset.dataset_aug

    # 80-20 split
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # menentukan model, loss function, and optimizer
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []

    for epoch in range(EPOCH):
        model.train()
        loss_train, correct_train, total_train = 0.0, 0, 0

        for src, trg in train_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        train_losses.append(loss_train / len(train_loader))

        model.eval()
        loss_val, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for src, trg in val_loader:
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)

                loss_val += loss.item()
                _, predicted = torch.max(pred, 1)
                total_val += trg.size(0)
                correct_val += (predicted == trg).sum().item()

        accuracy_val = 100 * correct_val / total_val
        val_losses.append(loss_val / len(val_loader))

        print(f"Epoch [{epoch + 1}/{EPOCH}], "
              f"Train Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy_train:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy_val:.2f}%")

    # menyimpan model
    torch.save(model.state_dict(), "trained_modelmobilev2_t.pth")

    # plot training dan validation loss
    plt.plot(range(EPOCH), train_losses, color="#33e6c2", label='Train Loss')
    plt.plot(range(EPOCH), val_losses, color="#ff338b", label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("./train_valid_loss.png")
    plt.show()

if __name__ == "__main__":
    main()