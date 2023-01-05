import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from sys import exit
import random
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Hyper Parameters ---
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01
GAMMA = 0.64


def fetch_CIFAR10_data():
    """
    Fetching PyTorch's CIFAR10 dataset.
    """
    train_augmentations = transforms.Compose([
        transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615))
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615))
    ])

    train_manual_augmentations_color = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomInvert(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data/',
                                     train=True,
                                     transform=transform,
                                     download=True)

    augmented_train_dataset = datasets.CIFAR10(root='./data/',
                                               train=True,
                                               transform=train_augmentations,
                                               download=True)

    manual_color_augments_train_dataset = datasets.CIFAR10(root='./data/',
                                                           train=True,
                                                           transform=train_manual_augmentations_color,
                                                           download=True)

    test_dataset = datasets.CIFAR10(root='./data/',
                                    train=False,
                                    transform=transform,
                                    download=True)

    full_train_dataset = torch.utils.data.ConcatDataset([
        train_dataset,
        augmented_train_dataset,
        manual_color_augments_train_dataset,
    ])

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=full_train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return full_train_dataset, train_loader, test_dataset, test_loader


def plot_convergence(epochs: int, train_list: list, test_list: list, mode: str) -> None:
    plt.plot(range(1, epochs + 1), train_list)
    plt.plot(range(1, epochs + 1), test_list)
    plt.legend(['Train', 'Test'])
    plt.xlabel("Epochs")
    plt.ylabel(f"{mode.capitalize()}")
    plt.title(f"{mode.capitalize()} over epochs")
    plt.savefig(f"./plots/{mode.capitalize()} over epochs.png")
    plt.show()


def train_model_q1():
    """
    Trains Q1's model and produces the convergence graphs
    :return: Model instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Fetching CIFAR10 dataset
    train_dataset, train_loader, test_dataset, test_loader = fetch_CIFAR10_data()

    # Initializing the model
    cnn = MyConvNet()
    cnn.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    # Check number of parameters
    params = sum(param.numel() for param in cnn.parameters())
    if params > 50000:
        print(f"Too many parameters: {params} > 50000")
        exit(0)
    else:
        print(f'Number of parameters: {params}')

    prev_loss = 0
    loss_counter = 0
    acc_counter = 0

    train_losses, train_errors = [], []
    test_losses, test_errors = [], []

    # Train the Model
    cnn.train()
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}/{EPOCHS}:", end=' ')

        correct, total = 0, 0
        cumulative_train_loss = 0
        for j, (train_images, train_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            train_outputs = cnn(train_images)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()

            # Calculate epoch's train accuracy
            train_predictions = torch.argmax(train_outputs, dim=1)
            total += train_labels.size(0)
            correct += (train_predictions == train_labels).sum().item()

            # Calculate batch's train loss
            cumulative_train_loss += loss.item()

        train_epoch_loss = cumulative_train_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        print(f"Train accuracy - {round(100 * (correct / total), 3)}%", end=', ')
        train_errors.append(1 - (correct / total))

        correct, total = 0, 0
        cnn.eval()
        cumulative_eval_loss = 0
        for (test_images, test_labels) in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = cnn(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            total += test_labels.size(0)
            correct += (test_predictions == test_labels).sum().item()

            # Calculate eval loss
            cumulative_eval_loss += criterion(test_outputs, test_labels)

        print(f"Evaluation accuracy - {round(100 * correct / total, 3)}%")
        test_errors.append(1 - (correct / total))

        eval_loss = cumulative_eval_loss / len(test_loader)
        test_losses.append(eval_loss.item())

        acc_counter += 1

        if i > EPOCHS - 11:  # For the last 10 epochs, update the learning rate after every iteration
            print("Final epochs - adjusting learning rate...")
            scheduler.step()
            continue

        if acc_counter == 20:
            print("20 consecutive epochs without updates, adjusting learning rate...")
            scheduler.step()
            acc_counter = 0
            continue

        if i != 0:
            if eval_loss >= prev_loss:
                loss_counter += 1
                prev_loss = eval_loss
                if loss_counter == 3:
                    print("Loss increased for 3 consecutive epochs, adjusting learning rate...")
                    scheduler.step()
                    loss_counter = 0
                    acc_counter = 0
                    continue
            else:
                prev_loss = eval_loss
                loss_counter = 0

    # Final evaluation
    print("Final accuracy: ", end=' ')
    correct, total = 0, 0
    cnn.eval()
    for (test_images, test_labels) in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = cnn(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        total += test_labels.size(0)
        correct += (test_predictions == test_labels).sum().item()

    print(f"{round(100 * correct / total, 3)}%")

    # Print graphs
    plot_convergence(epochs=EPOCHS, train_list=train_errors, test_list=test_errors, mode="error")
    plot_convergence(epochs=EPOCHS, train_list=train_losses, test_list=test_losses, mode="loss")

    return cnn


class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        # CIFAR10 images are 32*32
        # Number of parameters in a Conv layer:
        # (width of the filter * height of the filter * number of filters in the previous layer+1)*number of filters

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # in_dimension = 32*32, out_dimension = 32*32
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # in_dimension = 16*16, out_dimension = 16*16
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # in_dimension = 8*8, out_dimension = 8*8
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1024, 10)  # in_dimension = 4*4
        # self.fc2 = nn.Linear(300, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.batchnorm1(self.pool(F.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool(F.relu(self.conv2(x))))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)  # flatten() and view() are the same!
        x = self.dropout(x)
        x = self.fc1(x)
        return self.logsoftmax(x)


def main():
    # --- Seed for reproducibility ------
    seed = 1
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ------------------------------------

    # Get trained model
    cnn = train_model_q1()

    # Save model
    torch.save(cnn.state_dict(), "q1_model.pkl")


if __name__ == '__main__':
    main()
