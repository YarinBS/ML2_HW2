import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import winsound

# --- Hyper Parameters ---
BATCH_SIZE = 500
EPOCHS = 15


def fetch_CIFAR10_data():
    """
    Fetching PyTorch's CIFAR10 dataset.
    :return: train/test dataset/loader
    """
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data/',
                                     train=True,
                                     transform=transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='./data/',
                                    train=False,
                                    transform=transform,
                                    download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # CIFAR10 images are 32*32
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)    # in_dimension = 32*32, out_dimension = 32*32
        self.pool1 = nn.MaxPool2d(2, 2)                           # in_dimension = 32*32, out_dimension = 16*16
        self.conv2 = nn.Conv2d(9, 12, kernel_size=3, padding=1)   # in_dimension = 16*16, out_dimension = 16*16
        self.pool2 = nn.MaxPool2d(2, 2)                           # in_dimension = 16*16, out_dimension = 8*8
        self.conv3 = nn.Conv2d(12, 15, kernel_size=3, padding=1)  # in_dimension = 8*8, out_dimension = 8*8
        self.pool3 = nn.MaxPool2d(2, 2)                           # in_dimension = 8*8, out_dimension = 4*4
        self.conv4 = nn.Conv2d(15, 20, kernel_size=3, padding=1)  # in_dimension = 4*4, out_dimension = 4*4
        self.fc1 = nn.Linear(20*4*4, 100)
        self.fc2 = nn.Linear(100, 75)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(75, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 20*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.logsoftmax(x)


def main():
    # Fetching CIFAR10 dataset
    train_dataset, train_loader, test_dataset, test_loader = fetch_CIFAR10_data()

    cnn = ConvNet()

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters())

    print(f'Number of parameters: {sum(param.numel() for param in cnn.parameters())}')

    # Train the Model
    # cnn.train()
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}/{EPOCHS}...", end=' ')
        correct, total = 0, 0
        for j, (train_images, train_labels) in enumerate(train_loader):
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            train_outputs = cnn(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()

            total += train_labels.size(0)
            correct += (train_predictions == train_labels).sum().item()

        print(f"Accuracy: {round(100 * (correct / total), 3)}%")

    # cnn.eval()
    correct, total = 0, 0
    for (test_images, test_labels) in test_loader:
        test_outputs = cnn(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        total += test_labels.size(0)
        correct += (test_predictions == test_labels).sum().item()

    print(f"Test set accuracy: {round(100 * correct / total, 3)}%")

    winsound.Beep(700, 2000)


if __name__ == '__main__':
    main()
