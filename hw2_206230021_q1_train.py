import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Hyper Parameters ---
BATCH_SIZE = 64  # Best so far: ==
EPOCHS = 35  # Best so far: ==  # 50 epochs achieved lower accuracy on the test set


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
        # Number of parameters in a Conv layer:
        # (width of the filter * height of the filter * number of filters in the previous layer+1)*number of filters

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)  # in_dimension = 32*32, out_dimension = 28*28
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # in_dimension = 28*28, out_dimension = 28*28
        self.pool1 = nn.MaxPool2d(2, 2)  # in_dimension = 28*28, out_dimension = 14*14
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(32, 48, kernel_size=3)  # in_dimension = 14*14, out_dimension = 12*12
        self.pool2 = nn.MaxPool2d(2, 2)  # in_dimension = 12*12, out_dimension = 6*6
        self.batchnorm3 = nn.BatchNorm2d(48)

        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(48, 32, kernel_size=3, padding=1)  # in_dimension = 6*6, out_dimension = 6*6
        self.batchnorm4 = nn.BatchNorm2d(32)

        self.dropout4 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3)  # in_dimension = 6*6, out_dimension = 4*4
        self.batchnorm5 = nn.BatchNorm2d(32)

        self.dropout5 = nn.Dropout(p=0.3)

        self.conv6 = nn.Conv2d(32, 10, kernel_size=3, padding=1)  # in_dimension = 4*4, out_dimension = 4*4
        self.batchnorm6 = nn.BatchNorm2d(10)
        self.dropout6 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(10 * 4 * 4, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.dropout1(x)
        x = self.batchnorm2(self.pool1(F.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.batchnorm3(self.pool2(F.relu(self.conv3(x))))
        x = self.dropout3(x)
        x = self.batchnorm4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = self.batchnorm5(F.relu(self.conv5(x)))
        x = self.dropout5(x)
        x = self.batchnorm6(F.relu(self.conv6(x)))
        x = self.dropout6(x)
        x = x.view(-1, 10 * 4 * 4)
        x = self.fc1(x)
        return self.logsoftmax(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Setting seed for reproducibility
    torch.manual_seed(0)

    # Fetching CIFAR10 dataset
    train_dataset, train_loader, test_dataset, test_loader = fetch_CIFAR10_data()

    cnn = ConvNet()
    cnn.to(device)

    params = sum(param.numel() for param in cnn.parameters())
    if params > 50000:
        print(f"Too many params: {params} > 50000")
        exit(0)
    else:
        print(f'Number of parameters: {sum(param.numel() for param in cnn.parameters())}')

    # Train the Model
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}/{EPOCHS}...", end=' ')
        correct, total = 0, 0
        for j, (train_images, train_labels) in enumerate(train_loader):
            cnn.train()
            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(cnn.parameters(), lr=0.002)
            optimizer.zero_grad()
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            train_outputs = cnn(train_images)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()

            # Calculate epoch's accuracy
            train_predictions = torch.argmax(train_outputs, dim=1)
            total += train_labels.size(0)
            correct += (train_predictions == train_labels).sum().item()

        print(f"Accuracy: {round(100 * (correct / total), 3)}%")

    # Evaluation
    correct, total = 0, 0
    for (test_images, test_labels) in test_loader:
        cnn.eval()
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = cnn(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        total += test_labels.size(0)
        correct += (test_predictions == test_labels).sum().item()

    print(f"Test set accuracy: {round(100 * correct / total, 3)}%")


if __name__ == '__main__':
    main()
