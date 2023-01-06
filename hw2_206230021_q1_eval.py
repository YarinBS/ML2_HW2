import torch
import random
import os
import numpy as np
from hw2_206230021_q1_train import fetch_CIFAR10_data, MyConvNet


def evaluate_model_q1():
    """
    Loads CIFAR10's test set and the trained CNN network, and returns the average error on the test set
    :return: Average error rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    _, _, test_data, test_loader = fetch_CIFAR10_data()

    model = MyConvNet()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('q1_model.pkl'))
    else:
        model.load_state_dict(torch.load('q1_model.pkl', map_location=torch.device('cpu')))

    model.eval()
    total, correct = 0, 0
    for (test_images, test_labels) in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        total += test_labels.size(0)
        correct += (test_predictions == test_labels).sum().item()

    acc = 100 * correct / total
    error = 100 * (1 - (correct / total))
    print(f"Accuracy: {acc}%, Error: {error}%")
    return error


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

    evaluate_model_q1()


if __name__ == '__main__':
    main()
