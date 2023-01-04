import torch
from hw2_206230021_q1_train import fetch_CIFAR10_data


def evaluate_model_q1() -> float:
    """
    Loads CIFAR10's test set and the trained CNN network, and returns the average error on the test set
    :return: Average error rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_data, test_loader = fetch_CIFAR10_data()
    model = torch.load("q1_model.pkl")

    total, correct = 0, 0
    for (test_images, test_labels) in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)
        test_predictions = torch.argmax(test_outputs, dim=1)
        total += test_labels.size(0)
        correct += (test_predictions == test_labels).sum().item()

    error = 100 * (1 - (correct / total))
    print(f"{round(error, 3)}")
    return error


def main():
    evaluate_model_q1()


if __name__ == '__main__':
    main()
