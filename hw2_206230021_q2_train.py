from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from io import open
import unicodedata
import string
import random
import os
import matplotlib.pyplot as plt

# --- Hyper Parameters ---
learning_rate = 0.0005
EPOCHS = 250000
max_length = 20

letters_and_punctuation = string.ascii_letters + " .,;'-"
number_of_symbols = len(letters_and_punctuation) + 1  # Plus EOS marker


def read_file(filename):
    """
    Read a file and split into lines
    """

    def unicodeToAscii(s):
        """
        Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in letters_and_punctuation
        )

    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


def random_pair(all_categories, category_lines):
    """
    Get a random category and random line from that category
    """
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines[category][random.randint(0, len(category_lines[category]) - 1)]
    return category, line


def categoryTensor(category, all_categories, n_categories):
    """
    # One-hot vector for category
    """
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


def inputTensor(line):
    """
    One-hot matrix of first to last letters (not including EOS) for input
    """
    tensor = torch.zeros(len(line), 1, number_of_symbols)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][letters_and_punctuation.find(letter)] = 1
    return tensor


def targetTensor(line):
    """
    LongTensor of second letter to end (EOS) for target
    """
    letter_indexes = [letters_and_punctuation.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(number_of_symbols - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def random_example(all_categories, category_lines, n_categories):
    """
    Make category, input, and target tensors from a random category, line pair
    """
    category, line = random_pair(all_categories, category_lines)
    category_tensor = categoryTensor(category, all_categories, n_categories)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def train(model, criterion, category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.hiddenInitialization()

    model.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        current_loss = criterion(output, target_line_tensor[i])
        loss += current_loss

    loss.backward()
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def samples(rnn, category, all_categories, n_categories, start_letters='ABC'):
    """
    Get multiple samples from one category and single/multiple starting letters
    """

    def sample(model, category, all_categories, number_of_categories, letter='A'):
        """
        Sample from a category and starting letter
        """
        with torch.no_grad():  # no need to track history in sampling
            category_tensor = categoryTensor(category, all_categories, number_of_categories)
            input = inputTensor(letter)
            hidden = model.hiddenInitialization()

            output_name = letter

            for i in range(max_length):
                output, hidden = model(category_tensor, input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == number_of_symbols - 1:
                    break
                else:
                    letter = letters_and_punctuation[topi]
                    output_name += letter
                input = inputTensor(letter)

            print(output_name)
            return output_name

    uppercase_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if len(start_letters) == 1:
        return sample(rnn, category, all_categories, n_categories, start_letters)
    else:
        for start_letter in uppercase_letters:
            return sample(rnn, category, all_categories, n_categories, start_letter)


def train_model_q2():
    """
    Trains Q2's model and produces the convergence graphs
    :return: Model instance
    """
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []

    for file in os.scandir('data/names'):
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        lines = read_file(file)
        category_lines[category] = lines

    number_of_categories = len(all_categories)

    rnn_model = MyRNN(number_of_symbols, 128, number_of_symbols)
    criterion = nn.NLLLoss()

    losses = []
    total_loss = 0

    for i in range(EPOCHS):
        output, loss = train(rnn_model,
                             criterion,
                             *random_example(all_categories,
                                             category_lines,
                                             number_of_categories))
        total_loss += loss

        if (i + 1) % 1000 == 0:
            print(i + 1)
            losses.append(total_loss / 500)
            total_loss = 0

    # Plotting loss graph
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig("q2_plot.png")
    plt.show()

    print("German Sample: ")
    samples(rnn_model, 'German', all_categories, number_of_categories, random.choice(string.ascii_letters).upper())
    print("Russian Sample: ")
    samples(rnn_model, 'Russian', all_categories, number_of_categories, random.choice(string.ascii_letters).upper())
    print("Japanese Sample: ")
    samples(rnn_model, 'Japanese', all_categories, number_of_categories, random.choice(string.ascii_letters).upper())
    print("Spanish Sample: ")
    samples(rnn_model, 'Spanish', all_categories, number_of_categories, random.choice(string.ascii_letters).upper())
    print("Arabic Sample: ")
    samples(rnn_model, 'Arabic', all_categories, number_of_categories, random.choice(string.ascii_letters).upper())

    return rnn_model


class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyRNN, self).__init__()
        self.hidden_dim = hidden_dim  # Larger hidden_dim -> Longer generated names
        self.input_to_hidden = nn.Linear(18 + input_dim + hidden_dim, hidden_dim)
        self.input_to_output = nn.Linear(18 + input_dim + hidden_dim, output_dim)
        self.output_to_output = nn.Linear(hidden_dim + output_dim, output_dim)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.input_to_hidden(input_combined)
        output = self.input_to_output(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.output_to_output(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def hiddenInitialization(self):
        return torch.zeros(1, self.hidden_dim)


def main():
    rnn = train_model_q2()

    # Save model
    torch.save(rnn.state_dict(), "q2_model.pkl")


if __name__ == '__main__':
    main()
