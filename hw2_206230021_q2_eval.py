import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from sys import exit
import string
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from hw2_206230021_q2_train import MyRNN, samples

# --- Hyper-parameters ---
letters_and_punctuation = string.ascii_letters + " .,;'-"
number_of_symbols = len(letters_and_punctuation) + 1  # Plus EOS marker

all_categories = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian',
                  'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
number_of_categories = len(all_categories)


def evaluate_model_q2(language='German', letter='G'):
    """
    Given a source country and a first letter, generates a name using Q2's model.
    :return: Generated name
    """

    model = MyRNN(number_of_symbols, 128, number_of_symbols)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('q2_model.pkl'))
    else:
        model.load_state_dict(torch.load('q2_model.pkl', map_location=torch.device('cpu')))

    print(f'Language: {language}, name: ', end='')
    name = samples(model, language, all_categories, number_of_categories, letter)

    return name


def main():
    _ = evaluate_model_q2('German', random.choice(string.ascii_letters).upper())
    _ = evaluate_model_q2('Russian', random.choice(string.ascii_letters).upper())
    _ = evaluate_model_q2('Japanese', random.choice(string.ascii_letters).upper())
    _ = evaluate_model_q2('Spanish', random.choice(string.ascii_letters).upper())
    _ = evaluate_model_q2('Arabic', random.choice(string.ascii_letters).upper())


if __name__ == '__main__':
    main()
