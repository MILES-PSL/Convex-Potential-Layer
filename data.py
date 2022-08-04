import sys

sys.path.append('./cifar10-fast')
from core import *
import os
from torch_backend import *
from torchvision.transforms import RandomCrop


class DataClass:
    def __init__(self, dataset="c10", batch_size=256):
        self.dataset = dataset
        self.batch_size = batch_size
        print('Using batch size:', batch_size)

    def __call__(self):
        DATA_DIR = './data'
        if self.dataset == "c10":
            dataset = cifar10(DATA_DIR)
        elif self.dataset == "c100":
            dataset = cifar100(DATA_DIR)
        t = Timer()
        print('Preprocessing training data')

        train_set = list(zip(transpose(pad(dataset['train']['data'], 4)) / 255.0, dataset['train']['labels']))
        print(f'Finished in {t():.2} seconds')
        print('Preprocessing test data')
        test_set = list(zip(transpose(dataset['test']['data']) / 255.0, dataset['test']['labels']))
        print(f'Finished in {t():.2} seconds')

        train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])

        train_batches = Batches(train_set_x, self.batch_size, shuffle=True, set_random_choices=True, num_workers=20)
        test_batches = Batches(test_set, 256, shuffle=False, num_workers=20)

        return train_batches, test_batches
