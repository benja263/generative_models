"""
Script for training a MADE model on the MNIST dataset
"""
import argparse
from pathlib import Path

import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import AdamW

from MADE.model import MADE
from utils.helpers import save_training_plot, save_samples_plot, save_model
from utils.train import train_epoch, evaluate, DEVICE


def train(train_data, test_data, tr_params, model_params, image_shape, output_dir, filename):

    num_epochs, lr = tr_params['num_epochs'], tr_params['lr']
    binarize, batch_size, save_every = tr_params['binarize'], tr_params['batch_size'], tr_params['save_every']

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    print(f'device found: {DEVICE}')
    model = MADE(image_shape, **model_params, one_hot=True).to(DEVICE)

    optimizer = AdamW(params=model.parameters(), lr=lr)
    tr_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, binarize=binarize)
        test_loss = evaluate(model, test_loader, binarize=binarize)
        print(f'Epoch {epoch + 1}/{num_epochs} test_loss {test_loss:.5f}')
        tr_losses.append(tr_loss)
        test_losses.append(test_loss)
        # saving model
        if (epoch + 1) % save_every == 0:
            print('-- Saving Model --')
            save_model(model, output_dir / f'{filename}_model_epoch{epoch+1}.pt')
    samples = model.sample(100, visible=tr_params['visible'])
    samples = samples / (model_params['num_classes'] - 1)
    save_training_plot(tr_losses, test_losses, None,
                       output_dir / f'{filename}train_plot_3.png')
    save_samples_plot(samples, f'../results/mnist_MADE_samples_3.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training MADE",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate',
                    help='Optimizer learning rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs',
                        help='Number of training epochs', type=int, default=10)
    parser.add_argument('--h_layers', nargs='+', type=int, help='number of neurons in each hidden layer')
    parser.add_argument('--binarize', action='store_true', help='To binarize MNIST or not')
    parser.add_argument('-v', '--visible', action='store_true', help='Use visible sampling progress bar')
    parser.add_argument('-o', '--output_dir', type=Path, help='Use visible sampling progress bar', default='results')
    parser.add_argument('-s', '--save_every', type=int, help='Number of iterations between model saving every',
                        default=1)
    parser.add_argument('-bz', '--batch_size', type=int, help='training and test batch sizes',
                        default=128)
    args = parser.parse_args()
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')
    tr = datasets.MNIST('../data', train=True, download=False, transform=transforms.ToTensor())
    te = datasets.MNIST('../data', train=False, download=False, transform=transforms.ToTensor())
    C = 1
    N, H, W = tr.data.shape

    tr_params = {'num_epochs': args.num_epochs, 'lr': args.learning_rate, 'save_every': args.save_every,
                 'binarize': args.binarize, 'visible': args.visible, 'batch_size': args.batch_size}
    model_params = {'h_layers': args.h_layers, 'num_classes': 2 if args.binarize else 256}
    filename = 'binary_mnist_MADE' if args.binarize else 'grayscale_mnist_MADE'

    train(tr, te, tr_params, model_params, (C, H, W), args.output_dir, filename)

