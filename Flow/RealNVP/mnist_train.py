"""
Script for training a RealNVP model on the MNIST dataset
"""
import argparse
from pathlib import Path

import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam, lr_scheduler

from Flow.RealNVP.model import RealNVP
from Flow.utils.train import train_epoch, evaluate, pre_process, DEVICE
from utils.helpers import load_pickle, save_model, save_training_plot, save_samples_plot


def train(train_data, test_data, tr_params, model_params, data_shape, output_dir, filename):
    num_epochs, lr, grad_clip = tr_params['num_epochs'], tr_params['lr'], tr_params['grad_clip']
    binarize, batch_size, save_every = tr_params['binarize'], tr_params['batch_size'], tr_params['save_every']
    color_conditioning = tr_params['color_conditioning']
    num_colors = model_params['num_colors']

    train_loader = data.DataLoader(train_data, batch_size=batch_size, pin_memory=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, pin_memory=False)

    print(f'device found: {DEVICE}')
    model = RealNVP(data_shape, **model_params).to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda _: 0.9999999)

    tr_losses, test_losses = [], []
    for epoch in range(num_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, num_colors, grad_clip, scheduler,
                              visible=epoch + 1 if tr_params['visible'] is not None else None)
        print('-- Evaluating --')
        test_loss = evaluate(model, test_loader, num_colors)
        print(f'Epoch {epoch + 1}/{num_epochs} test_loss {test_loss:.5f}')
        tr_losses.append(tr_loss)
        test_losses.append(test_loss)
        # saving model
        if (epoch + 1) % save_every == 0:
            print('-- Saving Model --')
            save_model(model, output_dir / f'{filename}_model_epoch{epoch + 1}.pt')
    print('-- Sampling --')
    raw_samples = model.sample(100)
    samples = pre_process(raw_samples, reverse=True)
    if color_conditioning:
        samples /= 3
    save_training_plot(tr_losses, test_losses, None,
                       output_dir / f'{filename}_train_plot.png')
    # save model # save samples
    save_model(model, output_dir / f'{filename}_model.pt')
    save_samples_plot(samples, output_dir / f'{filename}_samples.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training RealNVP on the MNIST data set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate',
                        help='Optimizer learning rate', type=float, default=1e-3)
    parser.add_argument('-ne', '--num_epochs',
                        help='Number of training epochs', type=int, default=15)
    parser.add_argument('-g', '--grad_clip', type=float, help='Value to clip norm of gradient to')
    parser.add_argument('-nl', '--num_layers', type=int, help='Number of horizontal and vertical layers', default=8)
    parser.add_argument('-u', '--label_conditioning', action='store_true', help='Condition on  labels')
    parser.add_argument('-c', '--color_conditioning', action='store_true', help='Dependent color channels')
    parser.add_argument('-v', '--visible', action='store_true', help='Use visible sampling progress bar')
    parser.add_argument('-o', '--output_dir', type=Path, help='Use visible sampling progress bar', default='results')
    parser.add_argument('-s', '--save_every', type=int, help='Number of iterations between model saving every',
                        default=1)
    parser.add_argument('-bz', '--batch_size', type=int, help='training and test batch sizes',
                        default=64)
    parser.add_argument('--num_samples', type=int, help='num_samples',
                        default=None)
    parser.add_argument('--num_res_blocks', type=int, help='number of resnet blocks for scale parameter in checker board transform',
                        default=4)
    parser.add_argument('--num_scales', type=int, help='number of scales',
                        default=2)
    args = parser.parse_args()
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')

    if args.color_conditioning:
        mnist_colored = load_pickle('data/mnist_colored.pkl')
        tr, te = mnist_colored['train'], mnist_colored['test']
        tr = np.transpose(tr, (0, 3, 1, 2))
        te = np.transpose(te, (0, 3, 1, 2))
        if args.num_samples is not None:
            tr = tr[:args.num_samples]
            te = te[:args.num_samples]
        _, C, H, W = tr.shape
        num_colors = 4
        binarize = False
        filename = 'mnist_colored_realnvp'
    else:
        tr = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        te = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
        _, C, H, W = (tr.data.unsqueeze(1)).shape
        num_colors = 256
        binarize = True
        filename = 'mnist_label_cond_realnvp' if args.label_conditioning else 'mnist_realnvp'
        if args.num_samples is not None:
            tr = data.Subset(tr, list(range(args.num_samples)))
            te = data.Subset(te, list(range(args.num_samples)))
    # number of label classes
    num_classes = 10 if args.label_conditioning and not args.color_conditioning else None
    model_params = {'num_colors': num_colors, 'n_res_blocks': args.num_res_blocks, 'num_scales': args.num_scales}
    tr_params = {'num_epochs': args.num_epochs, 'lr': args.learning_rate,
                 'grad_clip': args.grad_clip, 'visible': args.visible,
                 'binarize': binarize, 'save_every': args.save_every,
                 'batch_size': args.batch_size, 'color_conditioning': args.color_conditioning}

    train(tr, te, tr_params, model_params, (C, H, W),
          args.output_dir, filename)
