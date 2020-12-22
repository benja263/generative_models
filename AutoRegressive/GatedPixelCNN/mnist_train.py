"""
Script for training a GatedPixelCNN model on the MNIST dataset
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

from AutoRegressive.GatedPixelCNN.model import GatedPixelCNN
from AutoRegressive.utils.train import train_epoch, evaluate, DEVICE
from utils.helpers import load_pickle, save_model_state, save_training_plot, save_samples_plot


def train(train_data, test_data, tr_params, model_params, data_shape, output_dir, filename):

    train_loader = data.DataLoader(train_data, batch_size=tr_params['batch_size'], shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=tr_params['batch_size'])

    print(f'device found: {DEVICE}')
    model = GatedPixelCNN(data_shape, **model_params).to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=tr_params['lr'])

    tr_losses, test_losses = [], []
    for epoch in range(tr_params['num_epochs']):
        tr_loss = train_epoch(model, train_loader, optimizer, model_params['label_conditioning'],
                              grad_clip=tr_params['grad_clip'], binarize=tr_params['binarize'],
                              epoch=epoch + 1 if tr_params['visible'] is not None else None)
        print('-- Evaluating --')
        test_loss = evaluate(model, test_loader, model_params['label_conditioning'], binarize=tr_params['binarize'])
        print(f"Epoch {epoch + 1}/{tr_params['num_epochs']} test_loss {test_loss:.5f}")
        tr_losses.append(tr_loss)
        test_losses.append(test_loss)

    if model_params['label_conditioning'] is not None:
        cond = torch.arange(model_params['label_conditioning']).unsqueeze(1).repeat(1, 100 // model_params['label_conditioning']).view(-1).long()
        one_hot = F.one_hot(cond, model_params['label_conditioning']).float().to(DEVICE)
        samples = model.sample(100, y=one_hot, visible=tr_params['visible'])
    else:
        samples = model.sample(100, visible=tr_params['visible'])

    samples = samples / (num_colors - 1)
    save_training_plot(tr_losses, test_losses, None,
                       output_dir / f'{filename}_train_plot.png')
    # save model # save samples
    save_model_state(model, output_dir / f'{filename}_model.pt')
    save_samples_plot(samples, output_dir / f'{filename}_samples.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training GatedPixelCNN on the MNIST data set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate',
                        help='Optimizer learning rate', type=float, default=1e-3)
    parser.add_argument('-ne', '--num_epochs',
                        help='Number of training epochs', type=int, default=15)
    parser.add_argument('-g', '--grad_clip', type=float, help='Value to clip norm of gradient to')
    parser.add_argument('-nl', '--num_layers', type=int, help='Number of horizontal and vertical layers', default=8)
    parser.add_argument('-nf', '--num_h_filters', type=int, help='Number of filters in hidden layers', default=120)
    parser.add_argument('-of', '--num_o_filters', type=int, help='Number of filters in output layer', default=30)
    parser.add_argument('-u', '--label_conditioning', action='store_true', help='Condition on  labels')
    parser.add_argument('-c', '--color_conditioning', action='store_true', help='Dependent color channels')
    parser.add_argument('-v', '--visible', action='store_true', help='Use visible sampling progress bar')
    parser.add_argument('-o', '--output_dir', type=Path, help='Use visible sampling progress bar', default='results/gatedpixelcnn')
    parser.add_argument('-bz', '--batch_size', type=int, help='training and test batch sizez',
                        default=128)
    parser.add_argument('--num_samples', type=int, help='num_samples',
                        default=None)
    args = parser.parse_args()
    print('-- Entered Arguments --')
    for arg in vars(args):
        print(f'- {arg}: {getattr(args, arg)}')

    if args.color_conditioning:
        mnist_colored = load_pickle('../../data/mnist_colored.pkl')
        tr, te = mnist_colored['train'], mnist_colored['test']
        tr = np.transpose(tr, (0, 3, 1, 2))
        te = np.transpose(te, (0, 3, 1, 2))
        if args.num_samples is not None:
            tr = tr[:args.num_samples]
            te = te[:args.num_samples]
        _, C, H, W = tr.shape
        num_colors = 4
        binarize = False
        filename = 'mnist_colored_gatedpixelcnn'
    else:
        tr = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        te = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
        _, C, H, W = (tr.data.unsqueeze(1)).shape
        num_colors = 2
        binarize = True
        filename = 'mnist_label_cond_gatedpixelcnn' if args.label_conditioning else 'mnist_gatedpixelcnn'
        if args.num_samples is not None:
            tr = data.Subset(tr, list(range(args.num_samples)))
            te = data.Subset(te, list(range(args.num_samples)))
    # number of label classes
    num_labels = 10 if args.label_conditioning and not args.color_conditioning else None
    model_params = {'num_layers': args.num_layers, 'num_h_filters': args.num_h_filters,
                    'num_o_filters': args.num_o_filters, 'label_conditioning': num_labels,
                    'color_conditioning': args.color_conditioning, 'num_colors': num_colors}

    tr_params = {'num_epochs': args.num_epochs, 'lr': args.learning_rate,
                 'grad_clip': args.grad_clip, 'visible': args.visible,
                 'binarize': binarize, 'save_every': args.save_every,
                 'batch_size': args.batch_size}

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    train(tr, te, tr_params, model_params, (C, H, W), args.output_dir, filename)


