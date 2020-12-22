"""
Script for training a RealNVP model on the MNIST dataset
"""
import argparse
from pathlib import Path

import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

from Flow.RealNVP.model import RealNVP
from Flow.utils.train import train_epoch, evaluate, pre_process, DEVICE
from utils.helpers import load_pickle, save_model_state, save_training_plot, save_samples_plot, load_model_checkpoint


def train(train_data, test_data, tr_params, model_params, data_shape, output_dir, filename, pre_trained_path=None):

    train_loader = data.DataLoader(train_data, batch_size=tr_params['batch_size'], pin_memory=False)
    test_loader = data.DataLoader(test_data, batch_size=tr_params['batch_size'], pin_memory=False)

    print(f'device found: {DEVICE}')
    model = RealNVP(data_shape, **model_params).to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=tr_params['lr'], weight_decay=5e-5)
    if pre_trained_path is None:
        tr_losses, test_losses = [], []
        tr_epoch = 0
    else:
        checkpoint = load_model_checkpoint(pre_trained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tr_epoch = checkpoint['epoch'] + 1
        tr_losses = checkpoint['tr_losses']
        test_losses = checkpoint['test_losses']

    for epoch in range(tr_epoch, tr_epoch + tr_params['num_epochs']):
        tr_loss = train_epoch(model, train_loader, optimizer, tr_params['num_colors'], tr_params['grad_clip'],
                              visible=epoch if tr_params['visible'] is not None else None)
        print('-- Evaluating --')
        test_loss = evaluate(model, test_loader, num_colors)
        print(f"Epoch {epoch + 1}/{tr_epoch + tr_params['num_epochs']} test_loss {test_loss:.5f}")
        tr_losses.append(tr_loss)
        test_losses.append(test_loss)
        # saving model
        if (epoch + 1) % tr_params['save_every'] == 0:
            print('-- Saving Model --')
            state = {'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'tr_losses': tr_losses,
                     'test_losses': test_losses}
            save_model_state(state, output_dir / f'{filename}_model_epoch{epoch + 1}.pt')
            print('-- Sampling --')
            raw_samples = model.sample(100)
            samples = pre_process(raw_samples, reverse=True)
            save_samples_plot(samples, output_dir / f'{filename}_epoch_{epoch + 1}_samples.png')
            save_model_state(state, output_dir / f'{filename}_model_epoch{epoch + 1}.pt')
    print('-- Sampling --')
    raw_samples = model.sample(100)
    samples = pre_process(raw_samples, reverse=True)
    save_training_plot(tr_losses, test_losses, None,
                       output_dir / f'{filename}_train_plot.png')
    # save model # save samples
    state = {'epoch': tr_epoch + tr_params['num_epochs'],
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'tr_losses': tr_losses,
             'test_losses': test_losses}
    save_model_state(state, output_dir / f'{filename}_model.pt')
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
    parser.add_argument('-b', '--binarize', action='store_true', help='Binarize grayscale mnist -- not relevant for rgb mnist')
    parser.add_argument('-c', '--color_conditioning', action='store_true', help='Dependent color channels')
    parser.add_argument('-v', '--visible', action='store_true', help='Use visible sampling progress bar')
    parser.add_argument('-o', '--output_dir', type=Path, help='Output directory', default='results/realnvp')
    parser.add_argument('-s', '--save_every', type=int, help='Number of iterations between model saving every',
                        default=1)
    parser.add_argument('-bz', '--batch_size', type=int, help='training and test batch sizes',
                        default=64)
    parser.add_argument('--num_res_blocks', type=int, help='number of resnet blocks for scale parameter in checker board transform',
                        default=4)
    parser.add_argument('--num_scales', type=int, help='number of scales',
                        default=2)
    parser.add_argument('--num_filters', type=int, help='number of filters',
                        default=64)
    parser.add_argument('-md', '--model_dir', type=Path, help='Directory of pre-trained model', default='results/realnvp')
    parser.add_argument('-fn', '--pre_trained_filename', type=str, help='filename of pre_trained model')
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
        tr = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        te = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

        tr = tr.data
        te = te.data

        if args.binarize:
            tr = (tr > 127.5).byte()
            te = (te > 127.5).byte()

        tr = tr.unsqueeze(1)
        te = te.unsqueeze(1)

        _, C, H, W = tr.shape
        num_colors = 2 if args.binarize else 256
        filename = 'mnist_realnvp'

    model_params = {'num_res_blocks': args.num_res_blocks, 'num_scales': args.num_scales,
                    'num_filters': args.num_filters}
    tr_params = {'num_colors': num_colors, 'num_epochs': args.num_epochs, 'lr': args.learning_rate,
                 'grad_clip': args.grad_clip, 'visible': args.visible, 'save_every': args.save_every,
                 'batch_size': args.batch_size, 'color_conditioning': args.color_conditioning}

    if args.pre_trained_filename is not None:
        print('-- Loaded pre-trained model --')
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    train(tr, te, tr_params, model_params, (C, H, W),
          args.output_dir, filename,
          pre_trained_path=args.model_dir / args.pre_trained_filename if args.pre_trained_filename is not None else None)
