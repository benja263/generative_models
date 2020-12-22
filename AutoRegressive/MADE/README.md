# MADE
Python implementation of the [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) paper.
## Architecture
Estimating the probability of an input x as a product of its nested conditionals by masking connections such that each output is connected solely to its preceding inputs.  

![MADE architecture](../../images/MADE.png)

## Run 
```
python3 AutoRegressive/MADE/mnist_train.py
```

add ```--binarize``` for binarized MNIST

## Results
### Binarized MNIST
![MADE results binarized](../../results/mnist_MADE_samples.png)

## References
1)  Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle.  Made: Masked autoencoder for distribution estimation, 2015. [arXiv:1502.03509](https://arxiv.org/abs/1502.03509)  

