# MADE
Python implementation of the [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) paper.
## Idea
Estimating the probability of an input x as a product of its nested conditionals  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" title="P(x) = \prod_{d=1}^D p(x_d|x_{1:d-1})" /></a>  
This is achieved by masking connections such that each output is connected solely to its preceding inputs.
![MADE architecture](MADE.png)

## Results
### MNIST
![MADE results](../results/mnist_MADE_samples.png)
### Binarized MNIST
![MADE results binarized](../results/mnist_MADE_samples_2.png)

