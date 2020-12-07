# MADE
Implementation of the [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) paper.
## Idea
Estimating <img src="https://render.githubusercontent.com/render/math?math=p(x)"> as a product of its nested conditionals
<img src="https://render.githubusercontent.com/render/math?math=p(x)= \prod_{d=1}^D">

<a href="https://www.codecogs.com/eqnedit.php?latex=P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_)" title="P(x) = \prod_{d=1}^D p(x_d|x_)" /></a>
