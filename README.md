# Generative Models
A small collection of generative models implemented in python
## Model list
### Autoregressive Models
Models that generate a probability distribution over an input x as a product of its
nested conditionals  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" title="P(x) = \prod_{d=1}^D p(x_d|x_{1:d-1})" /></a>  
[MADE](AutoRegressive/MADE/)  
[Gated Pixel CNN](AutoRegressive/GatedPixelCNN/)
### Flow models
