# Generative Models
A small collection of generative models implemented in python

Generative models 
## Model list
### Autoregressive Models
Probabilistic models that receive x as an input and output P(x). These models exploit the fact that P(x) can be expressed as the product of its
nested conditionals.  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x)&space;=&space;\prod_{d=1}^D&space;p(x_d|x_{1:d-1})" title="P(x) = \prod_{d=1}^D p(x_d|x_{1:d-1})" /></a>  

[MADE](AutoRegressive/MADE/)  
[Gated Pixel CNN](AutoRegressive/GatedPixelCNN/)
### Flow models
A class of probabilistic generative models that model P(x) using the change of variables formula.  
<img src="https://i.upmath.me/svg/p_%5Ctheta(x)%20%3D%20p(f_%5Ctheta(x))%7Cdet(%5Cfrac%7B%5Cpartial%20f_%5Ctheta(x)%7D%7B%5Cpartial%20x%5ET%7D)%7C" alt="p(x) = " />  
Where f(x) = z is a latent variable and f is a bijection.  

[RealNVP](Flow/RealNVP/)  
[Glow](Flow/Glow/)