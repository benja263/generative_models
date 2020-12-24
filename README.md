# Generative Models
A small collection of generative models implemented in python

## Model List
### Autoregressive Models
Probabilistic models that receive x as an input and output P(x). These models exploit the fact that P(x) can be expressed as the product of its
nested conditionals.  
<img src="https://i.upmath.me/svg/P(x)%20%3D%20%5Cprod_%7Bd%3D1%7D%5ED%20P(x_d%7Cx_%7B1%3Ad-1%7D)"/>  

[MADE](AutoRegressive/MADE/)  
[Gated Pixel CNN](AutoRegressive/GatedPixelCNN/)
### Flow Models
A class of probabilistic generative models that model P(x) using the change of variables formula.  
<img src="https://i.upmath.me/svg/p_%5Ctheta(x)%20%3D%20p(f_%5Ctheta(x))%7Cdet(%5Cfrac%7B%5Cpartial%20f_%5Ctheta(x)%7D%7B%5Cpartial%20x%5ET%7D)%7C" alt="p(x) = " />  
Where f(x) = z is a latent variable and f is a bijection.  

[RealNVP](Flow/RealNVP/)  
[Glow](Flow/Glow/)