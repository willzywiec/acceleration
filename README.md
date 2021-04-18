# Acceleration

These scripts test out an idea I had to speed up neural network training by performing linear and nonlinear weight fits and using these fits to predict future weights.
After mild tuning, it works reasonably well for the Boston Housing Prices data, although the code isn't optimized for general or practical use.
I think this would work much better on neural networks that are wider and deeper, possibly with no tuning or optimization needed.  

TL;DR: **forecast.R** speeds up neural network training by skipping epochs.
