# Spinglass_NADE
Constructs a neural network that aids in speeding up simulation of spin glasses

## Intoduction
NADE stands for Neural Autoregressive Distribution Estimator that makes use of variational autoregressor network to approximate an arbitrary (discrete) probability distribtion on a finite number of variables- spins on a square Edwards Anderson lattice in our case.

## Instructions
1. There are two ways one can explore this repo- local machine or Google colab.
2. In case of local machine, fork this repo to own Github account and pull the fork to your local machine using git. Open the main.ipynb file and run followng the instructions. In particular, ignore the Colab Instructions section.
3. In case of Google colab, simply open main.ipynb in this repo and click on the "Open in Colab" interactive button. In the notebook that opens, log in with your Google account and change runtime type to "GPU" to get better performance.

## References
1. McNaughton et al, Boosting Monte Carlo simulations of spin glasses using autoregressive neural networks, https://arxiv.org/abs/2002.04292
