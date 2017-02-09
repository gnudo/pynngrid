###################################################################
###################################################################
####                                                           ####
####                            PYNNGRID                       ####
####                                                           ####
###################################################################
###################################################################



##  Brief description
This repository contains a customized implementation of the Neural Network 
Filtered Backprojection (NN-FBP) algorithm devised by Daniel Pelt.
The original NN-FBP code is available at: https://github.com/dmpelt/pynnfbp.
If you intend to use this software, please cite the original publication:
Pelt, D., & Batenburg, K. (2013). "Fast tomographic reconstruction from limited
data using artificial neural networks". Image Processing, IEEE Transactions on,
22(12), pp.5238-5251.
The NN-FBP implementation in this repository does not require the
installation of the Astra Toolbox, which works with GPUs.
The gridding backprojector is used instead.




##  Installation
Basic compilers like gcc and g++ are required.
The simplest way to install all the code is to use Anaconda with python-2.7 and to 
add the installation of the python package scipy, scikit-image and Cython.

On a terminal, just type:

1.   `conda create -n iter-rec python=2.7 anaconda`
2.   `conda install -n iter-rec scipy scikit-image Cython`
3.   `source activate iter-rec`
4.   download the repo and type: `python setup.py`

If setup.py runs without giving any error all subroutines in C have been installed and
your python version meets all dependencies.



##  Test the package
Go inside the folder "tests/" and run: `python run_test.py`
Every time this script creates an image, the script is halted. To run the successive tests
just close the image.

