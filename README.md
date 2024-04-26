# Principal Component Analysis and Support Vector Machines 

This repository contains code for implementing Principal Component Analysis (PCA) and a simple linear Support Vector Machine (SVM) for binary classification using stochastic gradient descent (SGD) on the MNIST dataset.

## Overview

PCA is a statistical technique used to reduce the dimensionality of a dataset by projecting it onto a lower-dimensional space while retaining the maximum amount of variance in the data. It is commonly used for dimensionality reduction for the visualization of high-dimensional data.

SVM is a supervised learning algorithm used for classification and regression tasks. Here, we implement a simple linear SVM for binary classification using stochastic gradient descent (SGD). The SVM parameters (weight vector w and bias b) are initialized to zero, and the model is trained using stochastic gradient descent with the hinge loss and L2 regularization.

## Implementation Details

- **PCA**: The PCA algorithm is implemented to reduce the dimensionality of the MNIST dataset.
- **SVM**: The SVM model is implemented using stochastic gradient descent (SGD) with the hinge loss and L2 regularization.
- **Prediction**: After training the SVM model, predictions are made for new data points using the learned SVM parameters.

## Results

- Accuracy =  0.8581.
    - For Hyper Parameters
        -  C = 5.
        -  Learning Rate = 0.001.
        -  Number of Iterations = 100000.
          

 
## Getting Started

### Prerequisites

Make sure you have the following installed:

- [Python](https://www.python.org/) and [pip](https://pip.pypa.io/)
- [NumPy](https://numpy.org/) 

### Installation

### Installation

1. Clone the repository:

   git clone [PCA-and-SVM](https://github.com/ugendar07/PCA-and-SVM.git)


## Contact
For questions or inquiries, please contact [ugendar](mailto:ugendar07@gmail.com) .

