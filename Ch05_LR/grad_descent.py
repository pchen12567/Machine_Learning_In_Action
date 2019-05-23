"""
@Coding: uft-8 *
@Time: 2019-05-23 18:09
@Author: Ryne Chen
@File: grad_descent.py 
"""

import numpy as np


# Function to calculate gradient and cost
def get_gradient(theta, x, y):
    # Get total number of samples
    m = x.shape[0]

    # Get prediction value
    y_pred = x.dot(theta)

    # Calculate difference between prediction and real value
    error = y_pred - y

    # Calculate gradient
    gradient = (1.0 / m) * error.dot(x)

    # Calculate cost value
    cost = (1.0 / 2 * m) * np.sum(error ** 2)

    return gradient, cost


# Function to get best theta
def gradient_descent(x, y, max_iter=500, alpha=0.01):
    # Init theta randomly
    theta = np.random.randn(2)

    # Set tolerance value
    tolerance = 1e-3

    # Init iteration time
    iteration = 1

    # Set stop flag
    is_converged = False

    while not is_converged:
        # Get gradient and cost value
        gradient, cost = get_gradient(theta, x, y)

        # Get new theta
        new_theta = theta - alpha * gradient

        # Set stop condition
        if np.sum(abs(new_theta - theta) < tolerance):
            is_converged = True

        # Update iteration time
        iteration += 1

        # Update theta
        theta = new_theta

        # Set stop condition
        if iteration > max_iter:
            is_converged = True

    return theta
