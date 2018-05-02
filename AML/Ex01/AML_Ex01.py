# AML - Exercise 01

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import random

# 1) Loading the Dataset

digits = load_digits()
print(digits.keys())

data = digits['data']
images = digits['images']
target = digits['target']
target_names = digits['target_names']

print(data.dtype)

# reduced X and target vector Y
X = data[(target == 3) | (target == 8)]
X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
print(X.shape)
Y = target[(target == 3) | (target == 8)]
Y[Y == 3] = 1
Y[Y == 8] = -1
print(Y.shape)


# 1.1) Classification with sklearn

log_reg = LogisticRegression(C=0.5)
accur = cross_val_score(log_reg, X, Y)
print(accur.mean(), accur.std())

log_reg.C = 1.0
accur = cross_val_score(log_reg, X, Y)
print(accur.mean(), accur.std())

log_reg.C = 0.3
accur = cross_val_score(log_reg, X, Y)
print(accur.mean(), accur.std())

log_reg.C = 0.6
accur = cross_val_score(log_reg, X, Y)
print(accur.mean(), accur.std())

log_reg.C = 0.8
accur = cross_val_score(log_reg, X, Y)
print(accur.mean(), accur.std())


lambd = 0.8


#1.2) Optimization methods

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def gradient(beta, X, y):
    N = X.shape[0]
#    print(beta.shape, X.shape, y.shape)
    return (1/N) * (np.dot(sigmoid((np.dot(np.dot(X.T, y), beta) * -1)), (-1 * (np.dot(X.T, y)))) + (beta / lambd))


def predict(beta, X):
    prediction = np.dot(X, beta)
    prediction[prediction <= 0] = -1
    prediction[prediction > 0] = 1
    return prediction


def zero_one_loss(y_prediction, y_truth):
    N = len(y_truth)
    return (N - np.sum(np.equal(y_prediction, y_truth)))


def gradient_descent(X, y, beta, tau, m):
    for t in range(m):
        beta = beta - tau * gradient(beta, X, y)
    return beta


def stochastic_gradient(X, y, beta, tau, gamma, m):
    tau_zero = tau
    for t in range(m):
        index = random.randint(0, (X.shape[0] - 1))
        beta = beta - tau * gradient(beta, X[index], y[index])
        tau = tau_zero / (1 + gamma * t)
    return beta


def stochastic_gradient_minibatch(X, y, beta, tau, gamma, m):
    tau_zero = tau
    for t in range(m):
        rand_indices = random.sample(range(0, (X.shape[0] - 1)), 10)
        beta = beta - tau * gradient(beta, X[rand_indices], y[rand_indices])
        tau = tau_zero / (1 + gamma * t)
    return beta


def stochastic_gradient_momentum(X, y, beta, tau, gamma, g, mu, m):
    tau_zero = tau
    for t in range(m):
        rand_index = random.randint(0, (X.shape[0] - 1))
        g = mu * g + (1 - mu) * gradient(beta, X[rand_index], y[rand_index])
        beta = beta - tau * g
        tau = tau_zero / (1 + gamma * t)
    return beta


def ADAM(X, y, beta, g, m):
    mu_one = 0.9
    mu_two = 0.999
    tau = 10e-4
    epsilon = 10e-8
    q = 0
    for t in range(m):
        rand_index = random.randint(0, (X.shape[0] - 1))
        loss_gradient = gradient(beta, X[rand_index], y[rand_index])
        g = mu_one * g + (1 - mu_one) * loss_gradient
        q = mu_two * q + (1 - mu_two) * np.dot(loss_gradient, loss_gradient)
        beta = beta - (tau / (np.sqrt(q) + epsilon)) * g
    return beta


def stochastic_average_gradient(X, y, beta, tau, gamma, m):
    tau_zero = tau
    N = X.shape[0]
    g_stored = np.dot(sigmoid((np.dot(np.dot(X.T, y), beta) * -1)), (-1 * (np.dot(X.T, y))))
    g_t = g_stored / N
    for t in range(m):
        rand_index = random.randint(0, (X.shape[0] - 1))
        g_t = np.dot(sigmoid((np.dot(np.dot(X[rand_index].T, y[rand_index]), beta) * -1)), (-1 * (np.dot(X[rand_index].T, y[rand_index]))))
        g = g_t + (1 / N) * (g_t - g_stored)
        g_stored = g_t
        beta = beta * (1 - tau / lambd) - tau * g
        tau = tau_zero / (1 + gamma * t)
    return beta


bet_zero = np.zeros((65))
g_zero = np.zeros((65))
bet_m = stochastic_average_gradient(X, Y, bet_zero, 0.001, 0.1, 150)
print(zero_one_loss(Y, predict(bet_m, X)))
