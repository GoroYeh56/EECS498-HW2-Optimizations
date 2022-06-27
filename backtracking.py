import numpy as np
import matplotlib.pyplot as plt
import time
import random
def f(x):
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 5*x

def fiprime(x):
    return 0.5 * np.exp(0.5*x + 1) + -0.5 * np.exp(-0.5*x - 0.5) + 5

def grad(fprime):
    return np.transpose(fprime)

def backtracking(x, dx, t, alpha, beta):
    while f(x + t*dx) > (f(x) + alpha*t*np.transpose(grad(fiprime(x))*dx)) :
        t = beta*t
    return t
    
alpha = 0.1
beta = 0.6
t = 1
epsilon = 0.0001

## TODO modify!
x = xinit = 5
dx = 2

t = backtracking(x, dx, t, alpha, beta)
print("t after one step backtracking line search: ",t)