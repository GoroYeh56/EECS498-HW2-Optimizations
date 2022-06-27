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



def GD( x, tinit, thershold):
    delta_x = 0.1 # arbitrary large number
    xs = [x]
    t = tinit 
    while abs(delta_x) > thershold:
        delta_x = -grad(fiprime(x))
        t = backtracking(x, delta_x, t, alpha, beta)
        x = x + t*delta_x
        xs.append(x)
    # print(f"GD: delta_x: {delta_x}")
    return xs, xs[-1]

print("\n====== gradientdescent.py =======")
xinit = 5
tinit = 1 # initial step size
threshold = 1e-1
xs, x_final = GD(xinit, tinit, threshold )
# print(xs[-1])
print("GD: x* : ", x_final)
print("GD: f(x*): ", f(x_final))