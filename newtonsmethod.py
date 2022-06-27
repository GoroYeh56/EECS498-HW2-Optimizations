import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg.linalg import inv

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

def Hessian(x):
    return  0.5 *0.5 * np.exp(0.5*x + 1) + (-0.5)*(-0.5) * np.exp(-0.5*x - 0.5)


def Newtons( x, tinit, epsilon):
    delta_x = 10 # arbitrary large number
    xs = [x]
    t = tinit
    while True:
        delta_x = - (1/(Hessian(x))) * grad(fiprime(x))
        lambda2 =  fiprime(x) * (1/(Hessian(x))) * grad(fiprime(x)) 
        if (lambda2/2) <= epsilon:
            print("lambda2 : ",lambda2/2, "epsilon ",epsilon)
            break
        t = backtracking(x, delta_x, t, alpha, beta)
        x = x + t*delta_x
        xs.append(x)
    return xs, xs[-1]


xinit = 5
tinit = 1 # initial step size
    
alpha = 0.1
beta = 0.6
t = 1
epsilon = 0.0001

xs, x_final = Newtons(xinit, tinit, epsilon )
print("\n====== Newton's =======")
print(f"Newton's Method x*: {x_final}")
print("Newton's Method f(x*): ", f(x_final))

