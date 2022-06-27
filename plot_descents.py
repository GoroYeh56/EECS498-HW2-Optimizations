import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg.linalg import inv
from newtonsmethod import Newtons
from gradientdescent import GD
from backtracking import backtracking

def f(x):
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 5*x

def fiprime(x):
    return 0.5 * np.exp(0.5*x + 1) + -0.5 * np.exp(-0.5*x - 0.5) + 5

def grad(fprime):
    return np.transpose(fprime)

def Hessian(x):
    return  0.5 *0.5 * np.exp(0.5*x + 1) + (-0.5)*(-0.5) * np.exp(-0.5*x - 0.5)



xinit = 5
tinit = 1 # initial step size
alpha = 0.1
beta = 0.6
epsilon = 0.0001

threshold = 1e-4 # GD threshold

xs_GD = []
xs_NT = []
xs_GD, xstar_GD = GD(xinit, tinit, threshold )
xs_NT, xstar_NT= Newtons(xinit, tinit, epsilon )
# print(f"final x: {x_final}")
# print("f(x*): ", f(x_final))

print(f"GD: iter {len(xs_GD)}")
print(f"NT: iter {len(xs_NT)}")

# (i) plot f(x) - x
plt.subplot(1, 2, 1)
x = np.arange(-10, 10, 1)
true_f = f(x)
plt.plot(x, true_f, 'black', label='True Function')

print(f"xs_GD: {xs_GD}")
plt.plot(np.asarray(xs_GD), f(np.asarray(xs_GD)), 'ro' )
plt.plot(np.asarray(xs_GD), f(np.asarray(xs_GD)), 'r' , label='Gradient Descent')
plt.plot(np.asarray(xs_NT), f(np.asarray(xs_NT)), 'mo')
plt.plot(np.asarray(xs_NT), f(np.asarray(xs_NT)), 'm', label='Newton\'s Method')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc="upper center")
# plt.axis([-3,2,-30,20])
plt.title('f(x) v.s. x')


# (ii) plot f(x) - iteration
 
it_GD = [ i for i in range(len(xs_GD))]
it_NT = [ i for i in range(len(xs_NT))]

plt.subplot(1, 2, 2)
plt.plot(it_GD, f(np.asarray(xs_GD)), 'red', label='Gradient Descent')
plt.plot(it_NT, f(np.asarray(xs_NT)), 'm', label='Newton\'s Method')
plt.xlabel('i (iteration #)')
plt.ylabel('f(xi)')
plt.title('f(xi) v.s. i')
plt.legend(loc="upper center")
plt.show()
