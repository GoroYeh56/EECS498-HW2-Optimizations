import numpy as np
import matplotlib.pyplot as plt
import time
import random

maxi = 10000 #this is the number of functions
# Function itself
def fi(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (np.exp(coef1*x + 0.1) + np.exp(-coef1*x - 0.5) - coef2*x)/(maxi/100)
# First Derivative
def fiprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (coef1*np.exp(coef1*x + 0.1) -coef1*np.exp(-coef1*x - 0.5) - coef2)/(maxi/100)
# Second Derivative
def fiprimeprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    #coef2 = 1 + (6-1)*i/maxi
    return (coef1*coef1*np.exp(coef1*x + 0.1) +coef1*coef1*np.exp(-coef1*x - 0.5))/(maxi/100)


def fsum(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fi(x,i)
    return sum
# Sum of First Derivative
def fsumprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprime(x,i)
    return sum
# Sum of Second Derivative
def fsumprimeprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprimeprime(x,i)
    return sum

#this is just to see the function, you don't have to use this plotting code
xvals = np.arange(-100, 600, 1) # Grid of 0.01 spacing from -10 to 10
yvals = fsum(xvals) # Evaluate function on xvals
# print("xmin: ",np.argmin(yvals,0))
print("min: ",np.min(yvals))
print("xmin: ", np.argmin(yvals, 0)-100 )
plt.figure()
plt.subplot(2,1,1)
plt.axis([-100, 600, -1500, 1500])
plt.plot(xvals, yvals) # Create line plot with yvals against xvals
plt.title(" fsum(x) v.s. x ")

#YOUR ALGORITHM HERE#
stepsize = 1 # t = 1
ITERATIONS_B = [i for i in range(1000)]
x_init = -5
x = x_init
x = 1
x_next = x_init

def grad(fi_prime):
    return np.transpose(fi_prime)

# Key: update x_next!
def SGD(x, ITERATIONS):
    x_next = x
    xs = []
    for iter in ITERATIONS:
        # Each iter pick random i
        i = random.randint(0,maxi)
        delta_x = -grad(fiprime(x_next, i))
        # print("grad: ", grad(fiprime(x, i)), ", delta_x", delta_x)
        x_next  += stepsize * delta_x 
        xs.append(x_next)
    return xs, x_next

# ------- Stochastic Gradient Descent ------- #

# (a) (b) one time
XS = []
XS, x_final = SGD(x, ITERATIONS_B)
# print("XS:", XS)
print("Final x: ",x_final)
# -------------------------------------------- #
plt.subplot(2,1,2)
y_b = fsum(np.asarray(XS))
plt.plot(ITERATIONS_B, y_b)
plt.title(" fsum(xis) v.s. Iteration ")
plt.show()


