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

print("\n====== (a)(b): Implementing SGD =======")
print("True fsum(x) min: ",np.min(yvals))
print("True x*: ", np.argmin(yvals, 0)-100 )
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

# -------------------------------------------- #
plt.subplot(2,1,2)
y_b = fsum(np.asarray(XS))
plt.plot(ITERATIONS_B, y_b)
plt.title(" fsum(xis) v.s. Iteration ")
# plt.plot(XS, yvals) 
print("SGD's x*: ",x_final)
print("SGD's f(x*): ",fsum(x_final))


# (c) 1000 iter * 30 times v.s. 750 * 30 times

ITERATIONS_C = [i for i in range(750)]
ITERATIONS_C1000 = [i for i in range(1000)]
NUMS_LOOP = 30

XS_C = []
XS_C.append(x_init)
x_final = x_init
x_final750 = []
for i in range(NUMS_LOOP):
    XS_C, x_final = SGD(x_final, ITERATIONS_C)
    x_final750.append(x_final)

XS_C1000 = []
XS_C1000.append(x_init)
x_final1000 = []
x_final = x_init
for i in range(NUMS_LOOP):
    XS_C1000, x_final = SGD(x_final, ITERATIONS_C1000)
    x_final1000.append(x_final)

epochs = [i for i in range(NUMS_LOOP)] # [0 1 2 ... 29]

def fsum_list_input(x):
    return fsum(np.asarray(x))

# (c) Compute the mean & variance
print("\n====== (c): 30-iterations profiling: =======")
print("SGD-750 mean " + str(np.mean( fsum_list_input(x_final750))) +" var: " + str(np.var( fsum_list_input(x_final750))))
print("SGD-1000 mean " + str(np.mean( fsum_list_input(x_final1000)))+ " var: " + str(np.var( fsum_list_input(x_final1000))))


# plt.figure()
# plt.subplot(2,1,1)
# plt.title("750 iters * 30 times")
# plt.plot(epochs, fsum(np.asarray(x_final750)))
# plt.xlabel("epoch")
# plt.ylabel("fsum(x_final[i]")

# plt.subplot(2,1,2)
# plt.title("1000 iters * 30 times")
# plt.xlabel("epoch")
# plt.ylabel("fsum(x_final[i])")
# plt.plot(epochs, fsum(np.asarray(x_final1000)))





# (d) Compare with Gradient Descent, Newton's Method
# (i) Compare runtime in three methods & discuss the reason
# (ii) Compare fsum(x*) in 3 methods

print("\n====== (d): Compare SGD, GD and Newton's Method =======")
# SGD 1000 iters
start = time.time()
XS = []
XS, x_final = SGD(x_init, ITERATIONS_B)
end = time.time()
print("SGD Time: ", end - start)
print("fsum(x*): ",fsum(x_final))
# -------------------------------------------- #

xinit = -5
tinit = 1 # initial step size
alpha = 0.1
beta = 0.6
epsilon = 0.0001

threshold = 1e-4 # GD threshold

def Grad(x):
    return np.transpose(fsumprime(x))
def backtracking(x, dx, t, alpha, beta):
    while fsum(x + t*dx) > (fsum(x) + alpha*t*np.transpose(grad(fsumprime(x))*dx)) :
        t = beta*t
    return t

def GD( x, tinit, thershold):
    delta_x = 1 # arbitrary large number
    xs = [x]
    t = tinit 
    while abs(delta_x) > thershold:
        delta_x = -Grad(x)
        t = backtracking(x, delta_x, t, alpha, beta)
        x = x + t*delta_x
        xs.append(x)
    # print(f"GD: delta_x: {delta_x}")
    return xs, xs[-1]


# GD
start = time.time()
xs_GD = []
xs_GD, xstar_GD = GD(x_init, tinit, threshold )
end = time.time()
print("GD Time: ", end - start)
print("fsum(x*): ",fsum(xstar_GD))
# -------------------------------------------- #



def Hessian(x):
    return  fsumprimeprime(x)

def Newtons( x, tinit, epsilon):
    delta_x = 1 # arbitrary large number
    xs = [x]
    t = tinit
    while True:
        delta_x = - (1/(Hessian(x))) * Grad(x)
        lambda2 =  fsumprime(x) * (1/(Hessian(x))) * Grad(x) 
        if (lambda2/2) <= epsilon:
            # print("lambda2 : ",lambda2/2, "epsilon ",epsilon)
            break
        t = backtracking(x, delta_x, t, alpha, beta)
        x = x + t*delta_x
        xs.append(x)
    return xs, xs[-1]


# Newton's method
start = time.time()
xs_NT = []
xs_NT, xstar_NT= Newtons(x_init, tinit, epsilon )
end = time.time()
print("Newton's Time: ", end - start)
print("fsum(x*): ",fsum(xstar_NT))
# -------------------------------------------- #

# plt.show() #show the plot
