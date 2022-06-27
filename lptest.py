# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
# m = 15
# n = 10
# np.random.seed(1)
# s0 = np.random.randn(m)
# lamb0 = np.maximum(-s0, 0)
# s0 = np.maximum(s0, 0)
# x0 = np.random.randn(n)
# A = np.random.randn(m, n)
# b = A @ x0 + s0
# c = -A.T @ lamb0


hyperplanes = np.mat([[0.7071,    0.7071, 1.5], 
                [-0.7071,    0.7071, 1.5],
                [0.7071,    -0.7071, 1],
                [-0.7071,    -0.7071, 1]]);


A = np.mat([[0.7071,    0.7071],
            [-0.7071,    0.7071],
            [0.7071,    -0.7071],
            [-0.7071,    -0.7071]])
b = np.array([1.5, 1.5, 1, 1])
c = np.mat([2, 1]).T

# Define and solve the CVXPY problem.
x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()

# Print result.
# print("\nThe optimal value is", prob.value)
print(f"The optimal point: ({np.round(x.value[0], 2)} ,{np.round(x.value[1], 2)})")
# print("A solution x is")
# print(np.round(x.value, 2))
# print("A dual solution is")
# print(prob.constraints[0].dual_value)