# Description: Newton's Method for Optimization with Backtracking and without Backtracking 

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 4*x[0]**2 + x[1]**2 - 2*x[0]*x[1]

def grad_f(x):
    return np.array([8*x[0] - 2*x[1], 2*x[1] - 2*x[0]])

def hess_f(x):
    return np.array([[8, -2], [-2, 2]])


def backtracking_line_search(f, grad_f, x, dir_descent, alpha = 1, rho = 0.8, c = 1e-4):
    while f(x + alpha * dir_descent) > f(x) + c * alpha * np.dot(grad_f(x), dir_descent):
        alpha = rho * alpha
    return alpha

def steepest_descent(f, grad_f, x_0, alpha = 1, rho = 0.8, tol = 1e-6):
    x = x_0
    path = [x]
    while True:
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        descent = -grad
        alpha = backtracking_line_search(f, grad_f, x, descent)
        x = x + alpha * descent
        path.append(x)
    return np.array(path)

def newton_method(f, grad_f, hess_f, x_0, with_backtracking=False, tol = 1e-6):
    x = x_0
    path = [x]
    while True:
        hess_inv = np.linalg.inv(hess_f(x))
        grad = grad_f(x)
        descent = hess_inv @ grad
        if np.linalg.norm(grad) < tol:
            break
        if with_backtracking:
            alpha = backtracking_line_search(f, grad_f, x, descent)
            x = x + alpha * descent
        else:
            x = x - descent
        path.append(x)
    return np.array(path)



# plot the contour of the function 
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Run Newton's method to get the path
x_0 = np.array([2.0, 2.0])
path_newton = newton_method(f, grad_f, hess_f, x_0)
num_iter_newton = len(path_newton) - 1


# Now find the optimal point using first order method steepest descent method
path_sd = steepest_descent(f, grad_f, x_0)
num_iter_sd = len(path_sd) - 1
print(f'Number of iterations for Newton: {num_iter_newton}')
print(f'Number of iterations for Steepest Descent: {num_iter_sd}')

Z = f([X, Y])
cfp = plt.contourf(X, Y, Z, levels=np.linspace(0, 10, 10), cmap='Blues', extend='max', vmin=0, vmax=10)

# Visualize the path of descent and mark with number of iterations
plt.plot(path_newton[:, 0], path_newton[:, 1], 'o')
for i, txt in enumerate(range(num_iter_newton + 1)):
    plt.annotate(txt, (path_newton[i, 0], path_newton[i, 1]))\
    
# Visulaize the path of steepest descent and mark with number of iterations
plt.plot(path_sd[:, 0], path_sd[:, 1], '+-')
for i, txt in enumerate(range(num_iter_sd + 1)):
    plt.annotate(txt, (path_sd[i, 0], path_sd[i, 1]))

# mark the initial guess and the final result in red and green respectively
plt.plot(path_newton[0, 0], path_newton[0, 1], 'red')
plt.plot(path_newton[-1, 0], path_newton[-1, 1], 'green')

# mark the final result in blue for steepest descent
plt.plot(path_sd[-1, 0], path_sd[-1, 1], 'blue')

# show the optimal point on the contour plot
plt.colorbar(cfp)
plt.clim(0, 10)
plt.savefig(f'f_contour_with_descent.png')
plt.close()