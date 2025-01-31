import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 4*x[0]**2 + x[1]**2 - 2*x[0]*x[1]

def grad_f(x):
    return np.array([8*x[0] - 2*x[1], 2*x[1] - 2*x[0]])

def backtracking_line_search(f, grad_f, x, dir_descent, alpha=0.1, rho=0.8, c=1e-4, min_alpha=1e-8):
    while f(x + alpha * dir_descent) > f(x) + c * alpha * np.dot(grad_f(x), dir_descent):
        alpha *= rho
        if alpha < min_alpha:  # Avoid infinite loop
            break
    return alpha

def descent_coordinate(f, grad_f, x_0, alpha=0.1, tol=1e-6):
    x = np.array(x_0, dtype=float)
    path = [x.copy()]
    while True:
        gradient = grad_f(x)
        max_dir_idx = np.argmax(np.abs(gradient))
        descent = np.zeros_like(x)
        descent[max_dir_idx] = -gradient[max_dir_idx]
        alpha = backtracking_line_search(f, grad_f, x, descent)
        x += alpha * descent
        path.append(x.copy())
        if np.linalg.norm(gradient) < tol:
            break
    return np.array(path)

def descent_dir_generation_random(grad):
    descent_dir = np.zeros_like(grad)
    while np.dot(descent_dir, grad) >= 0:  # Ensure negative correlation with gradient
        descent_dir = -grad + np.random.randn(len(grad)) * 0.1  # Small randomness
    return descent_dir / np.linalg.norm(descent_dir)  # Normalize

def descent_random(f, grad_f, x_0, alpha=0.1, rho=0.8, tol=1e-6):
    x = np.array(x_0, dtype=float)
    path = [x.copy()]
    while True:
        gradient = grad_f(x)
        descent_direction = descent_dir_generation_random(gradient)
        alpha = backtracking_line_search(f, grad_f, x, descent_direction)
        x += alpha * descent_direction
        path.append(x.copy())
        if np.linalg.norm(gradient) < tol:
            break
    return np.array(path)

# Initial point
x_0 = np.array([2.0, 2.0])

# Generate paths
path_coordinate = descent_coordinate(f, grad_f, x_0)
path_random = descent_random(f, grad_f, x_0)

# Create contour plot
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

plt.figure(figsize=(8, 6))
cfp = plt.contourf(X, Y, Z, levels=20, cmap='Blues', extend='max')

# Plot paths
plt.plot(path_coordinate[:, 0], path_coordinate[:, 1], 'r-', label="Coordinate Descent")
plt.plot(path_random[:, 0], path_random[:, 1], 'g-', label="Random Descent")

# Mark start and end points with + and *
plt.scatter(path_coordinate[0, 0], path_coordinate[0, 1], color='red', marker='+', s=100, )
plt.scatter(path_coordinate[-1, 0], path_coordinate[-1, 1], color='red', marker='*', s=100)

# Display
plt.colorbar(cfp)
plt.legend()
plt.savefig("f_contour_with_coordinate_descent.png")
plt.show()
