import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

# Define majority function truth table
X = np.reshape([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], (8, 3, 1))
Y = np.reshape([[0], [0], [0], [1], [0], [1], [1], [1]], (8, 1, 1))

# Define the network
network = [
    Dense(3, 4),
    Tanh(),
    Dense(4, 1),
    Tanh()
]

# Train the network
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

# Decision boundary plotting
points = []
for x in np.linspace(0, 1, 10):
    for y in np.linspace(0, 1, 10):
        for z in np.linspace(0, 1, 10):  # Added third dimension
            output = predict(network, np.array([[x], [y], [z]]))
            points.append([x, y, z, output[0, 0]])

points = np.array(points)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot with color mapping based on output
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap="coolwarm")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Decision Boundary for Majority Function (3 Inputs)")

plt.show()
