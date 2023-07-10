import numpy as np
import math
import matplotlib.pyplot as plt

def black_karasinski(r0, n, dt, theta, sigma, alpha):
    """
    Simulate the Black-Karasinski short-rate model.

    Args:
        r0 (float): initial short-rate value
        n (int): number of time steps
        dt (float): time step size
        theta (ndarray): array of mean-reversion levels for each time step
        sigma (ndarray): array of volatility values for each time step
        alpha (ndarray): array of mean-reversion speed values for each time step

    Returns:
        ndarray: array of simulated short-rate values
    """
    r = np.zeros(n)
    r[0] = r0
    for i in range(1, n):
        d_r = theta[i-1] - alpha[i-1] * r[i-1]
        r[i] = r[i-1] + d_r * dt + sigma[i-1] * np.sqrt(dt) * np.random.normal()
    return r

# Example usage
r0 = 0.05
n = 1000
dt = 1/252
theta = np.linspace(0.05, 0.06, n)
sigma = np.linspace(0.03, 0.06, n)
alpha = np.ones(n) * 0.1
r = black_karasinski(r0, n, dt, theta, sigma, alpha)

plt.plot(r)
plt.show()
