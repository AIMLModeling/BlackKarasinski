import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def black_karasinski(r0, theta, alpha, sigma, T, n):
    dt = T / n
    rt = np.zeros(n+1)
    rt[0] = r0
    dwt = np.random.normal(size=n) * np.sqrt(dt)
    for i in range(n):
        rt[i+1] = rt[i] + theta[i]*dt - alpha[i]*rt[i]*dt + sigma[i]*dwt[i]
    return rt

def calibration_obj(x, r0, r_market, T, n):
    theta, alpha, sigma = x[:n], x[n:2*n], x[2*n:]
    r_model = black_karasinski(r0, theta, alpha, sigma, T, n)
    return np.sum((r_model - r_market)**2)

# set up parameters
r0 = 0.05
n = 100
T = 1
t = np.linspace(0, T, n+1)
r_market = 0.05 + 0.005*np.random.randn(n+1) # market rates with some noise
print(f"r_market:{r_market}")

with open(r'C:\Documents\MyOwnModel\BlackKarasinski/marketrate.txt', 'w') as fp:
    for number in r_market:
        # write each item on a new line
        fp.write("%s\n" % number)
    print('Market rates saved.')

# initial guess for parameters
theta0 = np.ones(n) * 0.05
alpha0 = np.ones(n) * 0.3
sigma0 = np.ones(n) * 0.1
x0 = np.concatenate([theta0, alpha0, sigma0])

# calibration
res = minimize(calibration_obj, x0, args=(r0, r_market, T, n))

# retrieve calibrated parameters
theta, alpha, sigma = res.x[:n], res.x[n:2*n], res.x[2*n:]
# plot market rates and calibrated model rates
plt.plot(t, r_market, label='Market Rates')
plt.plot(t, black_karasinski(r0, theta, alpha, sigma, T, n), label='Model Rates')
plt.legend()
plt.show()
