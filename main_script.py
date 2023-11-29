import numpy as np
from synthetic_data import state_space_model
import matplotlib.pyplot as plt
from gtpf import generalized_tpf

# System dimension, data size, time series length
dx = 20
dy = 10
T = 70

# sparsity in transition and observation coeffs
px = 3
py = 1

# Variance noise
var_x = 0.1
var_y = 0.1
var_c = 1
var_h = 1
var = 0.1

# Transition and Observation functions
f = lambda x: 1/(1 + np.exp(-x))
g = lambda x: x

# Package relevant inputs
fns = [f, g]
noise = [var_x, var_y, var_c, var_h, var]
p = [px, py]
dim = [dx, dy, T]

# CREATE SSM ======================================
y, x, C, H = state_space_model(dim, p, noise, fns)

# GTPF settings
M = 100

# Beta selection
B = np.linspace(0.01, 1, 100)
b_size = len(B)
beta_post = np.ones((1, b_size))[0]/b_size

# Chosen beta
chosen_beta = [0.2]
beta_info = [B, b_size, beta_post, chosen_beta]
model_info = [y, C, H]

# GTPF run =============================================================
x_est, beta = generalized_tpf(model_info, dim, M, fns, noise,  beta_info)


# PLOT =================================================================
j = int(np.random.choice(list(range(dx)), 1))
plt.plot(np.arange(T),x_est[j,:])
plt.plot(np.arange(T), x[j,:])
plt.show()