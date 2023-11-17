from random import sample
from numpy import random
from numpy import zeros

# CREATE DATA FUNCTION
def state_space_model(dim, p, noise, fns):

    # Unpack
    dx, dy, T = dim
    px, py = p
    var_x, var_y, var_c, var_h = noise
    f,g = fns

    # dx - system dimension
    # px - sparsity in coefficient matrix
    # py - sparsity in observation matrix
    # dy - data dimension
    # T - length of time series
    # var_y - variance of observation noise
    # var_x - variance of state noise
    # var_c - variance of coefficient matrix
    # var_h - variance of observation matrix
    # f - state transition function
    # g - observation transition function

    # STATE SPACE MODEL
    # x[t] = C f( x[t-1] ) + Gaussian noise
    # y[t] = H g( x[t] ) + Gaussian noise

    # Coefficient matrix and Observation matrix
    C = random.normal(0, var_c, (dx, dx))
    H = random.normal(0, var_h, (dy, dx))

    # Randomly sample (K-p) indices to set to 0
    for j in range(dx):
        idx = sample(list(range(dx)), dx-px)
        C[j,idx] = 0

    for j in range(dy):
        idx = sample(list(range(dy)), dy - py)
        H[j,idx] = 0

    # Initialize state and data arrays
    x = zeros((dx,T))
    y = zeros((dy,T))
    x[:,0] = random.normal(0, var_x, (1,dx))
    y[:,0] = random.normal( H @ g(x[:,0]), var_y, (1,dy))

    # Create Time series
    for t in range(1,T):
        x[:,t] = C @ f(x[:,t-1]) + random.normal(0, var_x, (1, dx))
        y[:,t] = H @ g(x[:,t]) + random.normal(0, var_y, (1, dy))

    return y, x, C, H



# CREATE DATA from linear model
def linear_model(K, p, T, var_y, var_h, var_t):

    # K - number of features
    # p - model order
    # T - length of data (number of observations)
    # var_y - variance of observation noise
    # var_h - variance used when generating random feature vectors
    # var_t - variance used when generating theta

    # LINEAR MODEL
    # y = H theta + zero-mean noise

    # Feature matrix
    H = random.normal(0, var_h, (T, K))

    # Model parameter
    theta = random.normal(0, var_t, (K, 1))

    # Randomly sample (K-p) indices to set to 0
    idx = sample(list(range(0, K)), K-p)
    theta[idx] = 0

    # Create linear model data
    y = H @ theta + random.normal(0, var_y, (T, 1))

    return y, H, theta, idx
