import numpy as np
import random as rnd
from synthetic_data import state_space_model
from posterior_compute import beta_posterior

dx = 7
dy = 3
T = 40
px = 3
py = 1
var_x = 0.1
var_y = 0.1
var_c = 1
var_h = 1
f = lambda x: 1/(1 + np.exp(-x))
g = lambda x: x

fns = [f, g]
noise = [var_x, var_y, var_c, var_h]
p = [px, py]
dim = [dx, dy, T]

# CREATE SSM
y, x, C, H = state_space_model(dim, p, noise, fns)


# GTPF settings
M = 100

# Beta selection
B = np.linspace(0.01, 1, 100)
b_size = len(B)
beta_post = np.ones((1, b_size))/b_size

# Chosen beta
chosen_beta = [0.2]

# Repetitive term
ln_coeff = - 0.5*dx*np.log(2*np.pi*var_y) - 0.5*dx*np.log(2*np.pi*var_x)

# Initialize
x_particles = np.zeros((dx,M))
tr_mean = np.zeros((dx,M))
ln_p = []
w = np.ones(M)/M
x_old = x_particles
x_est = np.zeros((dx,T))
covX = var_x*np.eye(dx)
covY = var_y*np.eye(dy)
m_star = []

for t in range(1,T):

    # FIRST STAGE
    for m in range(M):
        tr_mean[:,m] = C @ f(x_old[:,m])
        x_particles[:,m] = np.random.multivariate_normal(tr_mean[:,m], covX)

    # Predictions from proposed particles
    x_predicted = np.mean(x_particles, axis=1)

    # Initialize Set of Data indices S
    states_y = np.arange(dy)

    # Modify proposal
    for i in range(dx):

        # If all data points have been used
        if states_y.size == 0:
            states_y = np.arange(dy)

        # Sample at uniform a data index l
        l = rnd.sample(list(states_y), 1)

        # Update S by removing sampled index
        states_y = np.setdiff1d(states_y, l)

        for m in range(M):
            # Use vector of predictions and replace mth particle
            x_predicted[i] = x_particles[i,m]

            # Compute loglikelihood
            ln_p.append( -(0.5/var_y)*( (y[l,t] - H[l,:] @ g(x_predicted)) )**2 )

        # Find max
        if len(ln_p.index(ln_p == max(ln_p))) != 1:
            # Sample at random if all weights same (Avoid numerical issues)
            m_star.append(rnd.sample(list(range(M)), 1))
        else:
            m_star.append(ln_p.index(ln_p == max(ln_p)))

        # Form proposed mean from particles with ML
        x_predicted[i] = x_particles[i, m_star[i]]

        #BETA posterior computation___________________________________________
        #Collection of terms needed for computation
        xy_info = [y, x_predicted, tr_mean, w]
        beta_info = [B, b_size, beta_post]
        terms = [M, t, ln_coeff]

        #Call function to compute beta posterior and sample beta
        [beta, x_particles, w, beta_post] = beta_posterior(xy_info, beta_info, dim, C, H, fns, noise, terms)

        #Store beta sample
        chosen_beta.append(beta)

        #Resample and set weights to be equal
        idx = np.random.choice(list(range(M)), M, w)
        w = np.ones(M)/M

        #Set new particles
        x_particles = x_particles[:, idx]
        x_old = x_particles

        #State estimates
        x_est[:, t] = np.mean(x_particles, axis=1)

# GTPF call
