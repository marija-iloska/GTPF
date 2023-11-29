import numpy as np
from numpy.random import choice

def beta_posterior(xy_info, beta_info, dim, C, H, fns, noise, terms):

    dx, dy, T = dim
    var_x, var_y, var_c, var_h, var = noise
    f,g = fns
    y, x_predicted, tr_mean, w = xy_info
    B, b_size, beta_post = beta_info
    M, t, ln_coeff = terms

    weights_store = []
    x_beta_store = []

    # For each beta interval
    for b in range(b_size):

        # Reset
        x_particles = x_predicted.reshape(dx, 1)

        # Each beta sample
        beta = B[b]

        # Find mean and variance with beta
        new_mean = beta * x_predicted.reshape(dx,1) + (1 - beta)*tr_mean
        new_var = beta**2 *var_x + (1 - beta)**2*var

        # Propose new particles
        for m in range(M):
            samples = np.random.multivariate_normal(new_mean[:,m], new_var*np.eye(dx)).reshape(dx,1)
            x_particles = np.hstack([x_particles, samples])

        # Remove initialized column
        x_particles = np.delete(x_particles, 0,1)

        # Compute beta weights p( y_t | x_t(m) )  p(x_t(m) | x_t-1(m)) / q(x_t(m) |, beta,  y_t )
        # Log scale
        ln_l = - (0.5/var_y)*np.sum( (y[:,t].reshape(dy,1) - H @ g(x_particles) )**2 , axis=0 )
        ln_t = - (0.5/var_x)*np.sum( (x_particles - tr_mean)**2, axis=0 )
        ln_q = - (0.5/new_var)*np.sum( (x_particles - new_mean)**2, axis =0) - 0.5*dx*np.log(2*np.pi*new_var)

        # Log Likelihood
        ln_bn = ln_l + ln_t - ln_coeff - ln_q

        # Scale (avoid numerical underflow) and normalize
        bn = np.exp(ln_bn - max(ln_bn))
        bn = bn / np.sum(bn)

        # Store particles and weights for specific beta - for later use
        x_beta_store.append(x_particles)
        weights_store.append(bn)

        # Compute posterior
        beta_post[b] = beta_post[b]*np.sum(bn * w)

    # Normalize posterior
    beta_post = beta_post/np.sum(beta_post)

    # Avoid Numerical issues - reset
    if any(beta_post == np.nan):
        beta_post = np.ones((1,b_size))

    # Sample beta
    b_idx = int(choice(list(range(b_size)), 1, p=beta_post))
    beta = float(B[b_idx])

    # Particles and weights chosen for SECOND STAGE
    x_particles = x_beta_store[b_idx]
    w = weights_store[b_idx]/np.sum(weights_store[b_idx])

    return beta, x_particles, w, beta_post