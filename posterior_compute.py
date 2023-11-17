import numpy as np

def beta_posterior(xy_info, beta_info, dim, C, H, fns, noise):

    dx, dy, T = dim
    var_x, var_y, var_c, var_h = noise
    f,g = fns
    y, x_predicted, tr_mean, w = xy_info
    B, b_size, beta_post = beta_info

    beta =1
    x_particles =1
    w = 1
    beta_post =1

    return beta, x_particles, w, beta_post