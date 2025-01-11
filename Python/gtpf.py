import numpy as np
import random as rnd
from posterior_compute import beta_posterior

def generalized_tpf(model_info, dim, M, fns, noise,  beta_info):

    # Unpack data and model
    y, C, H = model_info

    # Unpack size variables and noise
    dx, dy, T = dim
    var_x, var_y, var_c, var_h, var = noise

    # Unpack functions
    f,g = fns

    # beta inits
    B, b_size, beta_post, chosen_beta = beta_info

    # Repetitive term
    ln_coeff = - 0.5 * dx * np.log(2 * np.pi * var_y) - 0.5 * dx * np.log(2 * np.pi * var_x)

    # Initialize
    x_particles = np.zeros((dx, M))
    tr_mean = np.zeros((dx, M))
    w = np.ones(M) / M
    x_old = x_particles
    x_est = np.zeros((dx, T))
    covX = var_x * np.eye(dx)

    for t in range(1, T):

        m_star = []
        # FIRST STAGE
        for m in range(M):
            tr_mean[:, m] = C @ f(x_old[:, m])
            x_particles[:, m] = np.random.multivariate_normal(tr_mean[:, m], covX)

        # Predictions from proposed particles
        x_predicted = np.mean(x_particles, axis=1)

        # Initialize Set of Data indices S
        states_y = np.arange(dy)

        # Modify proposal
        for i in range(dx):

            ln_p = []

            # If all data points have been used
            if states_y.size == 0:
                states_y = np.arange(dy)

            # Sample at uniform a data index l
            l = rnd.sample(list(states_y), 1)

            # Update S by removing sampled index
            states_y = np.setdiff1d(states_y, l)

            for m in range(M):
                # Use vector of predictions and replace mth particle
                x_predicted[i] = x_particles[i, m]

                # Compute loglikelihood
                ln_p.append(float(-(0.5 / var_y) * ((y[l, t] - H[l, :] @ g(x_predicted))) ** 2))

            # Find max
            if float(min(ln_p) == max(ln_p)):
                # Sample at random if all weights same (Avoid numerical issues)
                m_star.append(rnd.sample(list(range(M)), 1)[0])
            else:
                ln_p_array = np.array(ln_p)
                m_star.append(int(np.where(ln_p_array == max(ln_p_array))[0][0]))

            # Form proposed mean from particles with ML
            x_predicted[i] = x_particles[i, m_star[i]]

            # BETA posterior computation___________________________________________
            # Collection of terms needed for computation
            xy_info = [y, x_predicted, tr_mean, w]
            beta_info = [B, b_size, beta_post]
            terms = [M, t, ln_coeff]

            # Call function to compute beta posterior and sample beta
            [beta, x_particles, w, beta_post] = beta_posterior(xy_info, beta_info, dim, C, H, fns, noise, terms)

            # Store beta sample
            chosen_beta.append(beta)

            # Resample and set weights to be equal
            idx = np.random.choice(list(range(M)), M, p=w)
            w = np.ones(M) / M

            # Set new particles
            x_particles = x_particles[:, idx]
            x_old = x_particles

            # State estimates
            x_est[:, t] = np.mean(x_particles, axis=1)

    return x_est, chosen_beta