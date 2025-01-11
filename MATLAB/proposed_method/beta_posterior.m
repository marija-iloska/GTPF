function [beta, x_particles, w, beta_post] = beta_posterior(xy_info, beta_info, fns, coeffs, noise, terms)

% Extract coeffs
H = coeffs{2};

% Functions
g = fns{1};
h = fns{2};

% States and data
[y, x_predicted, tr_mean, w] = xy_info{:};

% Beta info
[B, b_size, beta_post] = beta_info{:};

% Current settings
[dx, M, t, ln_coeff] = terms{:};

% Noise
[var_x, var_y, var] = noise{:};

% For each beta interval
for b = 1:b_size

    % For each interval
    beta = B(b);

    % Find mean and variance with beta
    new_mean = beta*x_predicted + (1 - beta)*tr_mean;
    new_var = beta^2*var_x + (1- beta)^2*var;

    % Propose new particles
    x_particles = mvnrnd(new_mean', new_var*eye(dx))';

    % Compute beta weights p(y_t | x_t(m)) p(x_t(m) | x_t-1(m)) / q(x_t(m) |, beta,  y_t )
    % Log scale
    ln_l = - (0.5/var_y)*sum( (y(:,t) - H*h(x_particles) ).^2 ) ;
    ln_t = - (0.5/var_x)*sum( (x_particles- tr_mean ).^2 ) ;
    ln_q = - (0.5/new_var)*sum( (x_particles - new_mean ).^2 ) - 0.5*dx*log(2*pi*new_var) ;

    % log likelihood + log transition - log new porposal
    ln_bn = ln_l + ln_t + ln_coeff - ln_q;

    % Scale (avoid numerical issues)
    bn = exp(ln_bn - max(ln_bn));

    % Store particles and weights for specific beta - for later use
    x_beta_store{b} = x_particles;
    weigths_store{b} = bn;

    % Compute posterior
    beta_post(b) =  beta_post(b)*sum(bn.*w);
end

% Normalize beta posterior
beta_post = beta_post./sum(beta_post);

% Avoid numerical issues
if (isnan(beta_post))
    beta_post = ones(1, b_size)./b_size;
end

% Sample beta
b_idx = datasample(1:b, 1, 'Weights', beta_post);
beta = B(b_idx);

% Particles and weights for Second Stage
x_particles = x_beta_store{b_idx};
w = weigths_store{b_idx};

end