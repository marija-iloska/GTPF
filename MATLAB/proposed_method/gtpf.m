function [x_est, choice] = gtpf(y, coeffs, fns, noise, M, B)

% Extract coeffs
C = coeffs{1};
H = coeffs{2};

% Get dimensions
T = length(y(1,:));
dy =length(y(:,1));
dx = length(C(1,:));

% Functions
g = fns{1};
h = fns{2};

% Noise
[var_x, var_y, var] = noise{:};

% Initialize
x_particles = rand(dx,M);
x_old = x_particles;
x_est = zeros(dx, T);

% Beta selection initialization
beta = 0.2;
b_size = length(B);
beta_post = ones(1,b_size)./b_size;

% Initialize weights
w = ones(1,M)/M;

% Store beta
choice = zeros(1,T);
choice(1) = beta;

% Repetitive term used in computations of weights (log form)
ln_coeff = - 0.5*dx*log(2*pi*var_y)  - 0.5*dx*log(2*pi*var_x);

% Start filter
for t=2:T

    % FIRST STAGE_________________________________________________________
    % Propose iniital particles based on model transition
    for m = 1:M
        tr_mean(:,m) = C*g(x_old(:,m));
        x_particles(:,m) = mvnrnd(tr_mean(:,m), var_x*eye(dx))';
    end

    % Predictions from proposed particles
    x_predicted = mean(x_particles,2);

    % Initialize Set of Data indices S
    states_y = 1:dy;

    % Modify proposal
    for i = 1:dx

        % If all data points have been used, reset S
        if (isempty(states_y))
            states_y = 1:dy;
        end

        % Sample at uniform a data index l
        l = datasample(states_y, 1);

        % Update S by removing sampled index
        states_y = setdiff(states_y, l);

        % For the pair (y_l, x_i(m)) find ML x_i(m*)
        for m = 1:M
            % Use the vector of predictions and replace the mth
            % particle for evaluation in the likelihood
            x_predicted(i) = x_particles(i,m);

            % Compute loglikelihood
            ln_p(m) = - (0.5/var_y)*(y(l,t) - H(l,:)*h(x_predicted) ).^2;
        end

        % Find max
        if (length(find(ln_p == max(ln_p))) ~= 1)
            % Sample at random if all weigths same
            % (Avoid numerical issues)
            m_star(i) = datasample(1:M, 1);
        else
            m_star(i) = find(ln_p == max(ln_p));
        end

        % Form proposed mean from particles with ML
        x_predicted(i) = x_particles(i, m_star(i));

    end


    % BETA posterior computation___________________________________________
    % Collection of terms needed for computation
    xy_info = {y, x_predicted, tr_mean, w};
    beta_info = {B, b_size, beta_post};
    terms = {dx, M, t, ln_coeff};

    % Call function to compute beta posterior and sample beta
    [beta, x_particles, w, beta_post] = beta_posterior(xy_info, beta_info, fns, coeffs, noise, terms);

    % Store beta sample
    choice(t) = beta;


    % SECOND STAGE_________________________________________________________
    % Propose new particles
    % We use the particles and weights stored when finding beta since
    % it's the same steps, in order to avoid double sampling
    % and evaluation

    % Resample and set weights to be equal
    idx = datasample(1:M, M, 'Weights',w);
    w = ones(1,M)/M;

    % Set new particles
    x_particles = x_particles(:,idx);
    x_old = x_particles;

    % State estimates
    x_est(:,t) = mean(x_particles,2);


end

end