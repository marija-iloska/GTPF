clear all
close all
clc

% GENERALIZED TWO STAGE PARTICLE FILTER_________________________________
% To obtain a statistical result:
% Set R to the desired number of runs
% Uncomment line 12, comment out line 13, comment out PLOT section
R = 1;

tic
%parfor run = 1:R
for run = 1:R
    % Create data_____________________________________________________
    % Time series length
    %T = 70;
    T = 500;  %check convergence of beta

    % State and Observation dimension
    dx = 100;
    dy = 60;
    

    % State, Observation, and Proposal noise
    var_x = 0.1;
    var_y = 1;
    var = var_x; 
    noise = {var_x, var_y, var};

    % State and observation range (and percent sparsity)
    range = {[-1, 1, 0.3],[-3, 3, 0]} ;

    % State transition and observation functions
    g = @(x) 1./(1+exp(-x));
    h = @(x) x;
    fns = {g, h};


    % Create data
    [x, y, C, H] = create_data(dx, dy, T, var_x, var_y, fns, range);

    coeffs = {C, H};

    % GTPF settings________________________________________________________
    % Number of particles
    M = 100;

    % Beta selection
    B = 0.1 : 0.01 : 1;

    % Run filter
    % GTPF 
    [x_gtpf, choice] = gtpf(y, coeffs, fns, noise, M, B);

    % Get MSE
    mse_gtpf(run) = sum(sum( (x_gtpf - x).^2 ))/(dx*T);

end
toc

% Mean MSE of all filters
mean(mse_gtpf)


%% PLOTTING_________________________________________________________________
% Specific run to see. Only if one filter was run R=1.

% Plot settings_______________________________________________________
% Tracking: Plot a random section in time ( length of 60 for better view )
len = 50;
t0 = T - len;% datasample(1:T-len, 1);
time_plot = t0:T;

% Font sizes, linewidth, colors
load util/plot_settings.mat


% Plot trajectory of random state
figure;
j = datasample(1:dx, 1);
plot(time_plot, x(j,time_plot), 'k', 'LineWidth',lwd)
hold on
plot(time_plot, x_gtpf(j,time_plot), 'Color', 'm', 'LineStyle','--', 'LineWidth',sz)
set(gca, 'FontSize', fsz-5)
xlabel('Time', 'FontSize',fsz)
ylabel('State', 'FontSize',fsz)
legend('True State', 'GTPF', 'FontSize', fsz)


% Plot sampled values of beta with time (entire time series length)
figure;
plot(1:T, choice, 'linewidth',lwd)
ylim([0,1])
xlabel('Time', 'FontSize',fsz)
ylabel('\beta', 'FontSize', fsz+10)
