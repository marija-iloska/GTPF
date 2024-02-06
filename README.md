# Generalized Two-Stage Particle Filter for High Dimensions
# ICASSP 2024

### Particle filters (PF)
PFs are online algorithms that estimate a hidden process from noisy data. 
They operate on two main steps per step: i) propose particles from proposal, and ii) compute their weights.
Standard PFs do not work well for systems of high dimensions.

### Two-Stage PF
A two-stage PF for high dimensions was proposed that modifies the proposal distribution by tempering particles, but could only
be applied to systems with observation model that has separable equations, and a manually chosen tempering coefficient.

## Generalized Two-Stage PF
We propose a generalized two-stage particle filter (GTPF) that can be applied to all models, and computes the distribution of
the tempering coefficient. 

### Reference paper: http://tinyurl.com/ieeexplore-iloska-gtpf
(Reference for original two-stage PF can be found with-in).

### Code
1. main_script.py  - script to run the proposed filter
   - generates synthetic data according to user settings
   - calls and runs filter
   - plots the estimation of a random state trajectory
  
2. synthetic_data.py - module that can generate state-space model or linear model data
3. gtpf.py - module that stores the code of the proposed filter
4. posterior_compute.py - module that computes the posterior of the tempering parameter beta
