import numpy as np
from synthetic_data import state_space_model


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

# CREATE SSM
y, x, C, H = state_space_model(dx, px, T, dy, py, var_x, var_y, var_c, var_h, f, g)
