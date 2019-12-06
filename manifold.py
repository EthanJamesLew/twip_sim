''' Stable Manifold Solver

Ethan Lew
4/25/19
elew@pdx.edu
'''

import numpy as np
from math_utils import gradient_fast, get_ideal_f, get_stable_eigenvectors, next_curve, get_ideal_f, get_orig_f, get_distance, interp_1d, get_index

### Config the Solver ###

## Dynamical system description
def my_vec_field(point):
        force = np.zeros((3))
        force[0]=10*(point[1]-point[0])
        force[1]=28*point[0]-point[1]-point[0] * point[2]
        force[2]=point[0]*point[1] - 8/3 * point[2]
        return force

## Fixed point in the simulation
fixed_pt = np.array([[0],[0],[0]], ndmin = 2, dtype='float64')

## Manifold evolution time
time_total = 90

## Distance of initial ring of points
radius = 3

## Accuracy params
time_resolution = 5
gradient_func = gradient_fast
newf_func = get_ideal_f
pts_init = 2**5
pts_max = 2**10
step_size = 0.1
feedback_factor = 0
max_distance_percent = 4
min_distance_percent = 0.25

# Setup Interps
interp_func = interp_1d
interp_acc = 'cubic'


### Initialize the Solver ###

# Setup the fixed point, its eigenvectors, its eigenvalues and field multiplier
stab_vecs, stab_vals, field_multiplier = get_stable_eigenvectors(my_vec_field, fixed_pt)
eigenvector_1 = stab_vecs[:, 0]
eigenvector_2 = stab_vecs[:, 1]
eigenvalue_1 = stab_vals[0]
eigenvalue_2 = stab_vals[1]

# Manifold Results Allocation
time_steps = np.ceil(time_total/step_size).astype(int)
u = np.zeros((3, pts_max, time_steps ))
index = np.zeros((pts_max, time_steps), dtype='int32')

# Keep track of how many members of u correspond to the stable manifold
pts_used = np.zeros((time_steps), dtype='int32')

# Manifold Results Initialization - Create a ring of points
for pt in range(0, pts_init):
    u[:, pt, 0] = fixed_pt.flatten() + eigenvalue_1*np.sin((pt)/pts_init*2*np.pi)*eigenvector_1 + eigenvalue_2*np.cos((pt)/pts_init*2*np.pi)*eigenvector_2

pts_used[0] = pts_init
index[0:pts_used[0], 0] = np.arange(0, pts_used[0], 1)
# Setup distances
max_distance = max_distance_percent*np.mean(get_distance(u[:, 0:pts_used[0], 0]))
min_distance = min_distance_percent*np.mean(get_distance(u[:, 0:pts_used[0], 0]))

### Run the Solver ###
for ts in range(1, time_steps):
    # Assume points are used
    pts_used[ts] = pts_used[ts-1]

    # Update u, index
    u[:, 0:pts_used[ts], ts] = next_curve(u[:, 0:pts_used[ts-1], ts-1], my_vec_field, step_size, u, ts, index, pts_used, pts_init, getf_func=get_ideal_f, grad_func=gradient_func, origf_func=get_orig_f, feedback_factor=feedback_factor, field_multiplier=field_multiplier)
    index[0:pts_used[ts], ts] = index[0:pts_used[ts-1], ts-1]

    pts_bad = (get_distance(u[:, 0:pts_used[ts], ts]) > max_distance)

    # Too far away error TODO: Bug in intern indices
    if (np.sum(pts_bad) > 0):
        old_pts_used = pts_used[ts-1]
        temp_pts_used = pts_used
        temp_pts_used[ts-1] = pts_used[ts-1] + np.sum(pts_bad)

        # Interpolate
        temp_old_curve = interp_func(u[:, 0:old_pts_used, ts-1], pts_bad, old_pts_used, interp_acc)

        # Create temp index
        temp_idx = np.copy(index)

        temp_idx[0:temp_pts_used[ts-1], ts-1] = get_index(temp_idx[0:old_pts_used, ts-1], pts_bad)
        pts_used[ts] = temp_pts_used[ts - 1]

        # Get new curve and index
        C = next_curve(temp_old_curve, my_vec_field, step_size, u, ts, temp_idx, temp_pts_used, pts_init, getf_func=get_ideal_f, grad_func=gradient_func, origf_func=get_orig_f, feedback_factor=feedback_factor, field_multiplier=field_multiplier)
        u[:, 0:pts_used[ts], ts] = C
        index[0:pts_used[ts], ts] = temp_idx[0:temp_pts_used[ts-1], ts-1]
    

    # Too close error
    pts_bad = (get_distance(u[:, 0:pts_used[ts], ts]) < min_distance)
    if (np.sum(pts_bad) > 0):
        old_curve = u[:, 0:pts_used[ts], ts]
        new_curve = old_curve[:, pts_bad == 0]

        pad_val = np.shape(u[:, 0:pts_used[ts], ts]) - np.shape(new_curve)
        u[:, 0:pts_used[ts], ts] = np.pad(new_curve, ((0, pad_val[0]),(0, pad_val[1])), 'constant') 

        old_idx = index[0:pts_used[ts], ts]
        new_idx = old_idx[pts_bad == 0]

        pad_val = np.shape(index[0:pts_used[ts], ts]) - np.shape(new_idx)
        index[0:pts_used[ts], ts] = np.pad(new_idx, ((0, pad_val[0]),(0, pad_val[1])), 'constant') 
        pts_used[ts] = pts_used[ts] - np.sum(pts_bad)
    

import scipy.io as sio
sio.savemat('manifold.mat', {'u':u, 'index': index, 'pts_used': pts_used})

        

