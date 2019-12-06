'''Math Utilities for System Analysis

Ethan Lew
4/24/19
elew@pdx.edu

'''

import numpy as np
import scipy.integrate as spi
import numpy.matlib as ml
from scipy.interpolate import interp1d

def gradient_fast(s, u, gradAcc=1):
    r, c = np.shape(u)
    tangent_vector = np.zeros((r, c+2))

    u_big = np.concatenate((np.array(u[:, -1], ndmin=2).T, u, np.array(u[:,0],ndmin=2).T), axis=1)
    s_big_pre = np.concatenate((s, s+2*np.pi, s+4*np.pi))
    s_big = s_big_pre[len(s)-1:2*len(s)+1]

    for ndim in range(0, 3):
        x = np.gradient(u_big[ndim, :])/ np.gradient(s_big)
        tangent_vector[ndim, :] = x

    tangent_vector = tangent_vector[:, 1:-1]

    return tangent_vector

def get_orig_f(curve, f, multiplier=1):
    nc = np.shape(curve)

    orig_f = np.zeros((nc[0], nc[1]))

    for point in range(0, np.max(np.shape(curve))):
        vec_at_p = f(curve[:, point])
        vec_at_p = vec_at_p / np.linalg.norm(vec_at_p)
        orig_f[:, point] = vec_at_p

    orig_f = multiplier * orig_f

    return orig_f

def get_ideal_f(curve, f, u, ts, ind, pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad = gradient_fast, field_multiplier=1 ):
    # Get tangent vector by obtaining the gradient from the S curve, then normalize
    tangent_vector = grad(get_s_curve(curve), curve)
    norms = (np.sum(np.abs(tangent_vector.T)**2,axis=-1)**(1./2)).T
    tangent_vector = tangent_vector / norms

    tangent_derivative = grad(get_s_curve(curve), tangent_vector)

    curve_s_vec = get_s_curve(curve)
    f_orig = get_orig_f(curve, f, field_multiplier)
    dot_p = np.diag(np.dot(f_orig.T, tangent_derivative))
    
    dot_p_cum_int = spi.cumtrapz( dot_p, curve_s_vec)
    dot_p_cum_int = np.concatenate(([0], dot_p_cum_int))

    # Calculate phi
    pre_phi = dot_p_cum_int - get_s_curve(curve)/(2*np.pi)*np.trapz(np.concatenate((dot_p, [dot_p[0]])), np.concatenate((curve_s_vec, [2*np.pi])))
    phi = pre_phi - np.mean(pre_phi)
    dot_p_ft = np.diag(np.dot(f_orig.T, tangent_vector))

    # get error
    diff_vec = (ml.repmat((phi - dot_p_ft), 3, 1) * tangent_vector)
    err = find_error(u, ts - 1, ind, pts_u, pts_init)
    diff_vec_2 = feedback_factor*ml.repmat((-err).T, 3, 1)*tangent_vector

    # Correct to get f_ideal
    f_ideal = f_orig + diff_vec - diff_vec_2

    return f_ideal

def find_error(u, ts, ind, pts_u, pts_init):
    ind_vals = (ind[0:pts_u[ts], ts] - ind[0, ts])/pts_init
    s_vals = get_s_curve(u[:, 0:pts_u[ts], ts]).T / (2*np.pi)
    return ind_vals - s_vals

def interp_1d(old_curve, pts_bad, old_pts_used, interp_acc):
    s_vec = get_s_curve(old_curve)
    u_big = ml.repmat(old_curve, 1, 3)

    s_big = np.concatenate((s_vec, s_vec+2*np.pi, s_vec+4*np.pi ))
    new_pt = 0
    old_pt = 0
    old_curve = np.concatenate((old_curve, np.array(old_curve[:, 0], ndmin=2).T), axis=1)
    new_curve = np.zeros((3, old_pts_used + np.sum(pts_bad)))

    while (old_pt < old_pts_used):
        new_curve[:, new_pt] = old_curve[:, old_pt]
        if(pts_bad[old_pt] == 1):
            s_interp = (s_big[old_pt + np.max(np.shape(s_vec))] + s_big[old_pt + np.max(np.shape(s_vec)) + 1])/2
            interp_func = interp1d(s_big, u_big, kind=interp_acc)
            interp_point = interp_func(s_interp)
            new_pt += 1
            new_curve[:, new_pt] = interp_point.T

        new_pt += 1
        old_pt += 1

    return new_curve

def get_index(old_index, pts_bad):
    new_index = np.zeros((np.max(np.shape(old_index))+np.sum(pts_bad)))

    new_pt = 0
    old_pt = 0

    A = np.array(old_index, ndmin=2).T
    B = np.array([old_index[-1]+1], ndmin=2)
    old_index = np.concatenate((A, B),axis=0)

    while (old_pt < (np.max(np.shape(old_index)) - 1) ):
        new_index[new_pt] = old_index[old_pt]
        if pts_bad[old_pt] == 1:
            new_pt += 1
            interp_pt = (old_index[old_pt] + old_index[old_pt + 1])/2
            new_index[new_pt] = interp_pt
        new_pt += 1
        old_pt += 1

    return new_index

def get_s_curve(curve):
    # TODO: Replace with get_distance
    curve_big = np.concatenate((curve, curve, curve), axis=1)
    curve_inc = np.concatenate((curve[:, 1:], curve, curve, np.array(curve[:, 0], ndmin=2).T), axis=1)

    diff = curve_big - curve_inc
    diff = diff[:, np.max(np.shape(curve)):2*np.max(np.shape(curve))]
    diff_normal = np.sqrt(np.sum(diff**2, axis=0))

    tot_len = np.sum(diff_normal)

    s_curve = np.concatenate(([0], np.cumsum(diff_normal[:-1]))) / tot_len*2*np.pi
    return s_curve

def get_distance(curve):
    curve_big = np.concatenate((curve, curve, curve), axis=1)
    curve_inc = np.concatenate((curve[:, 1:], curve, curve, np.array(curve[:, 0], ndmin=2).T), axis=1)

    diff = curve_big - curve_inc
    diff = diff[:, np.max(np.shape(curve)):2*np.max(np.shape(curve))]
    diff_normal = np.sqrt(np.sum(diff**2, axis=0))

    return diff_normal


def get_stable_eigenvectors(f, fixed_pt):
    jacob, j_err = jacobian(f, fixed_pt)
    d, v = np.linalg.eig(jacob)
    
    neg_mask = (d < 0)
    d = np.arange(1, 4, 1)
    neg_d = d[neg_mask]
    neg_v = v[:, neg_mask]

    pos_d = d[np.invert(neg_mask)]
    pos_v =  v[:, np.invert(neg_mask)]

    if len(pos_d) == 2:
        field_multiplier = 1
        v = pos_v
        d = pos_d
    else:
        field_multiplier = -1
        v = neg_v
        d = neg_d

    v = v[:, 0:2]
    d = d[0:2]

    return v, d, field_multiplier


def jacobian(f, fixed_pt):
    # Convert syntax
    x0 = fixed_pt

    # Get number of dimensions of input
    nx = len(fixed_pt)

    # Iteration params
    max_step = 100
    step_ratio = 2.0000001

    # get dimensions of f
    f0 = f(fixed_pt)
    nf = len(f0)

    # Calculate the number of steps
    rel_delta = max_step*np.reciprocal(np.power(step_ratio, np.arange(0, 26, 1)))
    n_steps = len(rel_delta)

    # Initialize the jacobian and the error
    jac = np.zeros((nf, nx))
    err = np.zeros((nf, nx))

    for i in range(0, nx):
        x0_i = x0[i]

        # Calculate the delta
        if (x0_i != 0):
            delta = x0_i*rel_delta
        else:
            delta = rel_delta
    
        # Create a second order approximation
        fdel = np.zeros((nf, n_steps))
        for j in range(0, n_steps):
            fdif = f(swap_element(x0, i, x0_i + delta[j])) - f(swap_element(x0, i, x0_i - delta[j]))
            fdel[:, j] = np.ndarray.flatten(fdif)

        derest = fdel * np.tile(0.5 * np.reciprocal(delta),[nf,1])

        # Use Romberg extrapolation to improve result
        for j in range(0, nf):
            der_romb,errest = rombex_trap(step_ratio,derest[j,:],[2, 4])
            
            nest = len(der_romb)

            trim = np.array([0, 1, 2, nest-3, nest-2, nest-1])
            tags = np.argsort(der_romb)
            der_romb_s = np.sort(der_romb)

            mask = np.ones((nest), dtype=bool)
            mask[trim] = False
            der_romb_s = der_romb_s[mask]
            tags = tags[mask]

            errest = errest[tags]

            err[j,i] = np.min(errest)
            ind = np.argmin(errest)
            jac[j,i] = der_romb_s[ind]
    return jac, err

def swap_element(vec,ind,val):
    v = np.copy(vec)
    v[ind] = val
    return v


def vec2mat(vec, n, m):
    x = np.arange(0, m, 1)
    y = np.arange(0, n, 1)

    xv, yv = np.meshgrid(x, y)

    ind = xv + yv
    mat = vec[ind]

    if n == 1:
        mat = np.transpose(mat)

    return mat

def rombex_trap(step_ratio, der_init, rombexpon):
    rombexpon = np.array(rombexpon)
    
    srinv = 1/step_ratio

    nexpon = len(rombexpon)
    rmat = np.ones((nexpon+2, nexpon+1))
 
    rmat[1, 1:3] =  np.power(srinv, rombexpon)
    rmat[2, 1:3] = np.power(srinv, 2*rombexpon)
    rmat[3, 1:3] = np.power(srinv, 3*rombexpon)

    qromb, rromb = np.linalg.qr(rmat)

    ne = len(der_init)
    rhs = vec2mat(der_init, nexpon+2, ne - (nexpon + 2))

    #rombcoefs = np.linalg.lstsq(rromb, np.matmul(np.transpose(qromb), rhs), rcond=None)[0]
    rombcoefs = np.linalg.solve(rromb, np.matmul(np.transpose(qromb), rhs))
    der_romb = np.transpose(rombcoefs[0, :])

    s = np.sqrt(np.sum((rhs - np.matmul(rmat, rombcoefs))**2,axis=0))

    #rinv = np.linalg.lstsq(rromb, np.eye(nexpon + 1), rcond=None)[0]
    rinv = np.linalg.solve(rromb, np.eye(nexpon + 1))
    cov1 = np.sum(rinv**2, axis=1)

    errest = np.transpose(s)*12.7062047361747*np.sqrt(cov1[0])

    return der_romb, errest

def next_curve(curve, f, ts_size, u, ts, index, pts_u, pts_init, getf_func=get_ideal_f, grad_func=gradient_fast, origf_func=get_orig_f, feedback_factor=-1, field_multiplier=1):
    #get_ideal_f(curve, f, u, ts, ind, pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad = gradient_fast, field_multiplier=1 ):
    k1 =  ts_size*getf_func(curve, f, u, ts, index, pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad=grad_func, field_multiplier=field_multiplier)
    k2 =  ts_size*getf_func(curve+.5*k1, f, u, ts, index, pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad=grad_func, field_multiplier=field_multiplier)
    k3 =  ts_size*getf_func(curve+.5*k2, f, u, ts, index, pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad=grad_func, field_multiplier=field_multiplier)
    k4 =  ts_size*getf_func(curve+k3, f, u, ts, index,  pts_u, pts_init, feedback_factor, get_orig_f=get_orig_f, grad=grad_func, field_multiplier=field_multiplier)

    return curve+1/6*(k1+2*k2+2*k3+k4)


if __name__ == "__main__":

    def test_function(c):
        x_data = np.array([[0.0],[0.1],[0.2]])
        y_data = 1+2*np.exp(0.75*x_data)
        return ((c[0] + c[1] * np.exp(c[2]*x_data)) - y_data)**2

    def my_vec_field(point):
        force = np.zeros((3))
        force[0]=10*(point[1]-point[0])
        force[1]=28*point[0]-point[1]-point[0] * point[2]
        force[2]=point[0]*point[1] - 8/3 * point[2]
        return force

    def magic(n):
        n = int(n)
        if n < 3:
            raise ValueError("Size must be at least 3")
        if n % 2 == 1:
            p = np.arange(1, n+1)
            return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
        elif n % 4 == 0:
            J = np.mod(np.arange(1, n+1), 4) // 2
            K = J[:, None] == J
            M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
            M[K] = n*n + 1 - M[K]
        else:
            p = n//2
            M = magic(p)
            M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
            i = np.arange(p)
            k = (n-2)//4
            j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
            M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
            M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
        return M 
  
    jac, err = jacobian(test_function, [1, 1, 1])
    #print(jac)
    #print(err)

    get_stable_eigenvectors(test_function, [-1,-1,-1])

    #print(gradient_fast(np.array([1,2,3]), np.array([[1,2,3],[-1,0,4],[1,1,0]]), 0))

    #print(get_s_curve(np.array([[8,1,6],[3,5,7],[4,9,2]])))
    m = magic(8)
    m = m.T.flatten()
    m = m[0:60]
    u = np.reshape(m, (3, 4, 5), order='F')
    ind = np.zeros((4,5), dtype='int32')
    ind[0:4, 0] = np.arange(0, 4, 1)
    pts_u = np.zeros((5), dtype='int32')
    pts_u[0] = 3
    pts_u[1] = 3
    pts_u[2] = 3
    pts_init = 3
    feedback_factor = 1
    field_multiplier = -1
    curve = magic(3)

    #print(u[:, :, 4])

    #print(get_ideal_f(curve, my_vec_field, u, 2, ind, pts_u, pts_init, feedback_factor, field_multiplier=-1))

    curve = magic(3)
    pts_bad = np.array([0, 1, 0])
    pts_used = 3
    acc = "cubic"

    print(interp_1d(curve, pts_bad, pts_used, acc))