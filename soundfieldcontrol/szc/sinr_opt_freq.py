import cvxpy as cp
import numpy as np
import scipy.linalg as splin

import aspcol.matrices as mat

import soundfieldcontrol.szc.sinr_opt as sinropt


"""
Per frequency, there is one beamforming vector, with a complex value per loudspeaker

The most important variables are
- w is of shape (num_freqs, num_zones, num_sources)
- R is of shape (num_freqs, num_zones, num_sources, num_sources)
- noise_pow is of shape (num_freqs, num_zones) or (num_zones)
- sinr_constraint is of shape (num_freqs, num_zones) or (num_zones)


"""



def satisfies_sinr_constraint(bf_vec, R, noise_pow, sinr_constraint):
    return sinr_margin(bf_vec, R, noise_pow, sinr_constraint) >= 0

def sinr_margin(bf_vec, R, noise_pow, sinr_constraint):
    """
    bf_vec is of shape (num_freqs, num_zones, num_sources)
    R is of shape (num_freqs, num_zones, num_sources, num_sources)
    noise_pow is of shape (num_freqs, num_zones) or (num_zones)
    sinr_constraint is of shape (num_freqs, num_zones) or (num_zones)

    Calculates how much SINR can be reduced before violating the constraint
        frac{w_k^H R_k w_k}{ sum_{i \neq k} w_i^H R_k w_i + sigma_k^2} \geq sinr_constraint_k
        for each zone k. 
        The constraint is satisfied if all values are 0 or higher

    returns an array of shape (num_freqs, num_zones)
    """
    if bf_vec.ndim == 2:
        bf_vec = bf_vec[None,:,:]
    if R.ndim == 3:
        R = R[None,:,:,:]

    num_freqs = bf_vec.shape[0]
    num_zones = bf_vec.shape[1]
    num_sources = bf_vec.shape[2]
    assert bf_vec.ndim == 3
    assert R.shape == (num_freqs, num_zones, num_sources, num_sources)

    margin = np.zeros((num_freqs, num_zones))
    for f in range(num_freqs):
        margin[f,:] = _sinr_margin(bf_vec[f,:,:,None], R[f,:,:,:], noise_pow, sinr_constraint)
    return margin

def _sinr_margin(w, R, noise_pow, sinr_constraint):
    num_zones = w.shape[0]
    margin = np.zeros((num_zones))

    for k in range(num_zones):
        signal_power = np.real_if_close(np.squeeze(w[k,:,:].conj().T @ R[k,:,:] @ w[k,:,:]))
        interference_power = 0
        for i in range(num_zones):
            if i != k:
                interference_power += np.real_if_close(np.squeeze(w[i,:,:].conj().T @ R[k,:,:] @ w[i,:,:]))
        interference_power += noise_pow[k]
        margin[k] = (signal_power / interference_power) - sinr_constraint[k]
    return margin





def sinr_constrained_pow_min_downlink(R_values, avg_noise_pow, sinr_targets):
    """
    Solves the SINR constrained power minimization problem for transmit beamforming
        via semidefinite relaxation. Returns the optimal matrix, not the optimal 
        beamformer vector. Use any of the select_solution functions for that. 

    Solves the problem independently for num_freqs values gives for each parameter

    R_values is complex matrices of shape (num_freqs, num_sources, num_sources)
        which is the spatial covariance matrices for each receiver (or sound zone)
        is defined as R = sum_m h_m h_m^H for the m receivers of the zone. 

    avg_noise_pow is positive real values of shape (num_freqs, num_zones)
    sinr_targets is positive real values of shape (num_freqs, num_zones)
    """
    num_freqs = R_values.shape[0]
    num_zones = R_values.shape[1]
    num_speakers = R_values.shape[2]
    #assert all([r.shape[0] == num_freqs for r in R_values])
    #assert all([num_zones == len(v) for v in (avg_noise_pow, sinr_targets)])
    assert all([(r.shape[1] == num_speakers and r.shape[2] == num_speakers) for r in R_values])
    
    #ctrl_filters = np.zeros((num_freqs, num_speakers, num_zones), dtype=complex)
    opt_mat = np.zeros((num_freqs, num_zones, num_speakers, num_speakers), dtype=complex)
    W = [cp.Variable((num_speakers, num_speakers), complex=True) for _ in range(num_zones)]
    R = [cp.Parameter((num_speakers, num_speakers), complex=True) for _ in range(num_zones)] 
    sinr_target = [cp.Parameter(pos=True) for _ in range(num_zones)] 
    noise_pow = [cp.Parameter(pos=True) for _ in range(num_zones)] 
    constraints = []

    for k in range(num_zones):
        new_constr = cp.trace(R[k] @ W[k]) - sinr_target[k] * noise_pow[k]
        for i in range(num_zones):
            if k != i:
                new_constr -= sinr_target[k] * cp.trace(R[k] @ W[i])
        constraints.append(cp.real(new_constr) >= 0)
                    
    for k in range(num_zones):
        constraints.append(W[k] >> 0)
        constraints.append(W[k] == W[k].H)

    obj = cp.Minimize(cp.abs(cp.sum([cp.trace(W_z) for W_z in W])))
    prob = cp.Problem(obj, constraints)

    for f in range(num_freqs):
        for k in range(num_zones):
            R[k].value = R_values[f,k,:,:]
            sinr_target[k].value = sinr_targets[k]
            noise_pow[k].value = avg_noise_pow[k]
        prob.solve(verbose=False)
        print(prob.status)
        for k in range(num_zones):
            opt_mat[f,k,:,:] = W[k].value
    return opt_mat




def sinr_constrained_pow_min_uplink(R_values, avg_noise_pow, sinr_targets):
    """
    Solves the SINR constrained power minimization problem for transmit beamforming
        via semidefinite relaxation. Returns the optimal matrix, not the optimal 
        beamformer vector. Use any of the select_solution functions for that. 

    Solves the problem independently for num_freqs values gives for each parameter

    R_values is complex matrices of shape (num_freqs, num_sources, num_sources)
        which is the spatial covariance matrices for each receiver (or sound zone)
        is defined as R = sum_m h_m h_m^H for the m receivers of the zone. 

    avg_noise_pow is positive real values of shape (num_freqs, num_zones)
    sinr_targets is positive real values of shape (num_freqs, num_zones)
    """
    num_freqs = R_values.shape[0]
    num_zones = R_values.shape[1]
    num_speakers = R_values.shape[2]
    #assert all([r.shape[0] == num_freqs for r in R_values])
    #assert all([num_zones == len(v) for v in (avg_noise_pow, sinr_targets)])
    assert all([(r.shape[1] == num_speakers and r.shape[2] == num_speakers) for r in R_values])
    R_normalized = np.zeros_like(R_values)

    for f in range(num_freqs):
        for k in range(num_zones):
            R_normalized[f,k,:,:] = R_values[f,k,:,:] / avg_noise_pow[k]
    
    #ctrl_filters = np.zeros((num_freqs, num_speakers, num_zones), dtype=complex)
    opt_mat = np.zeros((num_freqs, num_zones, num_speakers, num_speakers), dtype=complex)
    W = [cp.Variable((num_speakers, num_speakers), complex=True) for _ in range(num_zones)]
    R = [cp.Parameter((num_speakers, num_speakers), complex=True) for _ in range(num_zones)] 
    sinr_target = [cp.Parameter(pos=True) for _ in range(num_zones)] 
    constraints = []

    for k in range(num_zones):
        new_constr = cp.trace(R[k] @ W[k]) - sinr_target[k]
        for i in range(num_zones):
            if k != i:
                new_constr -= sinr_target[k] * cp.trace(R[i] @ W[k])
        constraints.append(cp.real(new_constr) >= 0)
                    
    for k in range(num_zones):
        constraints.append(W[k] >> 0)
        constraints.append(W[k] == W[k].H)

    obj = cp.Minimize(cp.abs(cp.sum([cp.trace(W_z) for W_z in W])))
    prob = cp.Problem(obj, constraints)

    for f in range(num_freqs):
        for k in range(num_zones):
            R[k].value = R_normalized[f,k,:,:]
            sinr_target[k].value = sinr_targets[k]
        prob.solve(verbose=False)
        print(prob.status)
        for k in range(num_zones):
            opt_mat[f,k,:,:] = W[k].value
    return opt_mat





def link_gain(w, R):
    """
    returns the matrix G as defined in 
        'A general duality theory for uplink and downlink beamforming'
        which is defined as G_ki = w_k^H R_i w_k
    """
    num_freqs = w.shape[0]
    num_zones = w.shape[1]
    G = np.zeros((num_freqs, num_zones, num_zones))

    for f in range(num_freqs):
        for k in range(num_zones):
            for i in range(num_zones):
                G[f,k,i] = np.real_if_close(np.squeeze(w[f,k,:,None].T.conj() @ R[f,i,:,:] @ w[f,k,:,None]))
    return G



def power_assignment_minmax_downlink(w, R, sinr_targets, noise_pow, max_pow):
    """
    Solves the optimization max_p min_i \frac{SINR_i}{gamma_i}, constrained such that
        the power equals the given max power, sum_i p_i = P_{max}.
    
    For more details, check out Schubert & Bosche - Solution of the 
        Multiuser Downlink Beamforming Problem With Individual SINR Constraints
    """
    #w = normalize_beamformer(w)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1)
    #gain_mat = np.moveaxis(link_gain(w, R), -1, -2)
    gain_mat = link_gain(w, R)
    num_freqs = w.shape[0]
    num_sources = w.shape[1]
    p = np.zeros((num_freqs, num_sources))
    c = np.zeros((num_freqs))

    for f in range(w.shape[0]):
        p[f,:], c[f] = sinropt._power_alloc_minmax(gain_mat[f,:,:], sinr_targets, noise_pow, max_pow)

    return p, c

def power_assignment_minmax_uplink(w, R, sinr_targets, noise_pow, max_pow):
    """
    Solves the optimization max_p min_i \frac{SINR_i}{gamma_i}, constrained such that
        the power equals the given max power, sum_i p_i = P_{max}.
    
    For more details, check out Schubert & Bosche - Solution of the 
        Multiuser Downlink Beamforming Problem With Individual SINR Constraints
    """
    #w = normalize_beamformer(w)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1)
    gain_mat = np.moveaxis(link_gain(w, R), -1, -2)
    #gain_mat = link_gain(w, R)
    num_freqs = w.shape[0]
    num_sources = w.shape[1]
    p = np.zeros((num_freqs, num_sources))
    c = np.zeros((num_freqs))

    for f in range(w.shape[0]):
        p[f,:], c[f] = sinropt._power_alloc_minmax(gain_mat[f,:,:], sinr_targets, noise_pow, max_pow)

    return p, c





# def power_assignment_minmax_downlink(w, R, sinr_targets, noise_pow, max_pow):
#     """
#     Solves the optimization max_p min_i \frac{SINR_i}{gamma_i}, constrained such that
#         the power equals the given max power, sum_i p_i = P_{max}.
    
#     For more details, check out Schubert & Bosche - Solution of the 
#         Multiuser Downlink Beamforming Problem With Individual SINR Constraints
#     """
#     #w = normalize_beamformer(w)
#     gain_mat = link_gain(w, R)
#     for f in range(w.shape[0]):
#         sinropt.power_alloc_maxmin_dl(gain_mat[f,:,:], sinr_targets, noise_pow, max_pow)

#     #if_mat = sinr._interference_matrix(gain_mat)
#     #D = _signal_diag_matrix(gain_mat, sinr_targets)
#     #coupling_mat = _extended_coupling_matrix_downlink(D, if_mat, noise_pow, max_pow)

#     num_freqs = w.shape[0]
#     num_zones = w.shape[1]

#     capacity = np.zeros(num_freqs)
#     p = np.zeros((num_freqs, num_zones))
#     for f in range(num_freqs):
#         eigvals, eigvec = splin.eig(coupling_mat[f,:,:], left=False, right=True)
#         max_ev_idx = np.argmax(np.real(eigvals))
#         p[f,:] = eigvec[:-1, max_ev_idx] / eigvec[-1, max_ev_idx]
#         capacity[f] = np.real_if_close(1 / eigvals[max_ev_idx])

#     return p, capacity

# def _interference_matrix(gain_mat):
#     """
#     The inference matrix is the link gain but with diagonal elements
#         set to zero. 
        
#     According to the definition written at the function 
#         link_gain, to get the sum interference for a zone k, you take 
#         np.sum(if_mat, axis=0)[k] (to be verified)
#     """
#     if_mat = np.zeros_like(gain_mat)
#     if_mat[...] = gain_mat

#     num_zones = gain_mat.shape[-1]
#     for z in range(num_zones):
#         if_mat[:,z,z] = 0
#     return if_mat

# def _signal_diag_matrix(gain_mat, sinr_targets):
#     num_zones = gain_mat.shape[-1]
#     signal_mat = np.zeros_like(gain_mat)
#     for z in range(num_zones):
#         signal_mat[:,z,z] = sinr_targets[z] / gain_mat[:,z,z]
#     return signal_mat

# def _extended_coupling_matrix_downlink(signal_mat, if_mat, noise_pow, max_pow):
#     """
#     See documentation for power_assignment_minmax_downlink
#     Returns equation (12) from Schubert & Boche
#     """
#     num_freq = signal_mat.shape[0]
#     num_zones = signal_mat.shape[1]
#     ext_dim = num_zones + 1

#     noise_pow = noise_pow[None,:,None]           #make into column vector
#     max_pow_vec = (1 / max_pow) * np.ones((1, 1, num_zones))

#     coupling_mat = np.zeros((num_freq, ext_dim, ext_dim))
#     coupling_mat[:, :-1, :-1] = signal_mat @ if_mat    #top left block
#     coupling_mat[:, :-1, -1:] = signal_mat @ noise_pow                         #top right block
#     coupling_mat[:, -1:, :-1] = max_pow_vec @ signal_mat @ if_mat                      #bottom left block
#     coupling_mat[:, -1:, -1:] = max_pow_vec @ noise_pow                          #bottom right block
#     return coupling_mat










def power_allocation_downlink_cvxpy(bf_vec, R_values, noise_pow_val, sinr_targets):
    """
    bf_vec is of shape (num_freqs, num_zones, bf_len)
    R is of shape (num_freqs, num_zones, bf_len, bf_len)
    noise_pow is of shape (num_zones)

    bf_len is num_loudspeakers in the frequency domain case, and
        num_loudspeakers*filter_length in the time domain case
    
    Should be applied as sqrt{p_i} w_i for a given beamformer vector w_i (for zone i). 
    returns the optimal power allocation in the shape of (num_freqs, num_zones)
    """
    assert bf_vec.ndim == 3
    assert R_values.ndim == 4
    num_freqs = R_values.shape[0]
    num_zones = R_values.shape[1]
    num_sources = R_values.shape[2]

    p = [cp.Variable(pos=True) for _ in range(num_zones)]
    wRw = [[cp.Parameter(pos=True) for _ in range(num_zones)] for _ in range(num_zones)] # index [i][j] is w_i R_j w_i
    #w = [cp.Parameter((num_speakers, 1), complex=True) for _ in range(num_zones)]
    sinr_target = [cp.Parameter(pos=True) for _ in range(num_zones)]
    noise_pow = [cp.Parameter(pos=True) for _ in range(num_zones)] 
    constraints = []

    for k in range(num_zones):
        new_constr = p[k] * wRw[k][k] - sinr_target[k] * noise_pow[k]
        for i in range(num_zones):
            if k != i:
                new_constr -= p[i] * wRw[i][k] * sinr_target[k]
        constraints.append(new_constr >= 0)
                    
    obj = cp.Minimize(cp.sum([pow_vec for pow_vec in p]))
    prob = cp.Problem(obj, constraints)

    #make unit vectors
    #bf_vec /= np.sqrt(np.sum(np.abs(bf_vec)**2, axis=-1))[:,:,None]

    opt_power = np.zeros((num_freqs, num_zones)) 
    for f in range(num_freqs):
        for k in range(num_zones):
            for i in range(num_zones):
                wRw[i][k].value = np.squeeze(np.real_if_close(bf_vec[f,i,:,None].conj().T @ R_values[f,k,:,:] @ bf_vec[f,i,:,None]))
            sinr_target[k].value = sinr_targets[k]
            noise_pow[k].value = noise_pow_val[k]
        prob.solve(verbose=False)
        print(prob.status)
        for k in range(num_zones):
            opt_power[f,k] = p[k].value
    return opt_power




