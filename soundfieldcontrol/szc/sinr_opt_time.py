import cvxpy as cp
import numpy as np
import scipy.linalg as splin

import aspcol.matrices as mat
import aspcol.utilities as util

import soundfieldcontrol.szc.sinr_opt as sinropt


"""
Each control filter is filt_len samples long, and there is one per loudspeaker
    The beamformer vector w is therefore bf_len = num_ls * filt_len long. 

Here are the variables that are used for most of the functions

- w is the beamformer vectors,
    size (num_zones, bf_len)
- R is the spatial covariance matrices, 
    R[k,i,:,:] is the spatial covariance matrix associated with 
    the RIRs of zone k, and the audio signal of zone i
    size (num_zones, num_zones, bf_len, bf_len)
- noise_pow is the average noise power at the microphones in each zone. 
    size (num_zones,)
- sinr_targets is the SINR that should be achieved. User sets this value
    size (num_zones,)
    generally gamma by notation in the papers

"""




@util.measure("Downlink")
def sinr_constrained_pow_min_downlink(R_values, avg_noise_pow, sinr_targets):
    """
    Solves the SINR constrained power minimization problem for transmit beamforming
        via semidefinite relaxation. Returns the optimal matrix, not the optimal 
        beamformer vector. Use any of the select_solution functions for that. 

    R_values is the spatial covariance matrices R_{ki}, where k is the zone index of 
        the microphones and i is the zone index of the audio signal
        The array should have has shape (num_zones, num_zones, bf_len, bf_len)
        where bf_len = num_ls*ir_len, and is indexed as R_{ki} = R[k,i,:,:]

    avg_noise_pow is positive real values of shape (num_zones)
    sinr_targets is positive real values of shape (num_zones)

    returns opt_mat which has shape (num_zones, bf_len, bf_len)
    """
    num_zones = R_values.shape[1]
    mat_size = R_values.shape[2]
    assert R_values.shape == (num_zones, num_zones, mat_size, mat_size)
    assert len(avg_noise_pow) == num_zones
    assert len(sinr_targets) == num_zones

    opt_mat = np.zeros((num_zones, mat_size, mat_size), dtype=float)
    W = [cp.Variable((mat_size, mat_size), PSD=True) for _ in range(num_zones)]
    R = [[cp.Parameter((mat_size, mat_size), PSD=True) for _ in range(num_zones)] for _ in range(num_zones)]
    sinr_target = [cp.Parameter(pos=True) for _ in range(num_zones)]
    noise_pow = [cp.Parameter(pos=True) for _ in range(num_zones)]
    constraints = []

    for k in range(num_zones):
        new_constr = cp.trace(R[k][k] @ W[k]) - sinr_target[k] * noise_pow[k]
        for i in range(num_zones):
            if k != i:
                new_constr -= sinr_target[k] * cp.trace(R[k][i] @ W[i])
        constraints.append(new_constr >= 0)
                    
    for k in range(num_zones):
        constraints.append(W[k] == W[k].T)

    obj = cp.Minimize(cp.sum([cp.trace(W_z) for W_z in W]))
    prob = cp.Problem(obj, constraints)


    for k in range(num_zones):
        for i in range(num_zones):
            R[k][i].value = R_values[k,i,:,:]
        sinr_target[k].value = sinr_targets[k]
        noise_pow[k].value = avg_noise_pow[k]
    prob.solve(verbose=False, ignore_dpp=True)
    print(prob.status)
    for k in range(num_zones):
        opt_mat[k,:,:] = W[k].value
    return opt_mat



@util.measure("Uplink")
def sinr_constrained_pow_min_uplink(R_values, avg_noise_pow, sinr_targets):
    """
    Solves the SINR constrained power minimization problem for receive beamforming
        via semidefinite relaxation. Returns the optimal matrix, not the optimal 
        beamformer vector. Use any of the select_solution functions for that. 

    R_values is the spatial covariance matrices R_{ki}, where k is the zone index of 
        the microphones and i is the zone index of the audio signal
        The array should have has shape (num_zones, num_zones, num_ls*ir_len, num_ls*ir_len)
        and is indexed as R_{ki} = R[k,i,:,:]

    avg_noise_pow is positive real values of shape (num_zones)
    sinr_targets is positive real values of shape (num_zones)
    """
    num_zones = R_values.shape[1]
    mat_size = R_values.shape[2]
    assert R_values.shape == (num_zones, num_zones, mat_size, mat_size)
    assert len(avg_noise_pow) == num_zones
    assert len(sinr_targets) == num_zones


    assert np.allclose(avg_noise_pow, np.ones_like(avg_noise_pow))
    # R_normalized = np.zeros_like(R_values)
    # for k in range(num_zones):
    #     R_normalized[k,:,:,:] = R_values[k,:,:,:] / avg_noise_pow[k]

    opt_mat = np.zeros((num_zones, mat_size, mat_size), dtype=float)
    W = [cp.Variable((mat_size, mat_size), PSD=True) for _ in range(num_zones)]
    R = [[cp.Parameter((mat_size, mat_size), PSD=True) for _ in range(num_zones)] for _ in range(num_zones)]
    sinr_target = [cp.Parameter(pos=True) for _ in range(num_zones)]
    constraints = []

    for k in range(num_zones):
        new_constr = cp.trace(R[k][k] @ W[k]) - sinr_target[k]
        for i in range(num_zones):
            if k != i:
                new_constr -= sinr_target[k] * cp.trace(R[i][k] @ W[k])
        constraints.append(new_constr >= 0)
                    
    for k in range(num_zones):
        constraints.append(W[k] == W[k].T)

    obj = cp.Minimize(cp.sum([cp.trace(W_z) for W_z in W]))
    prob = cp.Problem(obj, constraints)


    for k in range(num_zones):
        for i in range(num_zones):
            R[k][i].value = R_values[k,i,:,:]
        sinr_target[k].value = sinr_targets[k]
    prob.solve(verbose=False, ignore_dpp=True)
    print(prob.status)
    for k in range(num_zones):
        opt_mat[k,:,:] = W[k].value
    return opt_mat


def sinr_constrained_pow_min_uplink_separated(R_values, noise_pow, sinr_targets):
    """
    Solves the SINR constrained power minimization problem for receive beamforming
        via semidefinite relaxation. Returns the optimal matrix, not the optimal 
        beamformer vector. Use any of the select_solution functions for that. 

    R_values is the spatial covariance matrices R_{ki}, where k is the zone index of 
        the microphones and i is the zone index of the audio signal
        The array should have has shape (num_zones, num_zones, num_ls*ir_len, num_ls*ir_len)
        and is indexed as R_{ki} = R[k,i,:,:]

    avg_noise_pow is positive real values of shape (num_zones)
    sinr_targets is positive real values of shape (num_zones)
    """
    num_zones = R_values.shape[1]
    mat_size = R_values.shape[2]
    assert R_values.shape == (num_zones, num_zones, mat_size, mat_size)
    assert len(noise_pow) == num_zones
    assert len(sinr_targets) == num_zones

    assert np.allclose(noise_pow, np.ones_like(noise_pow))
    #R_normalized = np.zeros_like(R_values)
    #for k in range(num_zones):
    #    R_normalized[k,:,:,:] = R_values[k,:,:,:] / noise_pow[k]

    opt_mat = np.zeros((num_zones, mat_size, mat_size), dtype=float)
    R_temp = np.zeros((mat_size, mat_size))

    W = cp.Variable((mat_size, mat_size), PSD=True)
    R_tot = cp.Parameter((mat_size, mat_size), PSD=True)
    sinr_target = cp.Parameter(pos=True)

    constraints = [cp.trace(W @ R_tot) >= sinr_target]
    for k in range(num_zones):
        constraints.append(W[k] == W[k].T)

    obj = cp.Minimize(cp.trace(W))
    prob = cp.Problem(obj, constraints)

    for k in range(num_zones):
        R_temp.fill(0)
        
        for i in range(num_zones):
            if k == i:
                R_temp += R_values[k,k,:,:]
            else:
                R_temp -= sinr_targets[k] * R_values[i,k,:,:]

        R_temp = mat.ensure_pos_semidef(R_temp)

        R_tot.value = R_temp
        sinr_target.value = sinr_targets[k]
        prob.solve(verbose=False, ignore_dpp=True)
        print(prob.status)
        opt_mat[k,:,:] = W.value
    return opt_mat


# def power_allocation_uplink(w, R, noise_pow, sinr_targets):
#     """
#     Minimizes the power when the SINR equals the sinr_targets

#     Closed form derivations and proofs given in 
#         'A general duality theory for uplink and downlink beamforming'
    
#     """
#     num_zones = w.shape[0]
#     gain_mat = sinropt._link_gain(w, R)
#     if_mat = sinropt.interference_matrix(gain_mat)
#     sig_mat = sinropt.signal_diag_matrix(gain_mat, sinr_targets)

#     system_mat = np.eye(num_zones) - sig_mat @ if_mat
#     answer_mat = sig_mat @ noise_pow[:,None]

#     p = splin.solve(system_mat, answer_mat)
#     return p[:,0]

# def power_allocation_downlink(w, R, noise_pow, sinr_targets):
#     """
#     Minimizes the power when the SINR equals the sinr_targets

#     Closed form derivations and proofs given in 
#         'A general duality theory for uplink and downlink beamforming'
    
#     returns the power vector of shape (num_zones)
#     """
#     num_zones = w.shape[0]
#     gain_mat = sinropt._link_gain(w, R)
#     if_mat = sinropt._interference_matrix(gain_mat)
#     sig_mat = sinropt._signal_diag_matrix(gain_mat, sinr_targets)

#     system_mat = np.eye(num_zones) - sig_mat @ if_mat.T
#     answer_mat = sig_mat @ noise_pow[:,None]

#     p = splin.solve(system_mat, answer_mat)
#     return p[:,0]






# def link_gain(w, R):
#     """
#     returns the matrix G as defined in 
#         'A general duality theory for uplink and downlink beamforming'
#         which is defined as G_ki = w_k^T R_ik w_k


#     """
#     num_zones = w.shape[0]
#     G = np.zeros((num_zones, num_zones))

#     for k in range(num_zones):
#         for i in range(num_zones):
#             G[k,i] = np.squeeze(w[k,:,None].T @ R[i,k,:,:] @ w[k,:,None])
#     return G


def link_gain_sdr(W, R):
    """
    Same as link_gain, but accepts a matrix directly the SDR optimzation 
        instead of a beamformer vector
    """
    num_zones = W.shape[0]
    G = np.zeros((num_zones, num_zones))

    for k in range(num_zones):
        for i in range(num_zones):
            G[k,i] = np.trace(W[k,:,:] @ R[i,k,:,:])
    return G


# def interference_matrix_direct(w, R):
#     """
#     Probably not needed, can be deleted

#     w is of shape (num_zones, bf_len)
#     R is of shape (num_zones, num_zones, bf_len, bf_len)

#     returns matrix of shape (num_zones, num_zones)
#         defined as mat[i,k] = w^T_k R_i w_k     if k != i
#         and 0 on the diagonal

#     Follows the definition needed for power_allocation_uplink
#     """
#     num_zones = w.shape[0]
#     if_mat = np.zeros((num_zones, num_zones))

#     for k in range(num_zones):
#         for i in range(num_zones):
#             if k != i:
#                 if_mat[i,k] = np.squeeze(w[k,:,None].T @ R[k,i,:,:] @ w[k,:,None])
#     return if_mat

# def signal_matrix_direct(w, R,sinr_targets):
#     """
#      Probably not needed, can be deleted

#     returns diagonal matrix of shape (num_zones, num_zones)
#         defined as mat[k,k] = w^T_k R_k w_k 

#     Follows the definition needed for power_allocation_uplink
#     """
#     num_zones = w.shape[0]
#     sig_mat = np.zeros((num_zones, num_zones))
#     for k in range(num_zones):
#         sig_mat[k,k] = sinr_targets[k] / np.squeeze(w[k,:,None].T @ R[k,k,:,:] @ w[k,:,None])
#     return sig_mat

# def sinr_margin_new(w, R, noise_pow, sinr_targets):
#     G = link_gain(w, R)
#     raise NotImplementedError

# def sinr_margin_downlink(bf_vec, R, noise_pow, sinr_constraint):
#     """
#     bf_vec is of shape (num_zones, bf_len)
#     R is of shape (num_zones, num_zones, bf_len, bf_len)m
#     noise_pow is of shape  (num_zones)
#     sinr_constraint is of shape (num_zones)

#     Calculates how much SINR can be reduced before violating the constraint
#         frac{w_k^H R_k w_k}{ sum_{i \neq k} w_i^H R_k w_i + sigma_k^2} \geq sinr_constraint_k
#         for each zone k. 
#         The constraint is satisfied if all values are 0 or higher

#     returns an array of shape (num_zones)
#     """
#     num_zones = bf_vec.shape[0]
#     bf_len = bf_vec.shape[1]
#     assert bf_vec.ndim == 2
#     assert R.shape == (num_zones, num_zones, bf_len, bf_len)

#     margin = np.zeros((num_zones))

#     for k in range(num_zones):
#         signal_power = np.squeeze(bf_vec[k,:,None].T @ R[k,k,:,:] @ bf_vec[k,:,None])
#         interference_power = 0
#         for i in range(num_zones):
#             if i != k:
#                 interference_power += np.squeeze(bf_vec[i,:,None].T @ R[k,i,:,:] @ bf_vec[i,:,None])
#         interference_power += noise_pow[k]
#         margin[k] = (signal_power / interference_power) - sinr_constraint[k]
#     return margin


def sinr_margin_downlink_sdr(W, R, noise_pow, sinr_constraint):
    """
    W is shape (num_zones, bf_len, bf_len)
    R is shape (num_zones, num_zones, bf_len, bf_len)
    noise_pow is shape (num_zones)
    sinr_constraint is shape (num_zones)
    

    returns the margin per constraint (one per zone),
        an array of shape (num_zones)
    """
    num_zones = W.shape[0]
    margin = np.zeros(num_zones)

    
    for k in range(num_zones):
        sinr = 0
        sinr += np.trace(W[k,:,:] @ R[k,k,:,:])
        sinr -= noise_pow[k] * sinr_constraint[k]
        for i in range(num_zones):
            if k != i:
                sinr -= sinr_constraint[k] * np.trace(W[i,:,:] @ R[k,i,:,:])
        margin[k] = sinr
    return margin








# def capacity_downlink(w, R, gamma, noise_pow):
#     """
    
#     For a given w, R, gamma, noise_pow, calculates the 
#         C_{DL} = max_{p} min_{i} \frac{SINR_i}{gamma_i}
#         with SINR being the downlink SINR

#     Using the definitions from Schubert & Bosche - solution to downlink...

#     return a scalar with the value for C_{DL}
#     """
#     sinr_values = sinr_downlink(w, R, noise_pow)
    


# def capacity_uplink(w, R, gamma, noise_pow):
#     """
    
#     For a given w, R, gamma, noise_pow, calculates the 
#         C_{UL} = max_{p} min_{i} \frac{SINR_i}{gamma_i}
#         with SINR being the uplink SINR

#     If noise_pow = np.ones(num_zones), then this should be equal to
#         capacity_downlink for any w, gamma, if the proof applies in
#         the time domain case too. 

#     return a scalar with the value for C_{DL}
#     """
#     pass



# def power_assignment_minmax_downlink(w, R, sinr_targets, noise_pow, max_pow):
#     """
#     Solves the optimization max_p min_i \frac{SINR_i}{gamma_i}, constrained such that
#         the power equals the given max power, sum_i p_i = P_{max}.
    
#     For more details, check out Schubert & Bosche - Solution of the 
#         Multiuser Downlink Beamforming Problem With Individual SINR Constraints
#     """
#     assert np.allclose(np.linalg.norm(w, axis=-1), 1)
#     gain_mat = sinropt._link_gain(w, R)
#     return sinropt._power_alloc_minmax(gain_mat, sinr_targets, noise_pow, max_pow)

# def power_assignment_minmax_uplink(w, R, sinr_targets, noise_pow, max_pow):
#     assert np.allclose(np.linalg.norm(w, axis=-1), 1)
#     gain_mat = sinropt._link_gain(w, R).T
#     return sinropt._power_alloc_minmax(gain_mat, sinr_targets, noise_pow, max_pow)






















def power_allocation_downlink_cvxpy(bf_vec, R_values, noise_pow_val, sinr_targets):
    """
    Probably overkill when there is a closed-form solution. 
    Can be used to compare against power_allocation_downlink. They should
    give identical answers

    bf_vec is of shape (num_zones, bf_len)
    R is of shape (num_zones, num_zones, bf_len, bf_len)
    noise_pow is of shape (num_zones)

    bf_len is num_loudspeakers*filter_length 
    
    Should be applied as sqrt{p_i} w_i for a given beamformer vector w_i (for zone i). 
    returns the optimal power allocation in the shape of (num_freqs, num_zones)
    """
    #num_freqs = R_values.shape[0]
    num_zones = R_values.shape[0]
    bf_len = R_values.shape[-1]
    assert R_values.shape == (num_zones, num_zones, bf_len, bf_len)
    assert bf_vec.shape == (num_zones, bf_len)

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
                new_constr -= p[i] * wRw[k][i] * sinr_target[k]
        constraints.append(new_constr >= 0)
                    
    obj = cp.Minimize(cp.sum([pow_vec for pow_vec in p]))
    prob = cp.Problem(obj, constraints)

    #make unit vectors
    #bf_vec /= np.sqrt(np.sum(np.abs(bf_vec)**2, axis=-1))[:,:,None]

    opt_power = np.zeros((num_zones)) 
    for k in range(num_zones):
        for i in range(num_zones):
            wRw[k][i].value = np.squeeze(bf_vec[k,:,None].T @ R_values[k,i,:,:] @ bf_vec[k,:,None])
        sinr_target[k].value = sinr_targets[k]
        noise_pow[k].value = noise_pow_val[k]
    prob.solve(verbose=False)
    print(prob.status)
    for k in range(num_zones):
        opt_power[k] = p[k].value
    return opt_power
