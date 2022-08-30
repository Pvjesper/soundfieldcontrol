import numpy as np
import scipy.linalg as splin

import aspcol.matrices as mat

import aspcol.utilities as util

def acc(Rb, Rd):
    """
    cov_bright and cov_dark is the szc spatial covariance matrices
     = H.T @ H, where H is a convolution matrix made up of the RIR, 
     summed over the appropriate microphones for bright and dark zones. 

    num_ls is the number of loudspeakers, and therefore the number 
    of blocks (in each axis) that the cov matrices consists of
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[0] == Rd.shape[1]
    assert Rb.ndim == 2
    #Rd += 1e-4*np.eye(Rd.shape[0])
    eigvals, evec = splin.eigh(Rb, mat.ensure_pos_def_adhoc(Rd, verbose=True))
    
    #ir = evec[:,-1].reshape(1, num_ls, -1)
    #norm = np.sqrt(np.sum(ir**2))
    #ir /= norm
    #ir *= 1e4
    return evec[:,-1] * np.sqrt(eigvals[-1])

@util.measure("ACC")
def acc_all_zones(R):
    """
    R is of shape (num_zones, num_zones, bf_len, bf_len)
    R[k,i,:,:] means spatial covariance associated with RIRs 
        of zone k, and audio signal of zone i
    
    returns beamformer vector of shape (num_zones, bf_len)
    """
    num_zones = R.shape[0]
    bf_len = R.shape[-1]
    assert R.shape == (num_zones, num_zones, bf_len, bf_len)

    Rd = np.zeros((bf_len, bf_len))
    w = np.zeros((num_zones, bf_len))

    for k in range(num_zones):
        Rd.fill(0)
        Rb = R[k,k,:,:]
        for i in range(num_zones):
            if i != k:
                Rd += R[i,k,:,:]
        w[k,:] = acc(Rb, Rd)

    return w
