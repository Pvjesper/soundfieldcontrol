import numpy as np
import scipy.linalg as splin

import aspcol.matrices as mat

import aspcol.utilities as util

def acc(Rb, Rd, reg=0):
    """
    
    Will calculate the principal generalized eigenvector of (Rb, Rd+reg*I)

    Rb and Rd are of shapes (num_freq, num_ls, num_ls)
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    num_freq = Rd.shape[0]
    num_ls = Rd.shape[-1]
    
    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd
    
    w = np.zeros((num_freq, num_ls))
    for f in range(num_freq):
        eigvals, evec = splin.eigh(Rb[f,:,:], mat.ensure_pos_def_adhoc(Rd_reg[f,:,:], verbose=True))
        w[f,:] = evec[:,-1] * np.sqrt(eigvals[-1])
    return w

def pressure_matching(Rb, Rd, rb, mu, reg=0):
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    num_freq = Rd.shape[0]
    num_ls = Rd.shape[-1]
    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd

    w = np.linalg.solve(Rb + mu*Rd_reg, rb)
    return w

def vast(Rb, Rd, rb, mu, rank, reg=0):
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    assert Rb.shape[0:2] == rb.shape[0:2]
    assert rb.shape[-1] == 1
    num_freq = Rb.shape[0]
    num_ls = Rb.shape[-1]

    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd

    eigval = []
    eigvec = []
    for f in range(num_freq):
        eva, eve = splin.eigh(Rb[f,:,:], Rd_reg[f,:,:], type=1)
        eigval.append(eva[None,:])
        eigvec.append(eve[None,:,:])
    eigval = np.concatenate(eigval, axis=0)
    eigvec = np.concatenate(eigvec, axis=0)
    eigval = eigval[:,-rank:]
    eigvec = eigvec[:,:,-rank:]
    diag_entries = 1 / (eigval + mu*np.ones(rank))
    diag_mat = np.concatenate([np.diag(diag_entries[i,:])[None,:,:] for i in range(num_freq)], axis=0)
    w = eigvec @ diag_mat @ np.moveaxis(eigvec.conj(),1,2) @ rb
    return w