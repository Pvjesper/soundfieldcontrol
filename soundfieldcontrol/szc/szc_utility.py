
import numpy as np
import scipy.linalg as splin
import copy

import ancsim.signal.filterdesign as fd
import ancsim.signal.filterclasses as fc
import ancsim.signal.freqdomainfiltering as fdf

import aspcol.matrices as mat
import aspcol.correlation as cr


def beamformer_vec_to_ir(bf_vec, num_ls):
    """
    bf_vec is of shape (num_ls*filt_len, 1) or (num_zones, num_ls*filt_len)

    returns ir of shape (1, num_ls, filt_len) or (num_zones, num_ls, filt_len)
    """
    bf_vec = np.squeeze(bf_vec)
    if bf_vec.ndim == 1:
        return bf_vec.reshape(num_ls, -1)
    elif bf_vec.ndim == 2:
        num_zones = bf_vec.shape[0]
        return bf_vec.reshape(num_zones, num_ls, -1)
    else:
        raise ValueError("Wrong dimensions for bf_vec")


def freq_to_time_beamformer(w, num_freqs):
    w = fd.insertNegativeFrequencies(w, True)
    ir,_ = fd.firFromFreqsWindow(w, num_freqs-1)
    ir = np.moveaxis(ir, 0,1)
    ir = ir[:1,:,:]
    return ir



def fpaths_to_spatial_cov(arrays, fpaths, source_name, zone_names):
    """
    - arrays is ArrayCollection object
    - fpaths is the frequency domain RIRs, see function get_fpaths() in this module
    - source name is a string for a source in arrays
    - zone_names is a list of strings to microphones in arrays

    returns a spatial covariance matrix R of shape (num_freqs, num_zones, num_src, num_src)
    """
    num_sources = arrays[source_name].num
    num_freqs = fpaths[source_name][zone_names[0]].shape[0]
    num_zones = len(zone_names)
    #num_zones = np.sum([arrays[z].num for z in zone_names])

    #H = np.zeros((num_freqs, num_zones, num_receivers, num_sources), dtype=complex)
    H = [fpaths[source_name][zone_name] for zone_name in zone_names]

    R = np.zeros((num_freqs, num_zones, num_sources, num_sources), dtype=complex)
    for k in range(num_zones):
        for f in range(num_freqs):
            R[f,k,:,:] = mat.ensure_pos_semidef(H[k][f,:,:].T.conj() @ H[k][f,:,:])
        num_mics = H[k][f].shape[0]
        R[:,k,:,:] /= num_mics
    return R

def get_fpaths(arrays, num_freqs, samplerate):
    """
    returns a dictionary with frequency domain RIRs 
        each entry has shape 
    """
    freqs = fd.getFrequencyValues(num_freqs, samplerate)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(fdf.fftWithTranspose(path, n=num_freqs),1,2)[:num_real_freqs,...]
    return fpaths, freqs


def paths_to_spatial_cov(arrays, source_name, zone_names, sources, filt_len, num_samples, margin=None):
    """
    sources should be a list of the audio sources associated with each zone
        naturally the list of zone names and sources should be of the same length

    by default it will use as many samples as possible (only remove rir_len-1 samples 
        in the beginning since they haven't had time to propagate properly). 
        margin can be supplied if a specific number of samples should be removed instead.
        might give questionable result if you set margin to less than rir_len-1.
    

    Returns K^2 spatial covariance matrices R_{ki}, where k is the zones index of 
        the microphones and i is the zone index of the audio signal
        The returned array has shape (num_zones, num_zones, num_ls*ir_len, num_ls*ir_len)
        and is indexed as R_{ki} = R[k,i,:,:]
    """
    num_sources = arrays[source_name].num
    num_zones = len(zone_names)
    assert len(sources) == num_zones

    R = np.zeros((num_zones, num_zones, filt_len*num_sources, filt_len*num_sources), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            R[k,i,:,:] = mat.ensure_pos_semidef(spatial_cov(arrays.paths[source_name][zone_names[k]], sources[i], filt_len, num_samples, margin=margin))
    return R

def paths_to_spatial_cov_delta(arrays, source_name, zone_names, filt_len):
    """
    See info for paths_to_spatial_cov
    """
    num_sources = arrays[source_name].num
    num_zones = len(zone_names)

    R = np.zeros((num_zones, num_zones, filt_len*num_sources, filt_len*num_sources), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            R[k,i,:,:] = mat.ensure_pos_semidef(spatial_cov_delta(arrays.paths[source_name][zone_names[k]], filt_len))
    return R

def rir_to_szc_cov(rir, ctrlfilt_len):
    """
    Takes a RIR of shape (num_ls, num_mic, ir_len) and
    turns it into the time domain sound zone control spatial
    covariance matrix made up of the blocks R_l1l2 = H_l1^T H_l2, 
    where H_l is a convolution matrix with RIRs associated with
     loudspeaker l

    output is of shape (num_ls*ctrlfilt_len, num_ls*ctrlfilt_len)
    
    """
    num_ls = rir.shape[0]
    num_mics = rir.shape[1]
    R = np.zeros((ctrlfilt_len*num_ls, ctrlfilt_len*num_ls))
    for m in range(num_mics):
        for l1 in range(num_ls):
            for l2 in range(num_ls):
                h1 = rir[l1,m,:]
                h2 = rir[l2,m,:]
                H1 = splin.convolution_matrix(h1, ctrlfilt_len ,mode="full")
                H2 = splin.convolution_matrix(h2, ctrlfilt_len ,mode="full")
                R[l1*ctrlfilt_len:(l1+1)*ctrlfilt_len, 
                    l2*ctrlfilt_len:(l2+1)*ctrlfilt_len] += H1.T @ H2
    R /= num_mics
    return R


def spatial_cov(ir, source, filt_len, num_samples, margin=None):
    """
    ir is the room impulse responses of shape (num_ls, num_mic, ir_len)
        from all loudspeakers to one of the zones. 

    source is a source object with get_samples(num_samples) method, which returns
        the audio signal that should be reproduced in the sound zones

    by default it will use as many samples as possible (only remove rir_len-1 samples 
        in the beginning since they haven't had time to propagate properly). 
        margin can be supplied if a specific number of samples should be removed instead.
        might give questionable result if you set margin to less than rir_len-1.
    

    The returned spatial covariance matrix is of size (num_ls*filt_len, num_ls*filt_len)

    """
    ir_len = ir.shape[-1]
    num_sources = ir.shape[0]
    if margin is None:
        margin = ir_len - 1

    rir_filt = fc.createFilter(ir=ir, sumOverInput=False)
    source = copy.deepcopy(source)
    in_sig = source.get_samples(num_samples+margin)
    in_sig = np.tile(in_sig, (num_sources, 1))
    out_sig = rir_filt.process(in_sig)
    out_sig = out_sig[...,margin:]
    out_sig = np.moveaxis(out_sig, 0, 1)
    R = cr.corr_matrix(out_sig, out_sig, filt_len, filt_len)
    return R



def spatial_cov_delta(ir, filt_len):
    """
    Calculates the spatial covariance matrices as if the input signal is a delta
    ir is the default shape given by arrays.paths (num_ls, num_mics, ir_len)
    
    Multiplies result with ir_len, because corr_matrix divides with the number of samples,
        but here that shouldn't be done to keep the correct scaling, as the values are just
        the ir and not filtered samples
        
    The returned spatial covariance matrix is of size (num_ls*filt_len, num_ls*filt_len)
    """
    ir_len = ir.shape[-1]
    ir = np.moveaxis(ir, 1, 0)
    R = cr.corr_matrix(ir, ir, filt_len, filt_len) * ir_len
    return R







# def rir_to_rir_cov(rir, sum_over_mics):
#     """
#     rir should have shape (num_ls, num_mic, rir_len)

#     if the stacked column rir vector is h, then this function returns h @ h.T

#     if sum_over_mics is True, the output is a block matrix with LxL blocks
#     each with rir_len x rir_len values. 
#     """
#     num_ls = rir.shape[0]
#     num_mic = rir.shape[1]
#     rir_len = rir.shape[2]
#     if sum_over_mics:
#         rir_vec_len = num_ls * rir_len
#         #rir_cov = np.zeros((rir_vec_len, rir_vec_len))
#         rir_vec = np.moveaxis(rir, 0,1).reshape(num_mic, rir_vec_len)
#         return rir_vec.T @ rir_vec / num_mic
#     else:
#         raise NotImplementedError
    

# def spatial_cov_delta(Hb, Hd, d, filt_len):
#     """Calculates the spatial covariance matrices 
#         as if the input signal is a delta"""
#     assert Hb.shape[0] == Hd.shape[0]
#     assert Hb.shape[2] == Hd.shape[2]
#     #num_micb = Hb.shape[1]
#     #num_micd = Hd.shape[1]
#     #num_speaker = Hb.shape[0]
#     #path_len = Hb.shape[2]

#     Hb = np.moveaxis(Hb, 1, 0)
#     Hd = np.moveaxis(Hd, 1, 0)
#     d = d[:,None,:]
#     #d = np.moveaxis(d, 1, 0)
#     Rb = mat.corr_matrix(Hb, Hb, filt_len, filt_len)
#     Rd = mat.corr_matrix(Hd, Hd, filt_len, filt_len)
#     rb = mat.corr_matrix(Hb, d, filt_len, 1)
#     return Rb, Rd, rb