import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import hypothesis as hyp

import numpy as np
import soundfieldcontrol.szc.sinr_opt as sinropt



def get_random_spatial_cov(num_zones, num_ls):
    rng = np.random.default_rng()
    rank = rng.integers(1, 2*num_ls)
    R = np.zeros((num_zones, num_ls, num_ls), dtype=complex)

    for r in range(rank):
        v = rng.normal(size=(num_ls, 1))
        R += v @ v.conj().T
    return R

def get_random_beamformer(num_zones, num_ls):
    rng = np.random.default_rng()
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls))
    return w

def get_random_beamformer_normalized(num_zones, num_ls):
    rng = np.random.default_rng()
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls))
    w /= np.sqrt(np.sum(np.abs(w)**2, axis=-1, keepdims=True))
    return w

def get_random_noise_pow(num_zones):
    rng = np.random.default_rng()
    return rng.uniform(low=1e-3, high=10, size=num_zones)

def get_random_sinr_targets(num_zones):
    rng = np.random.default_rng()
    return rng.uniform(low=0.3, high=5, size=num_zones)

def get_random_pow_vec(num_zones):
    rng = np.random.default_rng()
    return rng.uniform(low=1e-3, high=10, size=num_zones)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
)
def test_sinr_uplink_specific_and_link_gain_formulation_equal(num_zones, num_ls):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    p = get_random_pow_vec(num_zones)
    #sinr_targets = get_random_sinr_targets(num_zones)

    sinr_val1 = sinropt._sinr_uplink_comparison(w, R, p, noise_pow)
    sinr_val2 = sinropt.sinr_uplink(sinropt.apply_power_vec(w, p), R, noise_pow)
    assert np.allclose(sinr_val1, sinr_val2)