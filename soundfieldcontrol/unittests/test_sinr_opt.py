import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import hypothesis as hyp

import numpy as np
import soundfieldcontrol.szc.sinr_opt as sinropt

SEED = 5

def get_random_spatial_cov(num_zones, num_ls):
    rng = np.random.default_rng(SEED)
    rank = rng.integers(1, num_ls)
    R = np.zeros((num_zones, num_ls, num_ls), dtype=complex)
    for k in range(num_zones):
        for r in range(rank):
            v = rng.normal(size=(num_ls, 1)) + 1j*rng.normal(size=(num_ls, 1))
            R[k,:,:] += v @ v.conj().T

    assert np.allclose(R, np.moveaxis(R.conj(), 1,2))
    return R

def get_random_beamformer(num_zones, num_ls):
    rng = np.random.default_rng(SEED+10)
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls))
    return w

def get_random_beamformer_normalized(num_zones, num_ls):
    rng = np.random.default_rng(SEED+235)
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls)) + 1j*rng.normal(scale=rng.uniform(high=5),size=(num_zones, num_ls))
    w /= np.sqrt(np.sum(np.abs(w)**2, axis=-1, keepdims=True))
    return w

def get_random_noise_pow(num_zones):
    rng = np.random.default_rng(SEED+2354)
    return rng.uniform(low=1e-3, high=10, size=num_zones)

def get_random_sinr_targets(num_zones):
    rng = np.random.default_rng(SEED+12)
    return rng.uniform(low=0.3, high=5, size=num_zones)

def get_random_pow_vec(num_zones):
    rng = np.random.default_rng(SEED+2358)
    return rng.uniform(low=1e-3, high=10, size=num_zones)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_sinr_uplink_specific_and_link_gain_formulation_equal(num_zones, num_ls):
    w_norm = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    p = get_random_pow_vec(num_zones)
    w = sinropt.apply_power_vec(w_norm, p)
    #sinr_targets = get_random_sinr_targets(num_zones)

    sinr_val = sinropt.sinr_uplink(w, R, noise_pow)
    #sinr_val1 = sinropt._sinr_uplink_comparison(w_norm, R, p, noise_pow)
    sinr_val2 = sinropt._sinr_uplink_comparison_2(w, R, noise_pow)
    
    assert np.allclose(sinr_val, sinr_val2)



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_sinr_downlink_specific_and_link_gain_formulation_equal(num_zones, num_ls):
    w_norm = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    p = get_random_pow_vec(num_zones)
    w = sinropt.apply_power_vec(w_norm, p)
    #sinr_targets = get_random_sinr_targets(num_zones)

    sinr_val = sinropt.sinr_downlink(w, R, noise_pow)
    sinr_val2 = sinropt._sinr_downlink_comparison(w, R, noise_pow)
    
    assert np.allclose(sinr_val, sinr_val2)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
    st.floats(min_value=1, max_value=1000),
)
def test_power_alloc_minmax_equal_to_max_pow(num_zones, num_ls, max_pow):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = get_random_noise_pow(num_zones)
    sinr_targets = get_random_sinr_targets(num_zones)

    p_dl, _ = sinropt.power_alloc_minmax_downlink(w, R, noise_pow, sinr_targets, max_pow)
    assert np.allclose(np.sum(p_dl), max_pow)

    p_ul, _ = sinropt.power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, max_pow)
    assert np.allclose(np.sum(p_ul), max_pow)



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_power_alloc_qos_zero_sinr_margin_downlink(num_zones, num_ls):
    num_test_attempts = 100
    feasible = False
    for i in range(num_test_attempts):
        w = get_random_beamformer_normalized(num_zones, num_ls)
        R = get_random_spatial_cov(num_zones, num_ls)
        noise_pow = get_random_noise_pow(num_zones)
        rng = np.random.default_rng(SEED+12)
        sinr_targets = rng.uniform(low=0.05, high=0.3, size=num_zones)
        if sinropt._power_alloc_qos_is_feasible(sinropt.link_gain_downlink(w, R), sinr_targets):
            feasible = True
            break
    assert feasible
    
    p_dl = sinropt.power_alloc_qos_downlink(w, R, noise_pow, sinr_targets)
    margin_dl = sinropt.sinr_margin_downlink(sinropt.apply_power_vec(w, p_dl), R, noise_pow, sinr_targets)
    assert np.allclose(margin_dl, 0)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_power_alloc_qos_zero_sinr_margin_uplink(num_zones, num_ls):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = get_random_noise_pow(num_zones)
    rng = np.random.default_rng(SEED+12)
    sinr_targets = rng.uniform(low=0.05, high=0.2, size=num_zones)

    #R, noise_pow = sinropt.normalize_system(R, noise_pow)

    p = sinropt.power_alloc_qos_uplink(w, R, noise_pow, sinr_targets)
    margin = sinropt.sinr_margin_uplink(sinropt.apply_power_vec(w, p), R, noise_pow, sinr_targets)
    sinr = sinropt._sinr_uplink_comparison(w, R, p, noise_pow)
    sinr2 = sinropt.sinr_uplink(sinropt.apply_power_vec(w, p), R, noise_pow)
    assert np.allclose(margin, 0)

    # p_dl = sinropt.power_alloc_qos_downlink(w, R, noise_pow, sinr_targets)
    # margin_dl = sinropt.sinr_margin_downlink(sinropt.apply_power_vec(w, p_dl), R, noise_pow, sinr_targets)
    
    # assert np.allclose(margin_dl, 0)

    


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
    st.floats(min_value=1, max_value=10000),
)
def test_equal_capacity_uplink_downlink_duality(num_zones, num_ls, max_pow):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    sinr_targets = get_random_sinr_targets(num_zones)

    q, c_ul = sinropt.power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, max_pow)
    p, c_dl = sinropt.power_alloc_minmax_downlink(w, R, noise_pow, sinr_targets, max_pow)
    assert np.allclose(c_ul, c_dl)