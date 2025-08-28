"""
Utilities for Tully-Fisher Analysis

This module contains general utility functions including:
- MCMC analysis runner
- Coordinate transformations
- Peculiar velocity field calculations
- Data validation helpers
- Result analysis tools

Author: Paula Boubel
"""

import numpy as np
import warnings
import math
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

# Optional imports
try:
    from pvhub import pvhub
    PVHUB_AVAILABLE = True
except ImportError:
    PVHUB_AVAILABLE = False

# Constants
C_LIGHT = 299792.458  # km/s

# Carrick et al. peculiar velocity parameters
_carrick_coord = SkyCoord(304 * u.degree, 6 * u.degree, distance=159 * u.km / u.s, frame='galactic')
_carrick_sg = _carrick_coord.transform_to('supergalactic')
_carrick_sg.representation_type = "cartesian"
_carrick_coord.representation_type = "cartesian"

VEXT_CARRICK = [_carrick_sg.sgx.value, _carrick_sg.sgy.value, _carrick_sg.sgz.value]
BETA_CARRICK = 0.43

def sgxyz(ra, dec):
    """
    Convert RA/Dec to supergalactic cartesian coordinates.
    
    Parameters:
    -----------
    ra, dec : array_like
        Right ascension and declination in degrees
        
    Returns:
    --------
    np.array : Supergalactic cartesian coordinates [x, y, z]
    """
    coord_obj = SkyCoord(ra, dec, frame='icrs', unit='deg')
    sg_coord = coord_obj.transform_to('supergalactic')
    
    p = 180.0 / math.pi
    sgl = sg_coord.sgl.value
    sgb = sg_coord.sgb.value
    
    x = np.cos(sgb/p) * np.cos(sgl/p)
    y = np.cos(sgb/p) * np.sin(sgl/p)
    z = np.sin(sgb/p)
    
    return np.array([x, y, z]).T

def get_pv_field(ra, dec, z, model=3):
    """
    Convenience function to get peculiar velocity field from pvhub.
    
    Parameters:
    -----------
    ra, dec : array_like
        Right ascension and declination in degrees
    z : array_like  
        Redshift
    model : int
        pvhub model number:
        1 = Linear theory
        2 = 2nd order Lagrangian perturbation theory
        3 = 2M++ reconstruction (default, recommended)
        4 = Carrick et al. (2015) reconstruction
        
    Returns:
    --------
    np.array : Peculiar velocity field values in km/s
    
    Raises:
    -------
    ImportError : If pvhub is not installed
    """
    if not PVHUB_AVAILABLE:
        raise ImportError(
            "pvhub is required for PV field calculations. "
            "Install with: pip install git+https://github.com/KSaid-1/pvhub.git"
        )
    
    # Set the model in pvhub
    pvhub.choose_model(model)
    
    # Calculate PV field
    pv_field = pvhub.calculate_pv(ra, dec, z)
    
    return pv_field

def run_mcmc_analysis(analysis, data, selection=None, n_walkers=32, n_steps=5000, burn_in=500):
    """
    Run MCMC analysis using emcee.
    
    Parameters:
    -----------
    analysis : TullyFisherAnalysis
        Analysis object with log_probability method
    data : dict
        Data dictionary from prepare_data()
    selection : list, optional
        Selection function parameters:
        - [m_lim] for simple magnitude cut
        - [m1, m2, a] for complex selection function
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    burn_in : int
        Number of burn-in steps to discard
        
    Returns:
    --------
    dict : Results dictionary with samples and diagnostics
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee required for MCMC analysis. Install with: pip install emcee")
    
    # Get initial parameters
    initial_params = analysis.get_initial_parameters()
    n_dim = len(initial_params)
    
    # Initialize walkers around initial values
    pos = initial_params + 1e-4 * np.random.randn(n_walkers, n_dim)
    
    # Set up sampler with selection function
    def log_prob_wrapper(theta):
        return analysis.log_probability(theta, data, selection)
    
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_wrapper)
    
    # Run MCMC
    print(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
    if selection is not None:
        if len(selection) == 1:
            print(f"Selection: simple magnitude cut at m < {selection[0]}")
        else:
            print(f"Selection: complex function with parameters {selection}")
    
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Extract samples (after burn-in)
    samples = sampler.chain[:, burn_in:, :].reshape((-1, n_dim))
    
    # Calculate diagnostics
    acceptance_fraction = sampler.acceptance_fraction
    mean_acceptance = np.mean(acceptance_fraction)
    
    results = {
        'samples': samples,
        'chain': sampler.chain,
        'lnprob': sampler.lnprobability,
        'acceptance_fraction': acceptance_fraction,
        'mean_acceptance': mean_acceptance,
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'burn_in': burn_in,
        'selection': selection,
        'n_samples': len(samples),
        'effective_samples': len(samples) / n_walkers,  # Rough estimate
    }
    
    # Add warnings for poor convergence
    if mean_acceptance < 0.15:
        warnings.warn(f"Low acceptance rate ({mean_acceptance:.3f}). Consider adjusting step size or initial parameters.")
    elif mean_acceptance > 0.7:
        warnings.warn(f"High acceptance rate ({mean_acceptance:.3f}). Consider increasing step size.")
    
    print(f"MCMC completed. Mean acceptance rate: {mean_acceptance:.3f}")
    
    return results

def calculate_parameter_summary(samples, param_names=None, percentiles=[16, 50, 84]):
    """
    Calculate summary statistics for MCMC samples.
    
    Parameters:
    -----------
    samples : array_like
        MCMC samples with shape (n_samples, n_params)
    param_names : list, optional
        Parameter names
    percentiles : list
        Percentiles to calculate
        
    Returns:
    --------
    dict : Parameter summary statistics
    """
    samples = np.asarray(samples)
    n_params = samples.shape[1]
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    summary = {}
    
    for i, name in enumerate(param_names):
        param_samples = samples[:, i]
        
        # Basic statistics
        mean_val = np.mean(param_samples)
        std_val = np.std(param_samples)
        median_val = np.median(param_samples)
        
        # Percentiles
        percs = np.percentile(param_samples, percentiles)
