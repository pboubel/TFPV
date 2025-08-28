"""
Configuration file for Tully-Fisher analysis.

This file contains default parameters and settings that can be easily
modified without changing the main code.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# =============================================================================
# COSMOLOGICAL PARAMETERS
# =============================================================================

# Default cosmology (Planck 2018)
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=67.4 * u.km / u.s / u.Mpc, Om0=0.315)

# Alternative cosmologies for comparison
ALTERNATIVE_COSMOLOGIES = {
    'planck2015': FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Om0=0.308),
    'wmap9': FlatLambdaCDM(H0=69.3 * u.km / u.s / u.Mpc, Om0=0.287),
    'riess2019': FlatLambdaCDM(H0=74.0 * u.km / u.s / u.Mpc, Om0=0.3),
}

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792.458  # Speed of light in km/s

# =============================================================================
# TULLY-FISHER RELATION PARAMETERS
# =============================================================================

# Default TF parameter ranges for priors
TF_PARAM_RANGES = {
    'a0': (-25.0, -15.0),      # Zero-point range
    'a1': (-12.0, -4.0),       # Slope range  
    'a2': (-3.0, 3.0),         # Curvature range
    'epsilon_0': (0.01, 1.0),  # Intrinsic scatter range
    'epsilon_1': (-0.5, 0.5),  # Scatter slope range
}

# Initial parameter guesses for different models
TF_INITIAL_PARAMS = {
    'linear_fixed': [-21.0, -8.0, 0.3],
    'linear_variable': [-21.0, -8.0, 0.3, 0.1],
    'curved_fixed': [-21.0, -8.0, -1.0, 0.3],
    'curved_variable': [-21.0, -8.0, -1.0, 0.3, 0.1],
    'parabolic_fixed': [-21.0, -8.0, -0.5, 0.3],
    'parabolic_variable': [-21.0, -8.0, -0.5, 0.3, 0.1],
}

# =============================================================================
# PECULIAR VELOCITY PARAMETERS
# =============================================================================

# Default velocity field scatter
DEFAULT_SIGMA_V = 150.0  # km/s

# Carrick et al. (2015) parameters
CARRICK_PARAMS = {
    'l_gal': 304.0,      # Galactic longitude (degrees)
    'b_gal': 6.0,        # Galactic latitude (degrees)  
    'v_mag': 159.0,      # Velocity magnitude (km/s)
    'beta': 0.43,        # Growth rate parameter
}

# PV parameter ranges
PV_PARAM_RANGES = {
    'beta': (0.1, 1.5),
    'vext_x': (-500.0, 500.0),  # km/s
    'vext_y': (-500.0, 500.0),  # km/s
    'vext_z': (-500.0, 500.0),  # km/s
}

# =============================================================================
# MCMC SETTINGS
# =============================================================================

# Default MCMC parameters
MCMC_DEFAULTS = {
    'n_walkers': 32,
    'n_steps': 5000,
    'burn_in': 1000,
    'thin': 1,
    'progress': True,
}

# Convergence criteria
CONVERGENCE_CRITERIA = {
    'min_acceptance': 0.15,
    'max_acceptance': 0.70,
    'target_acceptance': 0.3,
    'rhat_threshold': 1.1,      # Gelman-Rubin statistic
    'min_eff_samples': 400,     # Minimum effective samples
}

# =============================================================================
# DATA PROCESSING SETTINGS
# =============================================================================

# Quality cuts
DATA_CUTS = {
    'min_redshift': 0.005,
    'max_redshift': 0.07,
    'min_log_width': 1.8,
    'max_log_width': 3.2,
    'max_width_error': 0.3,
    'max_mag_error': 0.5,
    'snr_threshold': 3.0,
}

# Distance approximations
USE_APPROXIMATIONS = {
    'linear_hubble': False,     # Use linear Hubble law instead of cosmology
    'small_z_expansion': True,  # Use small-z expansion for nearby galaxies
}

# =============================================================================
# SELECTION FUNCTION PARAMETERS  
# =============================================================================

# Common survey selection functions
SELECTION_FUNCTIONS = {
    'sdss': {
        'm1': 13.0,    # Bright limit
        'm2': 13.8,    # Transition magnitude
        'a': -0.11,    # Selection parameter
    },
    'wise': {
        'm1': 12.0,
        'm2': 13.5, 
        'a': -0.10,
    },
    'custom': {
        'm1': 14.0,
        'm2': 15.0,
        'a': -0.15,
    }
}

# =============================================================================
# PLOTTING SETTINGS
# =============================================================================

# Default plot parameters
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 150,
    'font_size': 12,
    'line_width': 2,
    'marker_size': 4,
    'alpha': 0.7,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
}

# Corner plot settings
CORNER_SETTINGS = {
    'bins': 30,
    'smooth': True,
    'show_titles': True,
    'title_kwargs': {'fontsize': 12},
    'label_kwargs': {'fontsize': 14},
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# File naming conventions
OUTPUT_SETTINGS = {
    'results_dir': 'results/',
    'plots_dir': 'plots/',
    'data_dir': 'data/',
    'filename_format': '{survey}_{model}_{date}',
    'save_chains': True,
    'save_plots': True,
    'plot_format': 'png',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_params(tf_model, scatter_model, include_pv=False):
    """
    Get initial parameters for a given model configuration.
    
    Parameters:
    -----------
    tf_model : str
        TF model type ('linear', 'curved', 'parabolic')
    scatter_model : str  
        Scatter model ('fixed', 'linear')
    include_pv : bool
        Whether to include PV parameters
        
    Returns:
    --------
    list : Initial parameter values
    """
    model_key = f"{tf_model}_{scatter_model}"
    
    if model_key not in TF_INITIAL_PARAMS:
        raise ValueError(f"Unknown model configuration: {model_key}")
    
    params = TF_INITIAL_PARAMS[model_key].copy()
    
    if include_pv:
        # Add PV parameters: [beta, vext_x, vext_y, vext_z]
        params.extend([CARRICK_PARAMS['beta'], 0.0, 0.0, 0.0])
    
    return params

def get_param_names(tf_model, scatter_model, include_pv=False):
    """
    Get parameter names for a given model configuration.
    
    Returns:
    --------
    list : Parameter names
    """
    if tf_model == 'linear':
        names = ['a0', 'a1']
    else:  # curved or parabolic
        names = ['a0', 'a1', 'a2']
    
    if scatter_model == 'fixed':
        names.append('epsilon_0')
    else:  # linear
        names.extend(['epsilon_0', 'epsilon_1'])
    
    if include_pv:
        names.extend(['beta', 'vext_x', 'vext_y', 'vext_z'])
    
    return names

def validate_data(data_dict, apply_cuts=True):
    """
    Validate and optionally apply quality cuts to data.
    
    Parameters:
    -----------
    data_dict : dict
        Data dictionary with galaxy observations
    apply_cuts : bool
        Whether to apply quality cuts
        
    Returns:
    --------
    dict : Validated data (possibly with cuts applied)
    bool : Array of valid indices
    """
    valid = np.ones(len(data_dict['z']), dtype=bool)
    
    if apply_cuts:
        # Apply redshift cuts
        valid &= (data_dict['z'] >= DATA_CUTS['min_redshift'])
        valid &= (data_dict['z'] <= DATA_CUTS['max_redshift'])
        
        # Apply log width cuts
        valid &= (data_dict['log_w'] >= DATA_CUTS['min_log_width'])
        valid &= (data_dict['log_w'] <= DATA_CUTS['max_log_width'])
        
        # Apply error cuts
        valid &= (data_dict['xerr'] <= DATA_CUTS['max_width_error'])
        valid &= (data_dict['yerr'] <= DATA_CUTS['max_mag_error'])
        
        # Apply S/N cuts
        width_snr = data_dict['log_w'] / data_dict['xerr']
        mag_snr = 1.0 / data_dict['yerr']  # Simplified S/N
        valid &= (width_snr >= DATA_CUTS['snr_threshold'])
        valid &= (mag_snr >= DATA_CUTS['snr_threshold'])
    
    n_total = len(valid)
    n_valid = np.sum(valid)
    
    print(f"Data validation: {n_valid}/{n_total} galaxies pass quality cuts")
    
    return valid

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Predefined analysis configurations
ANALYSIS_CONFIGS = {
    'basic': {
        'tf_model': 'linear',
        'scatter_model': 'fixed', 
        'include_pv': False,
        'mcmc': {'n_walkers': 16, 'n_steps': 1000, 'burn_in': 200}
    },
    'standard': {
        'tf_model': 'curved',
        'scatter_model': 'linear',
        'include_pv': False,  
        'mcmc': {'n_walkers': 32, 'n_steps': 3000, 'burn_in': 500}
    },
    'full': {
        'tf_model': 'curved',
        'scatter_model': 'linear',
        'include_pv': True,
        'mcmc': {'n_walkers': 48, 'n_steps': 5000, 'burn_in': 1000}
    }
}

if __name__ == "__main__":
    # Test configuration functions
    print("Testing configuration functions...")
    
    for config_name, config in ANALYSIS_CONFIGS.items():
        params = get_model_params(config['tf_model'], config['scatter_model'], 
                                config['include_pv'])
        names = get_param_names(config['tf_model'], config['scatter_model'],
                              config['include_pv'])
        
        print(f"\n{config_name.upper()} Configuration:")
        print(f"  Model: {config['tf_model']} TF, {config['scatter_model']} scatter")
        print(f"  Include PV: {config['include_pv']}")
        print(f"  Parameters ({len(params)}): {names}")
        print(f"  Initial values: {params}")
    
    print("\nConfiguration loaded successfully!")
