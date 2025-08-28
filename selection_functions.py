"""
Selection Functions for Tully-Fisher Analysis

This module handles magnitude selection effects in galaxy surveys, including:
- Selection function calculations (simple cuts and complex functions)
- Bias estimation and correction
- Visualization tools
- Survey-specific presets

Author: Paula Boubel
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import warnings

def calculate_selection_function(m, selection_params):
    """
    Calculate selection function F(m) for given magnitudes.
    
    Parameters:
    -----------
    m : array_like
        Apparent magnitudes
    selection_params : list
        Either [m_lim] for simple cut or [m1, m2, a] for complex function
        
    Returns:
    --------
    np.array : Selection function values F(m)
    """
    m = np.asarray(m)
    
    if len(selection_params) == 1:
        # Simple magnitude cut
        m_lim = selection_params[0]
        F_m = np.ones_like(m)
        F_m[m > m_lim] = 0
        
    elif len(selection_params) == 3:
        # Complex selection function
        m1, m2, a = selection_params
        F_m = np.ones_like(m)
        mask = m >= m1
        
        if np.any(mask):
            log_F = (a * (m[mask] - m2)**2 - 
                    a * (m1 - m2)**2 - 
                    0.6 * (m[mask] - m1))
            F_m[mask] += (10**log_F - 1)
    else:
        raise ValueError("Selection parameters must be [m_lim] or [m1, m2, a]")
    
    return F_m

def simple_selection_integral(m_pred, sigma, m_lim):
    """
    Integral of selection function for simple magnitude cut.
    
    Uses complementary error function for efficiency.
    """
    return 0.5 * (1 + erf((m_lim - m_pred) / (np.sqrt(2) * sigma)))

def complex_selection_integral(m_pred, sigma, m1, m2, a):
    """
    Analytic approximation for complex selection integral.
    
    This implements the analytic solution for the normalization integral.
    """
    # Transform parameters for analytic solution
    a_transformed = -a
    
    # Calculate integral components
    prefactor = sigma / np.sqrt(4.60517 * a_transformed * sigma**2 + 1)
    prefactor *= 10**(m1 * (a_transformed * (m1 - 2*m2) + 0.6))
    
    # Exponential term
    exp_term = (sigma**2 * (2.30259 * a_transformed**2 * m2**2 - 
                           1.38155 * a_transformed * m2 + 0.207233) +
               m_pred * (a_transformed * m2 - 0.3) - 
               0.5 * a_transformed * m_pred**2)
    exp_term /= (a_transformed * sigma**2 + 0.217147)
    
    prefactor *= np.exp(exp_term)
    
    # Error function argument
    erf_arg = (sigma**2 * (3.25635 * a_transformed * m1 - 
                          3.25635 * a_transformed * m2 + 0.976904) +
              0.707107 * m1 - 0.707107 * m_pred)
    erf_arg /= (sigma * np.sqrt(4.60517 * a_transformed * sigma**2 + 1))
    
    # Final result
    result = 1.25331 * prefactor * (1 - erf(erf_arg))
    
    return result

def selection_integral(m_pred, sigma, selection_params):
    """
    Calculate selection function normalization integral.
    
    Parameters:
    -----------
    m_pred : array_like
        Predicted magnitudes
    sigma : array_like
        Magnitude uncertainties
    selection_params : list
        Selection function parameters
        
    Returns:
    --------
    np.array : Normalization integrals
    """
    if len(selection_params) == 1:
        # Simple magnitude cut
        m_lim = selection_params[0]
        return simple_selection_integral(m_pred, sigma, m_lim)
        
    elif len(selection_params) == 3:
        # Complex selection function
        m1, m2, a = selection_params
        
        # Part 1: Integral from -inf to m1
        integral_1 = simple_selection_integral(m_pred, sigma, m1)
        
        # Part 2: Integral from m1 to +inf (complex part)
        integral_2 = complex_selection_integral(m_pred, sigma, m1, m2, a)
        
        return integral_1 + integral_2
    else:
        raise ValueError("Selection parameters must be [m_lim] or [m1, m2, a]")

def apply_selection_correction(m_obs, m_pred, sigma, selection_params):
    """
    Apply selection function correction to likelihood.
    
    Parameters:
    -----------
    m_obs : array_like
        Observed magnitudes
    m_pred : array_like
        Predicted magnitudes
    sigma : array_like
        Magnitude uncertainties
    selection_params : list
        Selection function parameters
        
    Returns:
    --------
    np.array : Selection-corrected log-likelihood terms
    """
    # Selection function for observed magnitudes
    F_m = calculate_selection_function(m_obs, selection_params)
    
    # Normalization integral
    integral = selection_integral(m_pred, sigma, selection_params)
    
    # Selection-corrected likelihood terms
    return np.log(F_m) - np.log(integral)

def estimate_selection_bias(log_w, m_app, selection_params=None, n_bootstrap=100):
    """
    Estimate potential selection bias in TF parameters.
    
    Parameters:
    -----------
    log_w : array_like
        Log velocity widths (offset by 2.5)
    m_app : array_like
        Apparent magnitudes
    selection_params : list, optional
        Selection function parameters
    n_bootstrap : int
        Number of bootstrap samples for uncertainty estimation
        
    Returns:
    --------
    dict : Bias estimates and uncertainties
    """
    log_w = np.asarray(log_w)
    m_app = np.asarray(m_app)
    
    # Fit simple linear relation to full sample
    A = np.vstack([log_w, np.ones(len(log_w))]).T
    params_all, _, _, _ = np.linalg.lstsq(A, m_app, rcond=None)
    a1_all, a0_all = params_all
    
    results = {
        'full_sample': {'a0': a0_all, 'a1': a1_all, 'n_galaxies': len(m_app)},
        'selection_bias': None
    }
    
    if selection_params is not None:
        # Apply selection and refit
        F_m = calculate_selection_function(m_app, selection_params)
        
        if len(selection_params) == 1:
            # Simple magnitude cut
            selected = m_app <= selection_params[0]
        else:
            # Complex selection - use probabilistic selection
            selected = np.random.random(len(m_app)) < F_m
        
        if np.sum(selected) > 10:  # Need minimum galaxies
            A_sel = np.vstack([log_w[selected], np.ones(np.sum(selected))]).T
            params_sel, _, _, _ = np.linalg.lstsq(A_sel, m_app[selected], rcond=None)
            a1_sel, a0_sel = params_sel
            
            # Calculate bias
            bias_a0 = a0_sel - a0_all
            bias_a1 = a1_sel - a1_all
            
            results['selected_sample'] = {
                'a0': a0_sel, 'a1': a1_sel, 
                'n_galaxies': np.sum(selected),
                'selection_fraction': np.sum(selected) / len(m_app)
            }
            
            results['selection_bias'] = {
                'bias_a0': bias_a0,
                'bias_a1': bias_a1,
                'magnitude_range': f"{m_app.min():.1f} - {m_app.max():.1f}",
                'selected_range': f"{m_app[selected].min():.1f} - {m_app[selected].max():.1f}"
            }
            
            # Bootstrap uncertainty estimation
            if n_bootstrap > 0:
                bias_a0_boot = []
                bias_a1_boot = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap resample
                    boot_idx = np.random.choice(len(m_app), len(m_app), replace=True)
                    m_boot = m_app[boot_idx]
                    log_w_boot = log_w[boot_idx]
                    
                    try:
                        # Fit full bootstrap sample
                        A_boot = np.vstack([log_w_boot, np.ones(len(log_w_boot))]).T
                        params_boot_all, _, _, _ = np.linalg.lstsq(A_boot, m_boot, rcond=None)
                        a1_boot_all, a0_boot_all = params_boot_all
                        
                        # Apply selection to bootstrap sample
                        F_m_boot = calculate_selection_function(m_boot, selection_params)
                        if len(selection_params) == 1:
                            selected_boot = m_boot <= selection_params[0]
                        else:
                            selected_boot = np.random.random(len(m_boot)) < F_m_boot
                        
                        if np.sum(selected_boot) > 10:
                            A_boot_sel = np.vstack([log_w_boot[selected_boot], 
                                                  np.ones(np.sum(selected_boot))]).T
                            params_boot_sel, _, _, _ = np.linalg.lstsq(A_boot_sel, 
                                                                     m_boot[selected_boot], rcond=None)
                            a1_boot_sel, a0_boot_sel = params_boot_sel
                            
                            bias_a0_boot.append(a0_boot_sel - a0_boot_all)
                            bias_a1_boot.append(a1_boot_sel - a1_boot_all)
                    except:
                        continue
                
                if len(bias_a0_boot) > 10:
                    results['selection_bias']['bias_a0_std'] = np.std(bias_a0_boot)
                    results['selection_bias']['bias_a1_std'] = np.std(bias_a1_boot)
    
    return results

def print_selection_bias_report(bias_results):
    """Print a formatted report of selection bias analysis."""
    print("Selection Bias Analysis Report")
    print("=" * 40)
    
    full = bias_results['full_sample']
    print(f"Full sample ({full['n_galaxies']} galaxies):")
    print(f"  a0 = {full['a0']:.3f}")
    print(f"  a1 = {full['a1']:.3f}")
    
    if bias_results['selection_bias'] is not None:
        selected = bias_results['selected_sample']
        bias = bias_results['selection_bias']
        
        print(f"\nSelected sample ({selected['n_galaxies']} galaxies, "
              f"{selected['selection_fraction']:.1%} of full sample):")
        print(f"  a0 = {selected['a0']:.3f}")
        print(f"  a1 = {selected['a1']:.3f}")
        
        print(f"\nSelection bias:")
        print(f"  Δa0 = {bias['bias_a0']:+.3f}", end="")
        if 'bias_a0_std' in bias:
            print(f" ± {bias['bias_a0_std']:.3f}")
        else:
            print()
        
        print(f"  Δa1 = {bias['bias_a1']:+.3f}", end="")
        if 'bias_a1_std' in bias:
            print(f" ± {bias['bias_a1_std']:.3f}")
        else:
            print()
        
        print(f"\nMagnitude ranges:")
        print(f"  Full sample: {bias['magnitude_range']}")
        print(f"  Selected:    {bias['selected_range']}")
        
        # Interpretation
        if abs(bias['bias_a0']) > 0.1:
            print(f"\n⚠ WARNING: Large zero-point bias detected!")
        if abs(bias['bias_a1']) > 0.5:
            print(f"⚠ WARNING: Large slope bias detected!")
            
        print(f"\n💡 TIP: Use selection function correction in your MCMC analysis")
    else:
        print("\nNo selection function specified - bias not estimated")

def plot_selection_function(selection_params, m_range=(12, 17), ax=None):
    """
    Plot the selection function.
    
    Parameters:
    -----------
    selection_params : list
        Either [m_lim] for simple cut or [m1, m2, a] for complex function
    m_range : tuple
        Magnitude range to plot
    ax : matplotlib axis, optional
        Axis to plot on
        
    Returns:
    --------
    matplotlib axis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required for plotting")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    m = np.linspace(m_range[0], m_range[1], 1000)
    F_m = calculate_selection_function(m, selection_params)
    
    if len(selection_params) == 1:
        m_lim = selection_params[0]
        ax.plot(m, F_m, 'b-', linewidth=2, label=f'Simple cut (m < {m_lim})')
        ax.axvline(m_lim, color='red', linestyle='--', alpha=0.7)
        
    elif len(selection_params) == 3:
        m1, m2, a = selection_params
        ax.plot(m, F_m, 'g-', linewidth=2, 
               label=f'Complex (m1={m1}, m2={m2}, a={a})')
        ax.axvline(m1, color='orange', linestyle='--', alpha=0.7, label=f'm1={m1}')
        ax.axvline(m2, color='red', linestyle='--', alpha=0.7, label=f'm2={m2}')
    
    ax.set_xlabel('Apparent Magnitude')
    ax.set_ylabel('Selection Function F(m)')
    ax.set_title('Survey Selection Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.2)
    
    return ax

def plot_bias_comparison(log_w, m_app, selection_list, ax=None):
    """
    Plot comparison of selection bias for different selection functions.
    
    Parameters:
    -----------
    log_w : array_like
        Log velocity widths
    m_app : array_like
        Apparent magnitudes
    selection_list : list
        List of (name, selection_params) tuples
    ax : matplotlib axis, optional
        Axis to plot on
        
    Returns:
    --------
    matplotlib axis, dict : Plot axis and bias results
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required for plotting")
        return None, {}
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    bias_results = {}
    names = []
    a0_biases = []
    a1_biases = []
    
    for name, selection_params in selection_list:
        bias_result = estimate_selection_bias(log_w, m_app, selection_params, n_bootstrap=0)
        bias_results[name] = bias_result
        
        if bias_result['selection_bias'] is not None:
            names.append(name)
            a0_biases.append(bias_result['selection_bias']['bias_a0'])
            a1_biases.append(bias_result['selection_bias']['bias_a1'])
    
    if names:
        x_pos = np.arange(len(names))
        width = 0.35
        
        ax.bar(x_pos - width/2, a0_biases, width, label='a0 bias', alpha=0.7)
        ax.bar(x_pos + width/2, a1_biases, width, label='a1 bias', alpha=0.7)
        
        ax.set_xlabel('Selection Function')
        ax.set_ylabel('Parameter Bias')
        ax.set_title('Selection Function Bias Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    return ax, bias_results

# Survey-specific selection function presets
SURVEY_SELECTION_FUNCTIONS = {
    # Simple magnitude cuts
    'bright_cut_14': [14.0],
    'bright_cut_15': [15.0], 
    'bright_cut_16': [16.0],
    
    # SDSS-like selection (r-band)
    'sdss_r': [13.0, 13.8, -0.11],
    'sdss_i': [12.8, 13.5, -0.10],
    
    # WISE selection (W1 band) 
    'wise_w1': [12.0, 13.5, -0.10],
    'wise_w2': [11.8, 13.2, -0.09],
    
    # 6dF Galaxy Survey
    '6df_k': [12.5, 13.8, -0.12],
    
    # Custom examples
    'shallow_survey': [14.5, 15.5, -0.15],
    'deep_survey': [15.0, 16.5, -0.08],
}

def get_survey_selection(survey_name):
    """
    Get selection function parameters for a specific survey.
    
    Parameters:
    -----------
    survey_name : str
        Survey name from SURVEY_SELECTION_FUNCTIONS
        
    Returns:
    --------
    list : Selection parameters
    """
    if survey_name not in SURVEY_SELECTION_FUNCTIONS:
        available = list(SURVEY_SELECTION_FUNCTIONS.keys())
        raise ValueError(f"Unknown survey '{survey_name}'. Available: {available}")
    
    return SURVEY_SELECTION_FUNCTIONS[survey_name]

def validate_selection_parameters(selection_params):
    """
    Validate selection function parameters.
    
    Parameters:
    -----------
    selection_params : list
        Selection parameters to validate
        
    Returns:
    --------
    bool : True if valid
    list : List of warnings/errors
    """
    issues = []
    
    if len(selection_params) == 1:
        # Simple magnitude cut
        m_lim = selection_params[0]
        if not (10.0 <= m_lim <= 20.0):
            issues.append(f"Magnitude limit {m_lim} outside reasonable range [10, 20]")
            
    elif len(selection_params) == 3:
        # Complex selection function
        m1, m2, a = selection_params
        
        if not (10.0 <= m1 <= 18.0):
            issues.append(f"m1 = {m1} outside reasonable range [10, 18]")
            
        if not (10.0 <= m2 <= 20.0):
            issues.append(f"m2 = {m2} outside reasonable range [10, 20]")
            
        if not (-1.0 <= a <= 0.0):
            issues.append(f"a = {a} outside reasonable range [-1, 0] (should be negative)")
            
        if m2 < m1:
            issues.append(f"m2 ({m2}) should be >= m1 ({m1})")
            
    else:
        issues.append(f"Selection parameters must be [m_lim] or [m1, m2, a], got {len(selection_params)} parameters")
    
    return len(issues) == 0, issues
