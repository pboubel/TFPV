"""

This library provides tools for analyzing galaxy peculiar velocities and 
Tully-Fisher relations using MCMC fitting with emcee.

Author: Paula Boubel

"""

import numpy as np
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import math
import warnings

from utils import sgxyz, VEXT_CARRICK, BETA_CARRICK, C_LIGHT
from selection_functions import apply_selection_correction
from pvhub import pvhub
PVHUB_AVAILABLE = True

# Constants and Configuration
DEFAULT_SIGMA_V = 150  # km/s velocity field scatter
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=100. * u.km / u.s / u.Mpc, Om0=0.315)

class TullyFisherAnalysis:
    """
    Main class for Tully-Fisher and peculiar velocity analysis.
    """
    
    def __init__(self, tf_model='linear', scatter_model='fixed', 
                 include_pv=False, 
                 sigma_v=DEFAULT_SIGMA_V, cosmology=None):
        """
        Initialize the analysis.
        
        Parameters:
        -----------
        tf_model : str
            Tully-Fisher model type: 'linear', 'curved', or 'parabolic'
        scatter_model : str
            Scatter model: 'fixed' or 'linear'
        include_pv : bool
            Whether to include peculiar velocity modeling
        sigma_v : float
            Velocity field scatter in km/s
        cosmology : astropy.cosmology object
            Cosmological model to use
        """
        self.tf_model = tf_model
        self.scatter_model = scatter_model
        self.include_pv = include_pv
        self.sigma_v = sigma_v
        self.cosmo = cosmology or DEFAULT_COSMOLOGY
        
        # Validate inputs
        if tf_model not in ['linear', 'curved', 'parabolic']:
            raise ValueError("tf_model must be 'linear', 'curved', or 'parabolic'")
        if scatter_model not in ['fixed', 'linear']:
            raise ValueError("scatter_model must be 'fixed' or 'linear'")
    
    def prepare_data(self, ra, dec, z, log_w, m, xerr, yerr, pv_field=None):
        """
        Prepare data for analysis.
        
        Parameters:
        -----------
        ra, dec : array_like
            Right ascension and declination in degrees
        z : array_like
            Redshift
        log_w : array_like
            Log of integrated velocity width
        m : array_like
            Apparent magnitude
        xerr, yerr : array_like
            Errors in log_w and magnitude
        pv_field : array_like, optional
            Peculiar velocity field values
            
        Returns:
        --------
        dict : Prepared data dictionary
        """
        data = {
            'ra': np.array(ra),
            'dec': np.array(dec),
            'z': np.array(z),
            'log_w': np.array(log_w),
            'm': np.array(m),
            'xerr': np.array(xerr),
            'yerr': np.array(yerr)
        }
        
        # Convert coordinates to supergalactic cartesian
        data['du'] = self._ra_dec_to_supergalactic_cartesian(ra, dec)
        
        # Calculate luminosity distance
        data['dL'] = self.cosmo.luminosity_distance(z).value
            
        # Handle peculiar velocity field
        if self.include_pv:
            if not PVHUB_AVAILABLE:
                raise ImportError(
                    "pvhub is required for peculiar velocity modeling. "
                    "Install with: pip install git+https://github.com/KSaid-1/pvhub.git"
                )
            if pv_field is None:
                raise ValueError("pv_field required when include_pv=True")
            # Normalize PV field using Carrick parameters
            vext_r_carrick = np.dot(data['du'], VEXT_CARRICK)
            data['pv_norm'] = (pv_field + vext_r_carrick) / BETA_CARRICK
        else:
            data['pv_norm'] = None
            
        return data
    
    def get_initial_parameters(self):
        """
        Get reasonable initial parameter values for MCMC.
        
        Returns:
        --------
        np.array : Initial parameter values
        """
        if self.tf_model == 'linear':
            if self.scatter_model == 'fixed':
                params = [-21.0, -8.0, 0.3]  # [a0, a1, epsilon_0]
            else:
                params = [-21.0, -8.0, 0.3, 0.1]  # [a0, a1, epsilon_0, epsilon_1]
        elif self.tf_model == 'curved':
            if self.scatter_model == 'fixed':
                params = [-21.0, -8.0, -1.0, 0.3]  # [a0, a1, a2, epsilon_0]
            else:
                params = [-21.0, -8.0, -1.0, 0.3, 0.1]  # [a0, a1, a2, epsilon_0, epsilon_1]
        elif self.tf_model == 'parabolic':
            if self.scatter_model == 'fixed':
                params = [-21.0, -8.0, -0.5, 0.3]  # [a0, a1, a2, epsilon_0]
            else:
                params = [-21.0, -8.0, -0.5, 0.3, 0.1]  # [a0, a1, a2, epsilon_0, epsilon_1]
        
        # Add PV parameters if needed
        if self.include_pv:
            params.extend([BETA_CARRICK, 0.0, 0.0, 0.0])  # [beta, vext_x, vext_y, vext_z]
            
        return np.array(params)
    
    def log_probability(self, theta, data, selection=None):
        """
        Calculate log probability for MCMC.
        
        Parameters:
        -----------
        theta : array_like
            Parameter values
        data : dict
            Data dictionary from prepare_data()
        selection : list or None
            Selection function parameters
            
        Returns:
        --------
        float : Log probability
        """
        # Check priors
        lp = self._log_prior(theta, data['log_w'])
        if not np.isfinite(lp):
            return -np.inf
            
        # Calculate likelihood
        if self.include_pv:
            ll = self._log_likelihood_combined(theta, data, selection)
        else:
            ll = self._log_likelihood_tf_only(theta, data, selection)
            
        return lp + ll
    
    def _ra_dec_to_supergalactic_cartesian(self, ra, dec):
        """Convert RA/Dec to supergalactic cartesian coordinates."""
        return sgxyz(ra, dec)
    
    def _tully_fisher_relation(self, log_w_offset, tf_params):
        """Calculate Tully-Fisher relation."""
        if self.tf_model == 'linear':
            a0, a1 = tf_params[:2]
            return a1 * log_w_offset + a0
        elif self.tf_model == 'curved':
            a0, a1, a2 = tf_params[:3]
            linear = a1 * log_w_offset + a0
            curvature = a2 * log_w_offset**2
            curvature_mask = log_w_offset > 0
            return linear + curvature * curvature_mask
        elif self.tf_model == 'parabolic':
            a0, a1, a2 = tf_params[:3]
            return a0 + a1 * log_w_offset + a2 * log_w_offset**2
    
    def _get_scatter(self, log_w, scatter_params):
        """Calculate scatter as function of log_w."""
        if self.scatter_model == 'fixed':
            return np.full_like(log_w, scatter_params[0])
        else:
            epsilon_0, epsilon_1 = scatter_params
            return epsilon_1 * (log_w - 2.5) + epsilon_0
    
    def _log_prior(self, theta, log_w):
        """Calculate log prior."""
        log_w_min, log_w_max = log_w.min(), log_w.max()
        
        # Extract parameters based on model
        if self.tf_model == 'linear':
            if self.scatter_model == 'fixed':
                a0, a1, epsilon_0 = theta[:3]
                scatter_min = scatter_max = epsilon_0
                idx = 3
            else:
                a0, a1, epsilon_0, epsilon_1 = theta[:4]
                scatter_min = epsilon_1 * (log_w_min - 2.5) + epsilon_0
                scatter_max = epsilon_1 * (log_w_max - 2.5) + epsilon_0
                idx = 4
        else:  # curved or parabolic
            if self.scatter_model == 'fixed':
                a0, a1, a2, epsilon_0 = theta[:4]
                scatter_min = scatter_max = epsilon_0
                idx = 4
            else:
                a0, a1, a2, epsilon_0, epsilon_1 = theta[:5]
                scatter_min = epsilon_1 * (log_w_min - 2.5) + epsilon_0
                scatter_max = epsilon_1 * (log_w_max - 2.5) + epsilon_0
                idx = 5
        
        # Check TF parameter bounds
        if not (-50.0 < a1 < 50.0 and -50.0 < a0 < 50.0 and 
                scatter_min >= 0 and scatter_max >= 0):
            return -np.inf
            
        # Check PV parameter bounds if needed
        if self.include_pv:
            beta = theta[idx]
            if beta <= 0:
                return -np.inf
                
        return 0.0
    
    def _log_likelihood_tf_only(self, theta, data, selection):
        """Calculate likelihood for TF-only model."""
        # Extract parameters
        if self.tf_model == 'linear':
            if self.scatter_model == 'fixed':
                a0, a1, epsilon_0 = theta
                tf_params = [a0, a1]
                scatter_params = [epsilon_0]
            else:
                a0, a1, epsilon_0, epsilon_1 = theta
                tf_params = [a0, a1]
                scatter_params = [epsilon_0, epsilon_1]
        else:  # curved or parabolic
            if self.scatter_model == 'fixed':
                a0, a1, a2, epsilon_0 = theta
                tf_params = [a0, a1, a2]
                scatter_params = [epsilon_0]
            else:
                a0, a1, a2, epsilon_0, epsilon_1 = theta
                tf_params = [a0, a1, a2]
                scatter_params = [epsilon_0, epsilon_1]
        
        # Calculate model
        log_w_offset = data['log_w'] - 2.5
        M_pred = self._tully_fisher_relation(log_w_offset, tf_params)
        m_pred = M_pred + 5 * np.log10(data['dL'] * 1e5)
        
        # Calculate scatter
        sigma_tf = self._get_scatter(data['log_w'], scatter_params)
        
        # Total variance
        if self.tf_model == 'curved':
            # Different variance for linear vs curved parts
            sigma2_linear = sigma_tf**2 + data['yerr']**2 + (data['xerr'] * tf_params[1])**2
            sigma2_curved = (sigma_tf**2 + data['yerr']**2 + 
                           (data['xerr']**2) * (tf_params[1] + 2 * tf_params[2] * log_w_offset)**2)
            sigma2 = np.where(log_w_offset <= 0, sigma2_linear, sigma2_curved)
        elif self.tf_model == 'parabolic':
            sigma2 = (sigma_tf**2 + data['yerr']**2 + 
                     (data['xerr']**2) * (tf_params[1] + 2 * tf_params[2] * log_w_offset)**2)
        else:  # linear
            sigma2 = sigma_tf**2 + data['yerr']**2 + (data['xerr'] * tf_params[1])**2
        
        # Calculate likelihood
        diff = (data['m'] - m_pred)**2
        
        # Handle selection function
        if selection is not None:
            try:
                from selection_functions import apply_selection_correction
            except ImportError:
                import sys, os
                sys.path.append(os.path.dirname(__file__))
                from selection_functions import apply_selection_correction
            selection_correction = apply_selection_correction(
                data['m'], m_pred, np.sqrt(sigma2), selection
            )
            log_like = -0.5 * np.sum(diff / sigma2) + np.sum(selection_correction)
        else:
            log_like = -0.5 * np.sum(diff / sigma2 + np.log(2 * np.pi * sigma2))
        
        return log_like
    
    def _log_likelihood_combined(self, theta, data, selection):
        """Calculate likelihood for combined PV+TF model."""
        # Extract TF parameters (PV parameters are at the end)
        pv_params = theta[-4:]  # [beta, vext_x, vext_y, vext_z]
        tf_theta = theta[:-4]   # TF parameters
        
        # Extract TF parameters
        if self.tf_model == 'linear':
            if self.scatter_model == 'fixed':
                a0, a1, epsilon_0 = tf_theta
                tf_params = [a0, a1]
                scatter_params = [epsilon_0]
            else:
                a0, a1, epsilon_0, epsilon_1 = tf_theta
                tf_params = [a0, a1]
                scatter_params = [epsilon_0, epsilon_1]
        else:  # curved or parabolic
            if self.scatter_model == 'fixed':
                a0, a1, a2, epsilon_0 = tf_theta
                tf_params = [a0, a1, a2]
                scatter_params = [epsilon_0]
            else:
                a0, a1, a2, epsilon_0, epsilon_1 = tf_theta
                tf_params = [a0, a1, a2]
                scatter_params = [epsilon_0, epsilon_1]
        
        # Calculate peculiar velocity corrections
        beta, vext_x, vext_y, vext_z = pv_params
        vext_r = np.dot(data['du'], [vext_x, vext_y, vext_z])
        u = beta * data['pv_norm'] + vext_r
        zc = (data['z'] - u/C_LIGHT) / (1 + u/C_LIGHT)
        
        # Calculate distance to corrected redshift
        Dc = self.cosmo.comoving_distance(zc).value
        
        # Calculate model magnitude
        log_w_offset = data['log_w'] - 2.5
        M_pred = self._tully_fisher_relation(log_w_offset, tf_params)
        m_pred = M_pred + 25 + 5*np.log10(1 + data['z']) + 5*np.log10(Dc)
        
        # Calculate scatter
        sigma_tf = self._get_scatter(data['log_w'], scatter_params)
        
        # Total variance (TF part only, PV has separate likelihood)
        if self.tf_model == 'curved':
            sigma2_linear = sigma_tf**2 + data['yerr']**2 + (data['xerr'] * tf_params[1])**2
            sigma2_curved = (sigma_tf**2 + data['yerr']**2 + 
                           (data['xerr']**2) * (tf_params[1] + 2 * tf_params[2] * log_w_offset)**2)
            sigma2 = np.where(log_w_offset <= 0, sigma2_linear, sigma2_curved)
        elif self.tf_model == 'parabolic':
            sigma2 = (sigma_tf**2 + data['yerr']**2 + 
                     (data['xerr']**2) * (tf_params[1] + 2 * tf_params[2] * log_w_offset)**2)
        else:  # linear
            sigma2 = sigma_tf**2 + data['yerr']**2 + (data['xerr'] * tf_params[1])**2
        
        # TF likelihood
        diff_tf = (data['m'] - m_pred)**2
        
        if selection is not None:
            try:
                from selection_functions import apply_selection_correction
            except ImportError:
                import sys, os
                sys.path.append(os.path.dirname(__file__))
                from selection_functions import apply_selection_correction
            selection_correction = apply_selection_correction(
                data['m'], m_pred, np.sqrt(sigma2), selection
            )
            log_like_tf = -0.5 * np.sum(diff_tf / sigma2) + np.sum(selection_correction)
        else:
            log_like_tf = -0.5 * np.sum(diff_tf / sigma2 + np.log(2 * np.pi * sigma2))
        
        # PV likelihood
        model_zc = (C_LIGHT * data['z'] - u) / (1 + u/C_LIGHT)
        diff_pv = (zc * C_LIGHT - model_zc)**2
        sigma2_pv = self.sigma_v**2
        log_like_pv = -0.5 * np.sum(diff_pv / sigma2_pv + np.log(2 * np.pi * sigma2_pv))
        
        return log_like_tf + log_like_pv

# Example usage
if __name__ == "__main__":
    # Example of how to use the library
    
    # Create analysis object
    analysis = TullyFisherAnalysis(
        tf_model='linear',
        scatter_model='fixed',
        include_pv=False
    )
    
    # Prepare some example data (replace with your actual data)
    n_gal = 100
    ra = np.random.uniform(0, 360, n_gal)
    dec = np.random.uniform(-90, 90, n_gal)
    z = np.random.uniform(0.01, 0.05, n_gal)
    log_w = np.random.normal(2.5, 0.3, n_gal)
    m = np.random.normal(14, 1, n_gal)
    xerr = np.random.uniform(0.05, 0.1, n_gal)
    yerr = np.random.uniform(0.1, 0.2, n_gal)
    
    # Prepare data
    data = analysis.prepare_data(ra, dec, z, log_w, m, xerr, yerr)
    
    # Run MCMC (uncomment to run)
    # from utils import run_mcmc_analysis
    # results = run_mcmc_analysis(analysis, data, n_steps=1000)
    # print("MCMC completed!")
    # print(f"Mean acceptance fraction: {results['mean_acceptance']:.3f}")
