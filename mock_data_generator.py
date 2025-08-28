"""
Mock Galaxy Data Generator for Tully-Fisher Analysis

This module generates realistic mock galaxy samples for testing Tully-Fisher 
relation analysis, including:
- Realistic coordinate distributions
- Peculiar velocity effects
- Observational biases (inclination, measurement errors)
- Selection effects
- Multiple photometric bands

Dependencies:
- numpy, scipy, pandas, astropy
- pvhub (for peculiar velocity fields)
- helio_cmb (local module for coordinate corrections)
- h5py, fits files for real data templates

Author: Paula Boubel
"""

import numpy as np
from scipy import linalg
from scipy.stats import uniform, multivariate_normal, reciprocal, norm, gaussian_kde
from scipy.optimize import minimize, curve_fit
import pandas as pd
import os
import h5py
import math
import warnings

# Astronomy packages
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM, Planck13, z_at_value
from astropy.modeling.models import Linear1D, Exponential1D
from astropy.io import fits
from astropy.table import Table

# Local imports
try:
    from pvhub import pvhub
    PVHUB_AVAILABLE = True
except ImportError:
    PVHUB_AVAILABLE = False
    warnings.warn("pvhub not available - PV field calculations disabled")

try:
    from helio_cmb import perform_corr
    HELIO_CMB_AVAILABLE = True
except ImportError:
    HELIO_CMB_AVAILABLE = False
    warnings.warn("helio_cmb not available - coordinate corrections disabled")

# Constants and Default Parameters
C_LIGHT = 299792.458  # km/s
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=100. * u.km / u.s / u.Mpc, Om0=0.315)

# Default Tully-Fisher parameters
DEFAULT_TF_PARAMS = {
    'a0': -20.0,    # Zero-point
    'a1': -8.0,     # Slope  
    'a2': 6.0,      # Curvature
    'd': 0.0,       # Curvature break point
}

# Default peculiar velocity parameters (Carrick et al.)
DEFAULT_PV_PARAMS = {
    'l_gal': 304.0,  # degrees
    'b_gal': 6.0,    # degrees
    'v_mag': 159.0,  # km/s
    'beta': 0.43,
}

# Schechter function parameters for HI mass function
SCHECHTER_PARAMS = {
    'phi_star': 10**(-2.23),
    'log_m_star': 9.94,
    'alpha': -1.25,
}

def _setup_pv_coordinates():
    """Setup Carrick et al. peculiar velocity coordinate system."""
    if not PVHUB_AVAILABLE:
        return None, None
    
    # Convert Carrick et al. dipole to supergalactic coordinates
    carrick_coord = SkyCoord(
        DEFAULT_PV_PARAMS['l_gal'] * u.degree, 
        DEFAULT_PV_PARAMS['b_gal'] * u.degree, 
        distance=DEFAULT_PV_PARAMS['v_mag'] * u.km / u.s, 
        frame='galactic'
    )
    
    sg_coord = carrick_coord.transform_to('supergalactic')
    sg_coord.representation_type = "cartesian"
    carrick_coord.representation_type = "cartesian"
    
    vext_sgxyz = [sg_coord.sgx.value, sg_coord.sgy.value, sg_coord.sgz.value]
    vext_gxyz = [carrick_coord.u.value, carrick_coord.v.value, carrick_coord.w.value]
    
    return vext_sgxyz, vext_gxyz

# Setup coordinate systems
VEXT_SGXYZ, VEXT_GXYZ = _setup_pv_coordinates()

def sgxyz(ra, dec):
    """Convert RA/Dec to supergalactic cartesian coordinates."""
    coord_obj = SkyCoord(ra, dec, frame='icrs', unit='deg')
    sg_coord = coord_obj.transform_to('supergalactic')
    
    p = 180.0 / math.pi
    sgl = sg_coord.sgl.value
    sgb = sg_coord.sgb.value
    
    x = np.cos(sgb/p) * np.cos(sgl/p)
    y = np.cos(sgb/p) * np.sin(sgl/p)
    z = np.sin(sgb/p)
    
    return np.array([x, y, z]).T

def linear_tfr(x, tf_params):
    """Linear Tully-Fisher relation."""
    a0, a1 = tf_params['a0'], tf_params['a1']
    return a1 * x + a0

def curved_tfr(x, tf_params):
    """Curved Tully-Fisher relation with break point."""
    a0, a1, a2, d = tf_params['a0'], tf_params['a1'], tf_params['a2'], tf_params['d']
    
    linear = a1 * x + a0
    curvature = a2 * (x - d)**2
    curvature_mask = x > d
    
    return linear + curvature * curvature_mask

def schechter_mass_function(log_m, params=None):
    """Schechter mass function for HI masses."""
    if params is None:
        params = SCHECHTER_PARAMS
    
    phi_star = params['phi_star']
    log_m_star = params['log_m_star'] 
    alpha = params['alpha']
    
    L = (10**log_m) / (10**log_m_star)
    phi = np.log10(10) * phi_star * L**(alpha + 1) * np.exp(-L)
    
    return phi / np.sum(phi)

def completeness_function(log_w, s=0.312, i=-0.282):
    """Completeness function for flux-limited surveys."""
    tt = log_w * s + i
    
    S_50 = []
    for w in log_w:
        if w < 2.5:
            S_90 = 0.5 * w - 1.11
        else:
            S_90 = w - 2.36
        S_50.append(S_90 + tt)
    
    return np.array(S_50)

class MockDataLoader:
    """Load real galaxy positions and properties for mock generation."""
    
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.data_loaded = False
        
        # Try to load CF4 data files
        self._load_cf4_data()
    
    def _load_cf4_data(self):
        """Load CF4 survey data if available."""
        try:
            # Try loading pre-computed arrays
            self.cf4_inclinations = np.load('data/cf4_data_names.npy')[3].astype(float)
            self.cf4_inclinations = np.deg2rad(self.cf4_inclinations)
            
            # Try loading pre-computed PV predictions  
            try:
                self.alfalfa_cf4_pred = np.load('data/alfalfa_cf4_pv_model.npy')
                self.full_cf4_pred = np.load('data/full_cf4_pv_model.npy')
            except FileNotFoundError:
                self.alfalfa_cf4_pred = None
                self.full_cf4_pred = None
                
            self.data_loaded = True
            
        except FileNotFoundError:
            warnings.warn("CF4 data files not found - using synthetic positions")
            self.data_loaded = False
            # Generate default inclination distribution
            self.cf4_inclinations = np.random.uniform(
                np.deg2rad(30), np.deg2rad(80), 1000
            )
    
    def sample_cf4_positions(self, n_galaxies=200, scale_redshift=1.0, seed=None):
        """Sample positions from CF4 catalog if available."""
        if not self.data_loaded:
            return self._generate_synthetic_positions(n_galaxies, seed)
        
        try:
            # Try to load from FITS file
            with fits.open('cf4_full.fit') as hdul:
                data_table = Table(hdul[1].data)
                
                # Extract observables
                cf4_data = {
                    'ra': np.array(data_table['_RA']),
                    'dec': np.array(data_table['_DE']),
                    'log_w': np.array(data_table['logWmxi']),
                    'w_err': np.array(data_table['e_logWmxi']),
                    'z_helio': scale_redshift * np.array(data_table['HRV']) / C_LIGHT,
                    'm21': np.array(data_table['m21']),
                    'w1_mag': np.array(data_table['W1_mag']),
                    'i_mag': np.array(data_table['i_mag']),
                    'inc': np.array(data_table['Inc']),
                    'inc_err': np.array(data_table['e_Inc']),
                }
                
                # Apply CMB correction if available
                if HELIO_CMB_AVAILABLE:
                    cf4_data['z_cmb'] = perform_corr(
                        cf4_data['z_helio'], cf4_data['ra'], cf4_data['dec']
                    )
                else:
                    cf4_data['z_cmb'] = cf4_data['z_helio']
                
                # Calculate PV field if available
                if PVHUB_AVAILABLE:
                    du = sgxyz(cf4_data['ra'], cf4_data['dec'])
                    pv = pvhub.calculate_pv(cf4_data['ra'], cf4_data['dec'], cf4_data['z_cmb'])
                    
                    if VEXT_SGXYZ is not None:
                        vext_r = np.dot(du, VEXT_SGXYZ)
                        cf4_data['pv_norm'] = (pv - vext_r) / DEFAULT_PV_PARAMS['beta']
                    else:
                        cf4_data['pv_norm'] = pv / DEFAULT_PV_PARAMS['beta']
                else:
                    cf4_data['pv_norm'] = np.zeros(len(cf4_data['ra']))
                
                # Sample subset
                np.random.seed(seed)
                n_available = len(cf4_data['ra'])
                indices = np.random.choice(n_available, size=n_galaxies, replace=True)
                
                sampled_data = {}
                for key, values in cf4_data.items():
                    sampled_data[key] = values[indices]
                
                return sampled_data
                
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"Error loading CF4 data: {e} - using synthetic positions")
            return self._generate_synthetic_positions(n_galaxies, seed)
    
    def _generate_synthetic_positions(self, n_galaxies, seed=None):
        """Generate synthetic galaxy positions and properties."""
        np.random.seed(seed)
        
        # Generate uniform sky coverage
        ra = np.random.uniform(0, 360, n_galaxies)
        dec_uniform = np.random.uniform(-1, 1, n_galaxies)
        dec = np.rad2deg(np.arcsin(dec_uniform))
        
        # Generate redshift distribution (roughly volume-limited)
        z_helio = np.random.uniform(0.005, 0.05, n_galaxies)
        z_cmb = z_helio  # Simplified
        
        # Generate log widths
        log_w = np.random.normal(2.5, 0.3, n_galaxies)
        
        # Generate inclinations
        inclinations = np.random.choice(self.cf4_inclinations, size=n_galaxies, replace=True)
        
        # Simple PV field (if available)
        if PVHUB_AVAILABLE:
            try:
                pv = pvhub.calculate_pv(ra, dec, z_cmb)
                du = sgxyz(ra, dec)
                if VEXT_SGXYZ is not None:
                    vext_r = np.dot(du, VEXT_SGXYZ)
                    pv_norm = (pv - vext_r) / DEFAULT_PV_PARAMS['beta']
                else:
                    pv_norm = pv / DEFAULT_PV_PARAMS['beta']
            except:
                pv_norm = np.zeros(n_galaxies)
        else:
            pv_norm = np.zeros(n_galaxies)
        
        return {
            'ra': ra,
            'dec': dec,
            'z_helio': z_helio,
            'z_cmb': z_cmb,
            'log_w': log_w,
            'w_err': np.random.uniform(0.05, 0.15, n_galaxies),
            'inc': inclinations,
            'inc_err': np.random.uniform(0.05, 0.1, n_galaxies),
            'pv_norm': pv_norm,
            'm21': np.random.uniform(13, 17, n_galaxies),
            'i_mag': np.random.uniform(12, 16, n_galaxies),
            'w1_mag': np.random.uniform(11, 15, n_galaxies),
        }

class TullyFisherMockGenerator:
    """Generate realistic mock Tully-Fisher data with observational effects."""
    
    def __init__(self, tf_params=None, pv_params=None, cosmology=None, 
                 data_loader=None, seed=None):
        """
        Initialize mock generator.
        
        Parameters:
        -----------
        tf_params : dict
            Tully-Fisher relation parameters
        pv_params : dict  
            Peculiar velocity parameters
        cosmology : astropy.cosmology object
            Cosmological model
        data_loader : MockDataLoader
            Data loader for real galaxy positions
        seed : int
            Random seed
        """
        self.tf_params = tf_params or DEFAULT_TF_PARAMS.copy()
        self.pv_params = pv_params or DEFAULT_PV_PARAMS.copy()
        self.cosmo = cosmology or DEFAULT_COSMOLOGY
        self.data_loader = data_loader or MockDataLoader()
        self.seed = seed
        
        # Observational bias parameters
        self.inclination_bias = 15.0  # degrees
        self.scatter_slope = -0.6
        self.scatter_intercept = 2.0
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    def rms_scatter(self, log_w):
        """Magnitude-dependent scatter (CF4-like)."""
        return self.scatter_slope * log_w + self.scatter_intercept
    
    def apply_tf_relation(self, log_w, use_curvature=True, add_scatter=True):
        """Apply Tully-Fisher relation with optional scatter."""
        log_w_offset = log_w - 2.5
        
        if use_curvature:
            M_abs = curved_tfr(log_w_offset, self.tf_params)
        else:
            M_abs = linear_tfr(log_w_offset, self.tf_params)
        
        if add_scatter:
            scatter = np.abs(self.rms_scatter(log_w_offset))
            M_abs += np.random.normal(0, scatter, len(M_abs))
        
        return M_abs
    
    def apply_inclination_bias(self, log_w, inclinations):
        """Apply inclination-dependent bias to velocity widths."""
        # Convert to velocities
        velocities = 10**(log_w + 2.5)
        
        # Apply inclination correction
        v_corrected = velocities / np.sin(inclinations)
        
        # Add inclination bias
        inc_deg = np.rad2deg(inclinations)
        delta_inc = inc_deg * (-self.inclination_bias / 45) + self.inclination_bias
        inc_biased = inclinations + np.deg2rad(delta_inc)
        
        # Measure with biased inclinations
        v_measured = v_corrected * np.sin(inc_biased)
        log_w_biased = np.log10(v_measured) - 2.5
        
        return log_w_biased, inc_biased
    
    def calculate_distances(self, z_obs, pv_field=None, add_pv=True):
        """Calculate distances including peculiar velocity effects."""
        if add_pv and pv_field is not None:
            # Apply PV correction
            z_hubble = (1 + z_obs) / (1 + pv_field / C_LIGHT) - 1
        else:
            z_hubble = z_obs
        
        # Calculate distances
        try:
            d_comoving = self.cosmo.comoving_distance(z_hubble).value
            d_luminosity = (1 + z_obs) * d_comoving
        except:
            # Fallback linear approximation
            d_comoving = z_hubble * C_LIGHT / 100.0  # Simplified
            d_luminosity = (1 + z_obs) * d_comoving
        
        return d_comoving, d_luminosity, z_hubble
    
    def generate_mock_sample(self, n_galaxies=200, add_pv=True, 
                           add_inclination_bias=True, add_measurement_errors=True,
                           use_curvature=True, uniform_coords=False,
                           scale_redshift=1.0):
        """
        Generate complete mock galaxy sample.
        
        Parameters:
        -----------
        n_galaxies : int
            Number of galaxies to generate
        add_pv : bool
            Include peculiar velocity effects
        add_inclination_bias : bool
            Include inclination bias
        add_measurement_errors : bool
            Add observational errors
        use_curvature : bool
            Use curved TF relation
        uniform_coords : bool
            Use uniform sky distribution instead of real positions
        scale_redshift : float
            Scale factor for redshifts
            
        Returns:
        --------
        dict : Mock galaxy data
        """
        np.random.seed(self.seed)
        
        # Get galaxy positions and basic properties
        if uniform_coords:
            # Generate uniform sky positions
            ra = np.random.uniform(0, 360, n_galaxies)
            dec_uniform = np.random.uniform(-1, 1, n_galaxies)
            dec = np.rad2deg(np.arcsin(dec_uniform))
            z_obs = np.random.uniform(0.005, 0.05, n_galaxies) * scale_redshift
            log_w = np.random.normal(2.5, 0.3, n_galaxies)
            inclinations = np.random.uniform(np.deg2rad(30), np.deg2rad(80), n_galaxies)
            
            # Calculate PV field for uniform positions
            if PVHUB_AVAILABLE and add_pv:
                try:
                    pv_field = pvhub.calculate_pv(ra, dec, z_obs)
                    du = sgxyz(ra, dec)
                    if VEXT_SGXYZ is not None:
                        vext_r = np.dot(du, VEXT_SGXYZ)
                        pv_norm = (pv_field - vext_r) / self.pv_params['beta']
                    else:
                        pv_norm = pv_field / self.pv_params['beta']
                except:
                    pv_field = np.zeros(n_galaxies)
                    pv_norm = np.zeros(n_galaxies)
            else:
                pv_field = np.zeros(n_galaxies)
                pv_norm = np.zeros(n_galaxies)
                
        else:
            # Use real galaxy positions
            real_data = self.data_loader.sample_cf4_positions(
                n_galaxies, scale_redshift=scale_redshift, seed=self.seed
            )
            ra = real_data['ra']
            dec = real_data['dec']
            z_obs = real_data['z_cmb'] if 'z_cmb' in real_data else real_data['z_helio']
            log_w = real_data['log_w']
            inclinations = np.deg2rad(real_data['inc']) if 'inc' in real_data else np.random.uniform(np.deg2rad(30), np.deg2rad(80), n_galaxies)
            pv_norm = real_data.get('pv_norm', np.zeros(n_galaxies))
            pv_field = pv_norm * self.pv_params['beta']
        
        # Apply inclination bias
        if add_inclination_bias:
            log_w_biased, inc_biased = self.apply_inclination_bias(log_w, inclinations)
        else:
            log_w_biased = log_w.copy()
            inc_biased = inclinations.copy()
        
        # Calculate distances
        d_comoving, d_luminosity, z_hubble = self.calculate_distances(
            z_obs, pv_field if add_pv else None, add_pv
        )
        
        # Apply Tully-Fisher relation
        M_abs = self.apply_tf_relation(log_w, use_curvature, add_scatter=True)
        
        # Calculate apparent magnitudes
        distance_modulus = 5 * np.log10(d_luminosity * 1e6) - 5  # Convert Mpc to pc
        m_app = M_abs + distance_modulus
        
        # Add measurement errors
        if add_measurement_errors:
            log_w_err = np.random.uniform(0.05, 0.15, n_galaxies)
            m_err = np.random.uniform(0.1, 0.3, n_galaxies)
            
            log_w_obs = log_w_biased + np.random.normal(0, log_w_err)
            m_obs = m_app + np.random.normal(0, m_err)
        else:
            log_w_err = np.zeros(n_galaxies)
            m_err = np.zeros(n_galaxies)
            log_w_obs = log_w_biased
            m_obs = m_app
        
        # Compile results
        mock_data = {
            # Observables
            'ra': ra,
            'dec': dec,
            'z_obs': z_obs,
            'log_w': log_w_obs,
            'm_app': m_obs,
            'log_w_err': log_w_err,
            'm_err': m_err,
            
            # True values
            'z_hubble': z_hubble,
            'M_abs': M_abs,
            'log_w_true': log_w,
            'inclinations_true': inclinations,
            'inclinations_biased': inc_biased,
            
            # Derived quantities
            'd_comoving': d_comoving,
            'd_luminosity': d_luminosity,
            'pv_field': pv_field,
            'pv_norm': pv_norm,
            
            # Supergalactic coordinates
            'du': sgxyz(ra, dec),
            'vext_r': np.dot(sgxyz(ra, dec), VEXT_SGXYZ) if VEXT_SGXYZ is not None else np.zeros(n_galaxies),
            
            # Metadata
            'n_galaxies': n_galaxies,
            'tf_params': self.tf_params.copy(),
            'pv_params': self.pv_params.copy(),
            'seed': self.seed,
            'flags': {
                'add_pv': add_pv,
                'add_inclination_bias': add_inclination_bias,
                'add_measurement_errors': add_measurement_errors,
                'use_curvature': use_curvature,
                'uniform_coords': uniform_coords,
            }
        }
        
        return mock_data

def generate_tf_mock_data(n_galaxies=200, tf_model='curved', add_pv=True,
                         seed=None, **kwargs):
    """
    Convenience function to generate mock TF data.
    
    Parameters:
    -----------
    n_galaxies : int
        Number of galaxies
    tf_model : str
        'linear' or 'curved'
    add_pv : bool
        Include peculiar velocities
    seed : int
        Random seed
    **kwargs : dict
        Additional parameters for TullyFisherMockGenerator
        
    Returns:
    --------
    dict : Mock data dictionary
    """
    generator = TullyFisherMockGenerator(seed=seed, **kwargs)
    
    return generator.generate_mock_sample(
        n_galaxies=n_galaxies,
        add_pv=add_pv,
        use_curvature=(tf_model == 'curved')
    )

# Example usage and testing
if __name__ == "__main__":
    print("Tully-Fisher Mock Data Generator")
    print("=" * 40)
    
    # Test basic functionality
    print("Generating mock data...")
    
    try:
        # Generate simple mock sample
        mock_data = generate_tf_mock_data(
            n_galaxies=50,
            tf_model='curved', 
            add_pv=True,
            seed=42
        )
        
        print(f"Generated {mock_data['n_galaxies']} galaxies")
        print(f"Redshift range: {mock_data['z_obs'].min():.4f} - {mock_data['z_obs'].max():.4f}")
        print(f"Log width range: {mock_data['log_w'].min():.2f} - {mock_data['log_w'].max():.2f}")
        print(f"Apparent mag range: {mock_data['m_app'].min():.2f} - {mock_data['m_app'].max():.2f}")
        
        if mock_data['flags']['add_pv']:
            pv_range = mock_data['pv_field']
            print(f"PV field range: {pv_range.min():.1f} - {pv_range.max():.1f} km/s")
        
        print("✓ Mock data generation successful!")
        
    except Exception as e:
        print(f"✗ Error generating mock data: {e}")
        print("This may be due to missing data files or dependencies")
    
    # Test different configurations
    print("\nTesting different configurations...")
    
    configs = [
        {'tf_model': 'linear', 'add_pv': False},
        {'tf_model': 'curved', 'add_pv': True},
        {'uniform_coords': True, 'tf_model': 'linear'},
    ]
    
    for i, config in enumerate(configs):
        try:
            test_data = generate_tf_mock_data(n_galaxies=20, seed=i, **config)
            print(f"✓ Configuration {i+1}: {config}")
        except Exception as e:
            print(f"✗ Configuration {i+1} failed: {e}")
    
    print("\nMock data generator ready for use!")
