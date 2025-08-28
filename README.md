# Simultaneously fitting the Tully-Fisher relation and peculiar velocity models

## Overview

This code allows you to:

- Fit various types of Tully-Fisher relations (linear, curved) and models for intrinsic scatter 
- Fit peculiar velocity models
- Handle magnitude selection effects
- Perform Bayesian parameter estimation with MCMC
- Constrain dipole and/or quadrupole Tully-Fisher zeropoint anisotropies
- Compare goodness-of-fits between models

Examples are demonstrated on the [Cosmicflows-4 Tully Fisher data](https://ui.adsabs.harvard.edu/abs/2020ApJ...902..145K/abstract).

For background, derivations, and details see:
- [Boubel et al. 2024a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531...84B/abstract)
- [Boubel et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.533.1550B/abstract)
- [Boubel et al. 2025](https://ui.adsabs.harvard.edu/abs/2025JCAP...03..066B/abstract)

## Installation

### Requirements

```bash
pip install numpy astropy scipy emcee healpy
```

### Optional Dependencies

For peculiar velocity field calculations, you'll also need pvhub:
```bash
pip install git+https://github.com/KSaid-1/pvhub.git
```

This package provides access to various peculiar velocity field models including 2M++ reconstructions.

## Quick Start

### Basic Tully-Fisher Analysis

```python
import numpy as np
from tf_analysis import TullyFisherAnalysis, run_mcmc_analysis

# Create analysis object
analysis = TullyFisherAnalysis(
    tf_model='linear',        # 'linear', 'curved', or 'parabolic'
    scatter_model='fixed',    # 'fixed' or 'linear'
    include_pv=False         # Set to True for peculiar velocity modeling
)

# Prepare your data
data = analysis.prepare_data(
    ra=ra,           # Right ascension (degrees)
    dec=dec,         # Declination (degrees)
    z=redshift,      # Redshift
    log_w=log_width, # Log of velocity width
    m=magnitude,     # Apparent magnitude
    xerr=w_errors,   # Errors in log_width
    yerr=m_errors    # Errors in magnitude
)

# Run MCMC
results = run_mcmc_analysis(
    analysis, data, 
    n_walkers=32, 
    n_steps=5000, 
    burn_in=500
)

# Analyze results
samples = results['samples']
print(f"Parameter means: {samples.mean(axis=0)}")
print(f"Parameter std: {samples.std(axis=0)}")
```

### With Peculiar Velocity Modeling

```python
# First install pvhub if you haven't already:
# pip install git+https://github.com/KSaid-1/pvhub.git

from pvhub import pvhub

# Initialize pvhub with desired model (e.g., 2M++)
pvhub.choose_model(3)  # 2M++ model

# Calculate PV field values for your galaxies
pv_field = pvhub.calculate_pv(ra, dec, redshift)

# Include peculiar velocity field data
analysis = TullyFisherAnalysis(
    tf_model='curved',
    scatter_model='linear',
    include_pv=True,
    sigma_v=150  # km/s velocity scatter
)

# You need to provide peculiar velocity field values
data = analysis.prepare_data(
    ra=ra, dec=dec, z=redshift,
    log_w=log_width, m=magnitude,
    xerr=w_errors, yerr=m_errors,
    pv_field=pv_field  # PV field from pvhub
)
```

## Model Options

### Tully-Fisher Models

1. **Linear**: `M = a₀ + a₁(log W - 2.5)`
2. **Curved**: Linear + quadratic term for `log W > 2.5`
3. **Parabolic**: `M = a₀ + a₁(log W - 2.5) + a₂(log W - 2.5)²`

### Scatter Models

1. **Fixed**: Constant intrinsic scatter `σ = ε₀`
2. **Linear**: Variable scatter `σ = ε₀ + ε₁(log W - 2.5)`

### Parameters

The number of parameters depends on your model choice:

| Model | Fixed Scatter | Linear Scatter | + Peculiar Velocity |
|-------|--------------|----------------|-------------------|
| Linear | 3 | 4 | +4 |
| Curved | 4 | 5 | +4 |
| Parabolic | 4 | 5 | +4 |

**Parameter meanings:**
- `a₀`: Zero-point of TF relation
- `a₁`: Slope of TF relation  
- `a₂`: Curvature term (curved/parabolic models)
- `ε₀`: Intrinsic scatter (zero-point)
- `ε₁`: Scatter slope (linear scatter model)
- `β`: Peculiar velocity coupling (PV models)
- `vₓ, vᵧ, vᵤ`: External bulk flow components (PV models)

## Data Format

Your data should include:

- `ra`, `dec`: Coordinates in degrees (J2000)
- `z`: Heliocentric redshift
- `log_w`: Log₁₀ of integrated HI line width (km/s)
- `m`: Apparent magnitude in your chosen band
- `xerr`, `yerr`: Measurement uncertainties
- `pv_field`: Peculiar velocity field values (if using PV modeling)

## Peculiar Velocity Field Setup

If you want to include peculiar velocity corrections in your analysis, you'll need to set up pvhub:

### Installing pvhub

```bash
# Install directly from GitHub
pip install git+https://github.com/KSaid-1/pvhub.git

# Or clone and install locally
git clone https://github.com/KSaid-1/pvhub.git
cd pvhub
pip install -e .
```

### Available PV Models

pvhub provides access to several peculiar velocity field reconstructions:

```python
from pvhub import pvhub

pvhub.choose_model(3)  # Select 2M++ model

# Calculate PV field for your galaxy sample
pv_field = pvhub.calculate_pv(ra_deg, dec_deg, redshift)
```

### PV Field Integration

The code automatically handles the normalization using Carrick et al. parameters:

```python
# The prepare_data method automatically:
# 1. Converts coordinates to supergalactic frame
# 2. Applies Carrick et al. external velocity correction  
# 3. Normalizes using β_Carrick = 0.43

data = analysis.prepare_data(
    ra=ra, dec=dec, z=z,
    log_w=log_w, m=m, 
    xerr=xerr, yerr=yerr,
    pv_field=pv_field  # Raw pvhub output
)
```

## Advanced Features

### Selection Effects

The code includes sophisticated selection function modeling to account for magnitude-limited surveys:

#### Simple Magnitude Cut

For a basic magnitude limit:

```python
# Simple cut at m_lim = 16.0
selection = [16.0]

results = run_mcmc_analysis(
    analysis, data, 
    selection=selection,  # Apply selection correction
    n_walkers=32, n_steps=5000
)
```

#### Complex Selection Function

For surveys with gradual selection effects:

```python
# SDSS-like selection function
# F(m) = 1 for m <= m1, with smooth transition for m > m1
selection = [13.0, 13.8, -0.11]  # [m1, m2, a]

# WISE-like selection
selection = [12.0, 13.5, -0.10]  # [m1, m2, a]

results = run_mcmc_analysis(analysis, data, selection=selection)
```

The selection function has the form:
- `F(m) = 1` for `m ≤ m₁` (complete sample)  
- `F(m) = 1 + (10^(a(m-m₂)² - a(m₁-m₂)² - 0.6(m-m₁)) - 1)` for `m > m₁`

Where:
- `m₁`: Bright magnitude limit (complete sample)
- `m₂`: Transition magnitude 
- `a`: Selection steepness parameter

## Output Analysis

### Basic Statistics

```python
import matplotlib.pyplot as plt

samples = results['samples']
param_names = ['a0', 'a1', 'epsilon_0']  # Adjust based on your model

# Parameter estimates
for i, name in enumerate(param_names):
    mean_val = samples[:, i].mean()
    std_val = samples[:, i].std()
    print(f"{name}: {mean_val:.3f} ± {std_val:.3f}")

# Plot parameter distributions
fig, axes = plt.subplots(len(param_names), 1, figsize=(8, 6))
for i, (ax, name) in enumerate(zip(axes, param_names)):
    ax.hist(samples[:, i], bins=50, alpha=0.7)
    ax.set_xlabel(name)
    ax.axvline(samples[:, i].mean(), color='red', linestyle='--')
```

### Corner Plots

```python
import corner

# Create corner plot showing parameter correlations
corner.corner(samples, labels=param_names)
```

### Model Comparison

Use information criteria to compare different models:

```python
# Calculate BIC for model comparison
n_params = samples.shape[1]
n_data = len(data['m'])
max_log_like = results['lnprob'].max()
bic = -2 * max_log_like + n_params * np.log(n_data)
print(f"BIC: {bic:.2f}")
```
