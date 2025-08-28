#!/usr/bin/env python3
"""
Example script showing how to use the Tully-Fisher analysis code.

This script demonstrates:
1. Loading data
2. Setting up different analysis configurations  
3. Running MCMC fits
4. Analyzing and plotting results
5. Using pvhub for peculiar velocity modeling (if available)
"""

import numpy as np
import matplotlib.pyplot as plt
from tf_analysis import TullyFisherAnalysis, run_mcmc_analysis, get_pv_field, PVHUB_AVAILABLE

def generate_mock_data(n_galaxies=200, add_noise=True):
    """
    Generate mock galaxy data for testing.
    
    Returns realistic-looking galaxy data with known TF parameters.
    """
    np.random.seed(42)  # For reproducible results
    
    # True TF parameters
    a0_true = -21.0   # Absolute magnitude zero-point
    a1_true = -8.0    # TF slope
    epsilon_true = 0.3  # Intrinsic scatter
    
    # Generate mock observations
    ra = np.random.uniform(0, 360, n_galaxies)
    dec = np.random.uniform(-30, 30, n_galaxies)
    z = np.random.uniform(0.01, 0.04, n_galaxies)  # Local universe
    
    # Log velocity widths (centered around 2.5)
    log_w = np.random.normal(2.5, 0.3, n_galaxies)
    
    # True absolute magnitudes from TF relation
    M_true = a0_true + a1_true * (log_w - 2.5)
    if add_noise:
        M_true += np.random.normal(0, epsilon_true, n_galaxies)
    
    # Convert to apparent magnitudes (simplified distance modulus)
    # Using approximate relation for nearby galaxies
    distance_modulus = 5 * np.log10(z * 3e5 / 70) + 25  # H0=70
    m = M_true + distance_modulus
    
    # Add measurement errors
    xerr = np.random.uniform(0.05, 0.15, n_galaxies)  # log_w errors
    yerr = np.random.uniform(0.1, 0.3, n_galaxies)    # magnitude errors
    
    if add_noise:
        log_w += np.random.normal(0, 1) * xerr
        m += np.random.normal(0, 1) * yerr
    
    return {
        'ra': ra, 'dec': dec, 'z': z,
        'log_w': log_w, 'm': m,
        'xerr': xerr, 'yerr': yerr,
        'true_params': [a0_true, a1_true, epsilon_true]
    }

def example_basic_analysis():
    """Example 1: Basic linear TF relation fit."""
    print("="*50)
    print("Example 1: Basic Linear Tully-Fisher Analysis")
    print("="*50)
    
    # Generate mock data
    mock_data = generate_mock_data(n_galaxies=150)
    print(f"Generated {len(mock_data['ra'])} mock galaxies")
    print(f"True parameters: a0={mock_data['true_params'][0]}, "
          f"a1={mock_data['true_params'][1]}, ε={mock_data['true_params'][2]}")
    
    # Set up analysis
    analysis = TullyFisherAnalysis(
        tf_model='linear',
        scatter_model='fixed',
        include_pv=False
    )
    
    # Prepare data
    data = analysis.prepare_data(
        ra=mock_data['ra'],
        dec=mock_data['dec'], 
        z=mock_data['z'],
        log_w=mock_data['log_w'],
        m=mock_data['m'],
        xerr=mock_data['xerr'],
        yerr=mock_data['yerr']
    )
    
    # Run MCMC (short run for example)
    print("Running MCMC...")
    results = run_mcmc_analysis(
        analysis, data, 
        n_walkers=16, 
        n_steps=1000, 
        burn_in=200
    )
    
    # Analyze results
    samples = results['samples']
    param_names = ['a0', 'a1', 'epsilon_0']
    
    print("\nResults:")
    print("-" * 30)
    for i, (name, true_val) in enumerate(zip(param_names, mock_data['true_params'])):
        mean_val = samples[:, i].mean()
        std_val = samples[:, i].std()
        print(f"{name:>10}: {mean_val:6.2f} ± {std_val:4.2f} (true: {true_val:6.2f})")
    
    print(f"\nMean acceptance fraction: {results['acceptance_fraction'].mean():.3f}")
    
    return results, mock_data, param_names

def example_curved_analysis():
    """Example 2: Curved TF relation with linear scatter."""
    print("\n" + "="*50)
    print("Example 2: Curved TF with Variable Scatter")
    print("="*50)
    
    # Generate mock data with more complex behavior
    mock_data = generate_mock_data(n_galaxies=200)
    
    # Set up analysis
    analysis = TullyFisherAnalysis(
        tf_model='curved',
        scatter_model='linear',
        include_pv=False
    )
    
    # Prepare data
    data = analysis.prepare_data(
        ra=mock_data['ra'],
        dec=mock_data['dec'],
        z=mock_data['z'],
        log_w=mock_data['log_w'],
        m=mock_data['m'],
        xerr=mock_data['xerr'],
        yerr=mock_data['yerr']
    )
    
    # Run MCMC
    print("Running MCMC for curved model...")
    results = run_mcmc_analysis(
        analysis, data,
        n_walkers=20,
        n_steps=1500,
        burn_in=300
    )
    
    # Analyze results
    samples = results['samples']
    param_names = ['a0', 'a1', 'a2', 'epsilon_0', 'epsilon_1']
    
    print("\nCurved Model Results:")
    print("-" * 40)
    for i, name in enumerate(param_names):
        mean_val = samples[:, i].mean()
        std_val = samples[:, i].std()
        print(f"{name:>10}: {mean_val:6.2f} ± {std_val:4.2f}")
    
    return results, mock_data, param_names

def plot_results(results, mock_data, param_names, model_name=""):
    """Plot MCMC results and data."""
    samples = results['samples']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Parameter traces
    n_params = len(param_names)
    for i in range(n_params):
        ax = plt.subplot(3, n_params, i + 1)
        # Plot traces for first few walkers
        for j in range(min(5, results['n_walkers'])):
            plt.plot(results['chain'][j, :, i], alpha=0.7)
        plt.ylabel(param_names[i])
        plt.title(f'Trace: {param_names[i]}')
    
    # 2. Parameter distributions
    for i in range(n_params):
        ax = plt.subplot(3, n_params, i + 1 + n_params)
        plt.hist(samples[:, i], bins=30, alpha=0.7, density=True)
        mean_val = samples[:, i].mean()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.xlabel(param_names[i])
        plt.ylabel('Density')
        plt.legend()
    
    # 3. Tully-Fisher relation plot
    ax = plt.subplot(3, 2, 5)
    
    # Plot data
    plt.errorbar(mock_data['log_w'], mock_data['m'], 
                xerr=mock_data['xerr'], yerr=mock_data['yerr'],
                fmt='o', alpha=0.6, markersize=3, label='Data')
    
    # Plot best-fit model
    log_w_model = np.linspace(mock_data['log_w'].min(), mock_data['log_w'].max(), 100)
    
    # Use median parameters for best-fit line
    best_params = np.median(samples, axis=0)
    
    # Simple linear model for plotting (adjust for your actual model)
    if len(param_names) >= 2:
        a0, a1 = best_params[0], best_params[1]
        # Approximate absolute magnitude (ignoring distance effects for visualization)
        M_model = a0 + a1 * (log_w_model - 2.5)
        # Convert to rough apparent magnitude for plotting
        m_model = M_model + np.mean(mock_data['m'] - (best_params[0] + best_params[1] * (mock_data['log_w'] - 2.5)))
        plt.plot(log_w_model, m_model, 'r-', linewidth=2, label='Best fit')
    
    plt.xlabel('log W [km/s]')
    plt.ylabel('Apparent Magnitude')
    plt.title(f'Tully-Fisher Relation {model_name}')
    plt.legend()
    plt.gca().invert_yaxis()  # Brighter magnitudes at top
    
    # 4. Residuals histogram
    ax = plt.subplot(3, 2, 6)
    if len(param_names) >= 3:
        # Calculate residuals using best-fit parameters
        a0, a1 = best_params[0], best_params[1]
        m_pred = a0 + a1 * (mock_data['log_w'] - 2.5)
        # Rough residuals (ignoring distance corrections)
        residuals = mock_data['m'] - (m_pred + np.mean(mock_data['m']) - np.mean(m_pred))
        plt.hist(residuals, bins=20, alpha=0.7, density=True)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Residual Distribution')
        
        # Overplot expected scatter
        if len(param_names) >= 3:
            sigma = best_params[2] if len(param_names) == 3 else best_params[3]
            x_gauss = np.linspace(residuals.min(), residuals.max(), 100)
            y_gauss = np.exp(-0.5 * (x_gauss/sigma)**2) / (sigma * np.sqrt(2*np.pi))
            plt.plot(x_gauss, y_gauss, 'r--', label=f'Expected (σ={sigma:.2f})')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'tf_analysis_results_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()

def example_model_comparison():
    """Example 3: Compare different TF models."""
    print("\n" + "="*50)
    print("Example 3: Model Comparison")
    print("="*50)
    
    # Generate data once
    mock_data = generate_mock_data(n_galaxies=250)
    
    models_to_test = [
        ('linear', 'fixed'),
        ('linear', 'linear'),
        ('curved', 'fixed'),
        ('parabolic', 'fixed')
    ]
    
    results_comparison = {}
    
    for tf_model, scatter_model in models_to_test:
        model_name = f"{tf_model}_{scatter_model}"
        print(f"\nTesting {model_name} model...")
        
        # Set up analysis
        analysis = TullyFisherAnalysis(
            tf_model=tf_model,
            scatter_model=scatter_model,
            include_pv=False
        )
        
        # Prepare data
        data = analysis.prepare_data(
            ra=mock_data['ra'], dec=mock_data['dec'], z=mock_data['z'],
            log_w=mock_data['log_w'], m=mock_data['m'],
            xerr=mock_data['xerr'], yerr=mock_data['yerr']
        )
        
        # Run shorter MCMC for comparison
        results = run_mcmc_analysis(
            analysis, data,
            n_walkers=16, n_steps=800, burn_in=200
        )
        
        # Calculate information criteria
        n_params = results['samples'].shape[1]
        n_data = len(mock_data['m'])
        max_log_like = results['lnprob'].max()
        
        aic = -2 * max_log_like + 2 * n_params
        bic = -2 * max_log_like + n_params * np.log(n_data)
        
        results_comparison[model_name] = {
            'results': results,
            'aic': aic,
            'bic': bic,
            'n_params': n_params,
            'max_log_like': max_log_like
        }
        
        print(f"  Parameters: {n_params}")
        print(f"  Max log-likelihood: {max_log_like:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
    
    # Summary table
    print("\nModel Comparison Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'N_params':<8} {'AIC':<10} {'BIC':<10} {'Δ_AIC':<10}")
    print("-" * 60)
    
    min_aic = min([r['aic'] for r in results_comparison.values()])
    
    for model_name, r in results_comparison.items():
        delta_aic = r['aic'] - min_aic
        print(f"{model_name:<15} {r['n_params']:<8} {r['aic']:<10.1f} {r['bic']:<10.1f} {delta_aic:<10.1f}")
    
    return results_comparison

def main():
    """Run all examples."""
    print("Tully-Fisher Analysis Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic analysis
        results1, mock_data1, params1 = example_basic_analysis()
        plot_results(results1, mock_data1, params1, "Linear Fixed")
        
        # Example 2: More complex model
        results2, mock_data2, params2 = example_curved_analysis()
        plot_results(results2, mock_data2, params2, "Curved Variable")
        
        # Example 3: Model comparison
        comparison_results = example_model_comparison()
        
        # Example 4: Peculiar velocity analysis (if pvhub available)
        if PVHUB_AVAILABLE:
            pv_results = example_peculiar_velocity_analysis()
            if pv_results is not None:
                results4, mock_data4, params4 = pv_results
                plot_results(results4, mock_data4, params4, "PV Model")
        
        print("\n" + "="*50)
        print("All examples completed successfully!")
        print("Check the generated plots: tf_analysis_results_*.png")
        print("="*50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install numpy astropy scipy emcee matplotlib")

if __name__ == "__main__":
    main()
