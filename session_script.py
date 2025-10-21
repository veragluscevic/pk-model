import numpy as np
import matplotlib.pyplot as plt


def plot_power_law_function(a_vals, b_vals, g_vals, k_range=(1e-3, 10), n_points=1000):
    """
    Plot the function [1 + (a*k)^b]^g for varying parameters.
    
    Creates 3 subplots showing how each parameter affects the function shape:
    - Varying 'a': Controls the SCALE - where the transition from ~1 to power-law occurs.
                   Small a → transition at large k; Large a → transition at small k
    - Varying 'b': Controls the STEEPNESS of the power law (ak)^b in the limit ak >> 1
                   Larger b → steeper power law
    - Varying 'g': Controls the OVERALL POWER of the function
                   g < 0 → suppression at large k (damping)
                   g > 0 → enhancement at large k (boost)
    
    Parameters:
    -----------
    a_vals : list or array
        Values for parameter 'a' (scale parameter)
    b_vals : list or array
        Values for parameter 'b' (power-law index)
    g_vals : list or array
        Values for parameter 'g' (overall power)
    k_range : tuple, optional
        Range of k values (k_min, k_max). Default: (1e-3, 10)
    n_points : int, optional
        Number of points to evaluate. Default: 1000
    """
    # Convert to arrays
    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)
    g_vals = np.array(g_vals)
    
    # Create k array
    k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_points)
    
    # Get middle values for fixed parameters (use middle index)
    n_vals = len(a_vals)
    mid_idx = n_vals // 2
    a_mid = a_vals[mid_idx]
    b_mid = b_vals[mid_idx]
    g_mid = g_vals[mid_idx]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Color map and line styles for better visibility with more curves
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_vals))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # 5 different styles
    
    # Plot 1: Vary 'a'
    ax = axes[0]
    for i, a in enumerate(a_vals):
        y = (1 + (a * k)**b_mid)**g_mid
        ax.loglog(k, y, color=colors[i], linestyle=linestyles[i % len(linestyles)], 
                  linewidth=2, label=f'a = {a:.2f}')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$[1 + (ak)^b]^g$', fontsize=12)
    ax.set_title(f'Varying a (b={b_mid:.2f}, g={g_mid:.2f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Vary 'b'
    ax = axes[1]
    for i, b in enumerate(b_vals):
        y = (1 + (a_mid * k)**b)**g_mid
        ax.loglog(k, y, color=colors[i], linestyle=linestyles[i % len(linestyles)], 
                  linewidth=2, label=f'b = {b:.2f}')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$[1 + (ak)^b]^g$', fontsize=12)
    ax.set_title(f'Varying b (a={a_mid:.2f}, g={g_mid:.2f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Vary 'g'
    ax = axes[2]
    for i, g in enumerate(g_vals):
        y = (1 + (a_mid * k)**b_mid)**g
        ax.loglog(k, y, color=colors[i], linestyle=linestyles[i % len(linestyles)], 
                  linewidth=2, label=f'g = {g:.2f}')
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$[1 + (ak)^b]^g$', fontsize=12)
    ax.set_title(f'Varying g (a={a_mid:.2f}, b={b_mid:.2f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Define parameter values to explore
    # For k in [0.001, 10], these values show clear variation
    a_values = [0.05, 0.2, 0.5, 1.0, 3.0]       # Scale: where transition happens
    b_values = [0.3, 0.7, 1.0, 1.5, 2.5]        # Steepness of power law
    g_values = [-3.0, -2.0, -1.0, -0.5, -0.1]   # Overall power (all negative for damping)
    
    # Create the plots
    fig = plot_power_law_function(a_values, b_values, g_values)

