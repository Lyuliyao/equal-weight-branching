import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import chex
import jax


def generate_density_estimation(n_freq = 10, extend = "periodic", period=None):
    """
    Generate a Fourier-based density estimation function.
    Input:
    - n_freq: number of Fourier modes in each direction.
    - extend: type of boundary conditions ("periodic" or "zero").
    - period: optional, period for the periodic boundary conditions.
    
     
    Output:
    - density_estimation: function to generate density estimation.
    - density_evaluate: function to evaluate the density estimation at given coordinates.
    - gradient_density_evaluate: function to evaluate the gradient of the density estimation.
    """
    
    K = n_freq
    norm = np.zeros((K, K))
    norm[0, 0] = 1 
    norm[0, 1:] = 2 
    norm[1:, 0] = 2 
    norm[1:, 1:] = 4 

    if extend == "periodic":
        if period is None:
            raise ValueError("period must be provided for periodic boundary conditions.")
        else:
            period = np.array(period)
            if period.shape != (2, 2):
                raise ValueError("period must be a 2x2 array.")
            if period[0, 0] >= period[0, 1] or period[1, 0] >= period[1, 1]:
                raise ValueError("Invalid period values.")
            
    def density_estimation(data):
        """
        Generate a density estimation using Fourier modes.
        Input:
        - data: 2D array of shape (n_samples, 2) containing x and y coordinates.
        
        Output:
        - coeffs: dictionary containing the coefficients for each Fourier mode.
        """
        
        x_data, y_data = data[:, 0], data[:, 1]
        if extend == "periodic":
            x_min = period[0][0]
            x_max = period[0][1]
            y_min = period[1][0]
            y_max = period[1][1]
        elif extend == "zero":
            x_min, x_max = jnp.min(x_data), jnp.max(x_data)
            y_min, y_max = jnp.min(y_data), jnp.max(y_data)
        Lx = x_max - x_min
        Ly = y_max - y_min
        
        
        coeffs = {}
        
        freq_k = jnp.arange(K)
        freq_l = jnp.arange(K)
        
        theta_x = 2 * jnp.pi * freq_k[None,:] * (x_data[:,None] - x_min) / Lx
        theta_y = 2 * jnp.pi * freq_l[None,:] * (y_data[:,None] - y_min) / Ly   
        
        basis_cos_cos = jnp.cos(theta_x[..., None]) * jnp.cos(theta_y[:,None, :])
        basis_cos_sin = jnp.cos(theta_x[..., None]) * jnp.sin(theta_y[:,None, :])
        basis_sin_cos = jnp.sin(theta_x[..., None]) * jnp.cos(theta_y[:,None, :])
        basis_sin_sin = jnp.sin(theta_x[..., None]) * jnp.sin(theta_y[:,None, :])
        
        norm_factor = norm / (Lx * Ly)
        
        coeffs["cos-cos"] = norm_factor * jnp.mean(basis_cos_cos, axis=0)
        coeffs["cos-sin"] = norm_factor * jnp.mean(basis_cos_sin, axis=0)
        coeffs["sin-cos"] = norm_factor * jnp.mean(basis_sin_cos, axis=0)
        coeffs["sin-sin"] = norm_factor * jnp.mean(basis_sin_sin, axis=0)
        coeffs["Lx"] = Lx
        coeffs["Ly"] = Ly
        coeffs["x_min"] = x_min
        coeffs["x_max"] = x_max
        coeffs["y_min"] = y_min
        coeffs["y_max"] = y_max
        coeffs["K"] = K
        return coeffs   
        # Normalization factor
        
    
    def density_evaluate(points,coeff):
        """
        Evaluate the density estimation at given coordinates using Fourier modes.
        Input:
        - points: 2D array of shape (n_samples, 2) containing x and y coordinates.
        - coeff: dictionary containing the coefficients for each Fourier mode.
        
        Output:
        - density: 1D array of shape (n_samples,) containing the estimated density at each point.
        """
        
        x_data, y_data = points[0], points[1]
        x_min, x_max = coeff["x_min"], coeff["x_max"]
        y_min, y_max = coeff["y_min"], coeff["y_max"]
        if extend == "periodic":
            x_data = jnp.mod(x_data - x_min, coeff["Lx"]) + x_min
            y_data = jnp.mod(y_data - y_min, coeff["Ly"]) + y_min
        Lx = coeff["Lx"]
        Ly = coeff["Ly"]
        K = coeff["K"]
        
        freq_k = jnp.arange(K)
        freq_l = jnp.arange(K)
        
        theta_x = 2 * jnp.pi * freq_k * (x_data - x_min) / Lx
        theta_y = 2 * jnp.pi * freq_l * (y_data - y_min) / Ly
        
        Z = jnp.zeros((K, K))
        Z += coeff["cos-cos"] * jnp.cos(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["cos-sin"] * jnp.cos(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        Z += coeff["sin-cos"] * jnp.sin(theta_x[..., None]) * jnp.cos(theta_y[None, :])
        Z += coeff["sin-sin"] * jnp.sin(theta_x[..., None]) * jnp.sin(theta_y[None, :])
        Z = jnp.sum(Z)
        
        if extend == "zero":
            Z =  Z*(x_data<x_max)*(x_data>x_min)*(y_data<y_max)*(y_data>y_min)
            
        return Z
    
    return density_estimation, density_evaluate
