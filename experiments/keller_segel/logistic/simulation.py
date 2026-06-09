import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax as jax
import sys
import os
from density import generate_density_estimation

print("\n=== GPU Diagnostic ===")
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_log_compiles', True)
os.makedirs("samples", exist_ok=True)
n_samples = int(sys.argv[1])
max_samples  = 2*n_samples





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
    norm = jnp.zeros((K, K))
    norm = norm.at[0, 0].set(1)
    norm = norm.at[0, 1:].set(2)
    norm = norm.at[1:, 0].set(2)
    norm = norm.at[1:, 1:].set(4)

    if extend == "periodic":
        if period is None:
            raise ValueError("period must be provided for periodic boundary conditions.")
        else:
            period = jnp.array(period)
            if period.shape != (2, 2):
                raise ValueError("period must be a 2x2 array.")
            if period[0, 0] >= period[0, 1] or period[1, 0] >= period[1, 1]:
                raise ValueError("Invalid period values.")
            
    def density_estimation(data,mask=None):
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
            x_min, x_max = jnp.min(x_data[mask]), jnp.max(x_data[mask])
            y_min, y_max = jnp.min(y_data[mask]), jnp.max(y_data[mask])
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
        if mask is not None:
            coeffs["cos-cos"] = norm_factor * jnp.sum(basis_cos_cos*mask[:,None,None], axis=0)/jnp.sum(mask)
            coeffs["cos-sin"] = norm_factor * jnp.sum(basis_cos_sin*mask[:,None,None], axis=0)/jnp.sum(mask)
            coeffs["sin-cos"] = norm_factor * jnp.sum(basis_sin_cos*mask[:,None,None], axis=0)/jnp.sum(mask)
            coeffs["sin-sin"] = norm_factor * jnp.sum(basis_sin_sin*mask[:,None,None], axis=0)/jnp.sum(mask)
        else:
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




def generate_samples(target_density, batch_size=10_000_0, period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]),max_value=1.0):
    """
    Generates samples from a target density using rejection sampling.
    Input:
        target_density: function that takes two arguments (x, y) and returns the density value.
        batch_size: number of samples to generate in each batch.
        period: the period of the target density function.
    Output:
        fcn_samples: function that takes n_samples and rng as arguments and returns the samples.
    """
    
    @jax.jit
    def sample(rng):
        rng, key = jax.random.split(rng)
        x = jax.random.uniform(key, (batch_size,), minval=period[0][0], maxval=period[0][1])
        rng, key = jax.random.split(rng)
        y = jax.random.uniform(key, (batch_size,), minval=period[1][0], maxval=period[0][1])
        rng, key = jax.random.split(rng)
        u = jax.random.uniform(key, (batch_size,), minval=0, maxval=max_value)
        keep = u < target_density(x, y)
        return x, y, keep
    
    
    def fcn_samples(rng, n_samples):
        samples = []
        sample_now = 0
        while sample_now < n_samples:
            print(f"Number of iterations: {len(samples)} and Number of samples: {sample_now}", end="\r")
            rng, key = jax.random.split(rng)
            x, y, keep = sample(rng)
            accepted = jnp.column_stack([x[keep], y[keep]])
            samples.append(accepted)
            sample_now += jnp.sum(keep)
        samples = jnp.concatenate(samples, axis=0)
        return samples[:n_samples]
    return fcn_samples


x = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
y = jnp.linspace(0, 2 * jnp.pi, 100, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = jnp.meshgrid(x, y)


C_u = 1/(3*jnp.pi)
C_v = 1/(4*jnp.pi)

def target_density_u(x,y):
    dx = x - 0.5*jnp.pi
    dy = y - 0.5*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    u = jnp.exp(-(dx**2 + dy**2))
    
    dx = x - jnp.pi
    dy = y - jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    u += jnp.exp(-(dx**2 + dy**2))
    
    dx = x - 1.5*jnp.pi
    dy = y - 1.5*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    u += jnp.exp(-(dx**2 + dy**2))
    
    return u*C_u


def target_density_v(x,y):
    dx = x - 0.4*jnp.pi
    dy = y - 0.4*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    v = jnp.exp(-(dx**2 + dy**2))
    
    dx = x - 0.8*jnp.pi
    dy = y - 0.8*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    v += jnp.exp(-(dx**2 + dy**2))
    
    dx = x - 1.2*jnp.pi
    dy = y - 1.2*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    v += jnp.exp(-(dx**2 + dy**2))
    
    dx = x - 1.6*jnp.pi
    dy = y - 1.6*jnp.pi
    dx = jnp.where(dx > jnp.pi, dx - 2*jnp.pi, dx)
    dx = jnp.where(dx < -jnp.pi, dx + 2*jnp.pi, dx)
    dy = jnp.where(dy > jnp.pi, dy - 2*jnp.pi, dy)
    dy = jnp.where(dy < -jnp.pi, dy + 2*jnp.pi, dy)
    v += jnp.exp(-(dx**2 + dy**2))
    
    return v*C_v

fcn_samples_u = generate_samples(target_density_u, period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]), max_value=3*C_u)
fcn_samples_v = generate_samples(target_density_v, period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]), max_value=4*C_v)

rng = jax.random.PRNGKey(0)
key, rng = jax.random.split(rng)
X1_tmp = fcn_samples_u(rng,n_samples)
X1 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
X1 = X1.at[:X1_tmp.shape[0]].set(X1_tmp)
X1_mask = jnp.zeros((X1.shape[0],), dtype=bool)
X1_mask = X1_mask.at[:X1_tmp.shape[0]].set(True)

X2_tmp = fcn_samples_v(rng,n_samples)
X2 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
X2 = X2.at[:X2_tmp.shape[0]].set(X2_tmp)
X2_mask = jnp.zeros((X2.shape[0],), dtype=bool)
X2_mask = X2_mask.at[:X2_tmp.shape[0]].set(True)


fcn_density_estimation, fcn_density_evaluate  = generate_density_estimation(n_freq=10, extend="periodic", period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]))
fcn_grad_density_eval  = jax.grad(fcn_density_evaluate, argnums=0)

coeff_rho_1 = fcn_density_estimation(X1,X1_mask)
coeff_rho_2 = fcn_density_estimation(X2,X2_mask)


@jax.jit
def step(X1, X1_mask, X2, X2_mask,rng):
    coeff_rho_1 = fcn_density_estimation(X1,X1_mask)
    coeff_rho_2 = fcn_density_estimation(X2,X2_mask)
    
    n_X1  = jnp.sum(X1_mask)
    n_X2  = jnp.sum(X2_mask)
        
    rng,key = jax.random.split(rng)
    random_X1 = jax.random.normal(key, shape=(X1.shape[0],2), dtype=jnp.float64)
    rng,key = jax.random.split(rng)
    random_X2 = jax.random.normal(key, shape=(X2.shape[0],2), dtype=jnp.float64)
    
    u_X1 = (n_X1/n_samples)*jax.vmap(fcn_density_evaluate,(0,None))(X1, coeff_rho_1)/C_u
    v_X1 = (n_X2/n_samples)*jax.vmap(fcn_density_evaluate,(0,None))(X1, coeff_rho_2)/C_v
    u_X2 = (n_X1/n_samples)*jax.vmap(fcn_density_evaluate,(0,None))(X2, coeff_rho_1)/C_u
    v_X2 = (n_X2/n_samples)*jax.vmap(fcn_density_evaluate,(0,None))(X2, coeff_rho_2)/C_v
    
    nabla_v_X1 = (n_X2/n_samples)*jax.vmap(fcn_grad_density_eval, in_axes=(0, None))(X1, coeff_rho_2)/C_v
    
    F_1 = 4*nabla_v_X1/(1+u_X1[...,None]**2)
    dX1 = F_1 * dt + jnp.sqrt(2*dt)*random_X1
    dX2 = jnp.sqrt(2*dt)*random_X2
    X1 = X1 + dX1
    X2 = X2 + dX2
    
    alpha_X1 = 1 - u_X1;
    alpha_X2 = u_X2/v_X2 -1;
    
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X1.shape[0],))
    death_X1 = (alpha_X1 < 0) & (random_numbers < (1-jnp.exp(alpha_X1*dt)))
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X1.shape[0],))
    duplicate_X1 = (alpha_X1 > 0) & (random_numbers < (1-jnp.exp(-alpha_X1*dt)))
    
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X2.shape[0],))
    death_X2 = (alpha_X2 < 0) & (random_numbers < (1-jnp.exp(alpha_X2*dt)))
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X2.shape[0],))
    duplicate_X2 = (alpha_X2 > 0) & (random_numbers < (1-jnp.exp(-alpha_X2*dt)))
    
    return X1, X2, rng, death_X1, duplicate_X1, death_X2, duplicate_X2



T = 1
dt = 1e-3
N = int(T/dt)

Samples = {}

for i in range(N):
    print(f"Iteration: {i+1}/{N}: numerical density: {X2.shape[0]/n_samples}, exact: {C_v/C_u + (1-C_v/C_u)*jnp.exp(-i/N)}", end="\r")
    rng, key = jax.random.split(rng)
    X1, X2, rng, death_X1, duplicate_X1, death_X2, duplicate_X2 = step(X1, X1_mask, X2, X2_mask,rng)
    
    X1_tmp = X1[X1_mask]
    X2_tmp = X2[X2_mask]
    death_X1 = death_X1[X1_mask]
    duplicate_X1 = duplicate_X1[X1_mask]
    death_X2 = death_X2[X2_mask]
    duplicate_X2 = duplicate_X2[X2_mask]
    
    X1_tmp = jnp.concatenate([X1_tmp[~death_X1], X1_tmp[duplicate_X1]], axis=0)
    X2_tmp = jnp.concatenate([X2_tmp[~death_X2], X2_tmp[duplicate_X2]], axis=0)
    X1_tmp = jnp.mod(X1_tmp, 2*jnp.pi)
    X2_tmp = jnp.mod(X2_tmp, 2*jnp.pi)
    
    if X1_tmp.shape[0] > max_samples:
        max_samples = 2*max
    
    X1 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
    X1 = X1.at[:X1_tmp.shape[0]].set(X1_tmp)
    X1_mask = jnp.zeros((X1.shape[0],), dtype=bool)
    X1_mask = X1_mask.at[:X1_tmp.shape[0]].set(True)
    
    
    X2 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
    X2 = X2.at[:X2_tmp.shape[0]].set(X2_tmp)
    X2_mask = jnp.zeros((X2.shape[0],), dtype=bool)
    X2_mask = X2_mask.at[:X2_tmp.shape[0]].set(True)
    Samples[f"X1_{i}"] = jnp.array(X1_tmp)
    Samples[f"X2_{i}"] = jnp.array(X2_tmp)  

     
jnp.savez(f"samples/samples_N_{n_samples}.npz", **Samples)