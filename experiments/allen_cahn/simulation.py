import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax as jax
import sys
import os

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

def target_density_u(x,y):
    return (jnp.sin(x)**2 * jnp.cos(y)**2)*C_u

C_u = 1/jnp.pi**2
fcn_samples_u = generate_samples(target_density_u, period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]), max_value=1.0/jnp.pi**2)


rng = jax.random.PRNGKey(0)
key, rng = jax.random.split(rng)
X1_tmp = fcn_samples_u(rng,n_samples)
X1 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
X1 = X1.at[:X1_tmp.shape[0]].set(X1_tmp)
X1_mask = jnp.zeros((X1.shape[0],), dtype=bool)
X1_mask = X1_mask.at[:X1_tmp.shape[0]].set(True)



fcn_density_estimation, fcn_density_evaluate  = generate_density_estimation(n_freq=10, extend="periodic", period=jnp.array([[0, 2*jnp.pi], [0, 2*jnp.pi]]))

coeff_rho_1 = fcn_density_estimation(X1,X1_mask)


@jax.jit
def step(X1, X1_mask,rng):
    coeff_rho_1 = fcn_density_estimation(X1,X1_mask)
    
    n_X1  = jnp.sum(X1_mask)
        
    rng,key = jax.random.split(rng)
    random_X1 = jax.random.normal(key, shape=(X1.shape[0],2), dtype=jnp.float64)
    
    u_X1 = (n_X1/n_samples)*jax.vmap(fcn_density_evaluate,(0,None))(X1, coeff_rho_1)/C_u
    
    dX1 = jnp.sqrt(2*0.01*dt)*random_X1
    X1 = X1 + dX1

    alpha_X1 = 1- (u_X1)**2

    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X1.shape[0],))
    death_X1 = (alpha_X1 < 0) & (random_numbers < (1-jnp.exp(alpha_X1*dt)))
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X1.shape[0],))
    duplicate_X1 = (alpha_X1 > 0) & (random_numbers < (1-jnp.exp(-alpha_X1*dt)))
    
    
    return X1, rng, death_X1, duplicate_X1



T =  2
dt = 1e-3
N = int(T/dt)

Samples = {}

for i in range(N):
    print(f"Iteration: {i+1}/{N}: numerical density: {jnp.sum(X1_mask)/n_samples}", end="\r")
    rng, key = jax.random.split(rng)
    X1,  rng, death_X1, duplicate_X1 = step(X1, X1_mask,rng)
    
    X1_tmp = X1[X1_mask]
    death_X1 = death_X1[X1_mask]
    duplicate_X1 = duplicate_X1[X1_mask]
    
    X1_tmp = jnp.concatenate([X1_tmp[~death_X1], X1_tmp[duplicate_X1]], axis=0)
    X1_tmp = jnp.mod(X1_tmp, 2*jnp.pi)
    
    if X1_tmp.shape[0] > max_samples:
        max_samples = 2*max_samples
    
    X1 = jnp.zeros((max_samples, 2), dtype=jnp.float64)
    X1 = X1.at[:X1_tmp.shape[0]].set(X1_tmp)
    X1_mask = jnp.zeros((X1.shape[0],), dtype=bool)
    X1_mask = X1_mask.at[:X1_tmp.shape[0]].set(True)
    
    Samples[f"X1_{i}"] = jnp.array(X1_tmp)

     
jnp.savez(f"samples/samples_N_{n_samples}.npz", **Samples)






