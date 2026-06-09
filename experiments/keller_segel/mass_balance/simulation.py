import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax as jax
import sys
import os
from density import generate_density_estimation
jax.config.update("jax_enable_x64", True)
os.makedirs("samples", exist_ok=True)

def generate_samples(target_density, batch_size=10_000_0, period=jnp.array([[0, 2*np.pi], [0, 2*np.pi]]),max_value=1.0):
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



def step(X1, X2,rng):
    coeff_rho_1 = fcn_density_estimation(X1)
    coeff_rho_2 = fcn_density_estimation(X2)
    rng,key = jax.random.split(rng)
    random_X1 = jax.random.normal(key, shape=(X1.shape[0],2), dtype=jnp.float64)
    rng,key = jax.random.split(rng)
    random_X2 = jax.random.normal(key, shape=(X2.shape[0],2), dtype=jnp.float64)
    F_1 = (X2.shape[0]/n_samples)*jax.vmap(fcn_grad_density_eval, in_axes=(0, None))(X1, coeff_rho_2)/C_v
    dX1 = F_1 * dt + jnp.sqrt(2*dt)*random_X1
    dX2 = jnp.sqrt(2*dt)*random_X2
    X1 = X1 + dX1
    X2 = X2 + dX2
    
    rho_1_X2 = jax.vmap(fcn_density_evaluate,(0,None))(X2, coeff_rho_1)
    rho_2_X2 = jax.vmap(fcn_density_evaluate,(0,None))(X2, coeff_rho_2)*(X2.shape[0]/n_samples)
    alpha = C_v*rho_1_X2/rho_2_X2/C_u -1
    
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X2.shape[0],))
    death = (alpha < 0) & (random_numbers < (1-jnp.exp(alpha*dt)))
    rng, key = jax.random.split(rng)
    random_numbers = jax.random.uniform(key, shape=(X2.shape[0],))
    duplicate = (alpha > 0) & (random_numbers < (1-jnp.exp(-alpha*dt)))
    
    return X1, X2, rng, death, duplicate



def target_density_u(x,y):
    return (jnp.sin(x)**2 * jnp.cos(y)**2)*C_u

def target_density_v(x,y):
    return (jnp.cos(x) + jnp.cos(y) + 2)*C_v


fcn_density_estimation, fcn_density_evaluate  = generate_density_estimation(n_freq=5, extend="periodic", period=jnp.array([[0, 2*np.pi], [0, 2*np.pi]]))
fcn_grad_density_eval  = jax.grad(fcn_density_evaluate, argnums=0)


C_u = 1/np.pi**2
C_v = 1/(8*np.pi**2)
fcn_samples_u = generate_samples(target_density_u, period=jnp.array([[0, 2*np.pi], [0, 2*np.pi]]), max_value=1.0/np.pi**2)
fcn_samples_v = generate_samples(target_density_v, period=jnp.array([[0, 2*np.pi], [0, 2*np.pi]]), max_value=4.0/(8*np.pi**2))
n_samples = int(sys.argv[1])
rng = jax.random.PRNGKey(0)
X1 = fcn_samples_u(rng, n_samples)
X2 = fcn_samples_v(rng, n_samples)

T =  2e-1
dt = 1e-3
N = int(T / dt)

Samples = {}
for i in range(N):
    print(f"Iteration: {i+1}/{N}: numerical density: {X2.shape[0]/n_samples}, exact: {C_v/C_u + (1-C_v/C_u)*jnp.exp(-i*dt)}", end="\r")
    rng, key = jax.random.split(rng)
    X1, X2, rng, death, duplicate = step(X1, X2, rng)
    X2 = np.concatenate([X2[~death], X2[duplicate]], axis=0)
    X1 = np.mod(X1, 2*np.pi)
    X2 = np.mod(X2, 2*np.pi)
    Samples[f"X1_{i}"] = X1
    Samples[f"X2_{i}"] = X2
    
np.savez(f"samples/samples_{n_samples}.npz", **Samples)