import numpy as np
import sys

N = int(sys.argv[1])
x = np.linspace(-0.5, 0.5, N, endpoint=False)
y = np.linspace(-0.5, 0.5, N, endpoint=False)
# x= np.linspace(-0.5, 0.5, 100, endpoint=False)
# y= np.linspace(-0.5, 0.5, 100, endpoint=False)
# def target_density_u(x,y):
#     return 840*jnp.exp(-84*(x**2+y**2))* C_u

# def target_density_v(x,y):
#     return 420*jnp.exp(-42*(x**2+y**2))* C_v

X, Y = np.meshgrid(x, y)
u_0 = 840*np.exp(-84*(X**2+Y**2))
v_0 = 420*np.exp(-42*(X**2+Y**2))
# u_0 = 840*np.exp(-84*(X**2+Y**2)) #np.sin(X)**2*np.cos(Y)**2
# v_0 = 420*np.exp(-42*(X**2+Y**2))#np.cos(X)+np.cos(Y)+2
dx =  x[1] - x[0]
idx = np.arange(u_0.shape[0])
idx_left = np.roll(idx, -1)
idx_right = np.roll(idx, 1)
idx_left
u  = u_0

def step(u,v):
    laplace_u = (u[idx_left, :] + u[idx_right, :] + u[:, idx_left] + u[:, idx_right] - 4 * u) / dx**2
    laplace_v = (v[idx_left, :] + v[idx_right, :] + v[:, idx_left] + v[:, idx_right] - 4 * v) / dx**2
    nabla_u_x = (u[idx_left, :] - u[idx_right, :]) / (2 * dx)
    nabla_u_y = (u[:, idx_left] - u[:, idx_right]) / (2 * dx)
    nabla_v_x = (v[idx_left, :] - v[idx_right, :]) / (2 * dx)
    nabla_v_y = (v[:, idx_left] - v[:, idx_right]) / (2 * dx)
    du = laplace_u - nabla_u_x * nabla_v_x - nabla_u_y * nabla_v_y  - u*laplace_v
    dv = laplace_v + u - v
    u_n = u + dt * du
    v_n = v + dt * dv
    return u_n, v_n


T =  1e-4
dt = 1e-9
steps = int(T / dt)
u = u_0
v = v_0
u_list = [u]
v_list = [v]
for i in range(steps):
    u, v = step(u, v)
    if i % 1000 ==0:
        u_list.append(u)
        v_list.append(v)
        print(f"time {i*dt}, max u: {u.max()}, max v: {v.max()}")

u = np.array(u_list)
v = np.array(v_list)
np.savez(f"./finite_difference_{N}_{N}.npz", u=u, v=v)
