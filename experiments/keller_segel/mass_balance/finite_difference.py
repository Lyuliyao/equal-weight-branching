import numpy as np



x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
# x= np.linspace(-0.5, 0.5, 100, endpoint=False)
# y= np.linspace(-0.5, 0.5, 100, endpoint=False)
X, Y = np.meshgrid(x, y)
u_0 = np.sin(X)**2*np.cos(Y)**2
v_0 = np.cos(X) + np.cos(Y) + 2
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


T =  2e-1
dt = 1e-3
steps = int(T / dt)
u = u_0
v = v_0
u_list = []
v_list = []
for i in range(steps):
    u, v = step(u, v)
    u_list.append(u)
    v_list.append(v)
    
    
u_list = np.array(u_list)
v_list = np.array(v_list)
np.savez("./finite_difference.npz", u=u_list, v=v_list)