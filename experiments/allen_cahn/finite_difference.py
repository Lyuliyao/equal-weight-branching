import numpy as np



x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
# x= np.linspace(-0.5, 0.5, 100, endpoint=False)
# y= np.linspace(-0.5, 0.5, 100, endpoint=False)
X, Y = np.meshgrid(x, y)
u_0 = np.sin(X)**2*np.cos(Y)**2
# u_0 = 840*np.exp(-84*(X**2+Y**2)) #np.sin(X)**2*np.cos(Y)**2
# v_0 = 420*np.exp(-42*(X**2+Y**2))#np.cos(X)+np.cos(Y)+2
dx =  x[1] - x[0]
idx = np.arange(u_0.shape[0])
idx_left = np.roll(idx, -1)
idx_right = np.roll(idx, 1)
idx_left
u  = u_0

def step(u):
    laplace_u = (u[idx_left, :] + u[idx_right, :] + u[:, idx_left] + u[:, idx_right] - 4 * u) / dx**2
    du = 0.01 * laplace_u + u - u**3 
    u_n = u + dt * du
    return u_n


T =  2
dt = 1e-3
steps = int(T / dt)
u = u_0
u_list = []
for i in range(steps):
    u = step(u)
    u_list.append(u)
    
    
u_list = np.array(u_list)
np.savez("./finite_difference.npz", u=u_list)