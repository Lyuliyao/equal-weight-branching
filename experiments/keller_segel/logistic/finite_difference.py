import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
y = np.linspace(0, 2 * np.pi, 100, endpoint=False)
# x= np.linspace(-0.5, 0.5, 100, endpoint=False)
# y= np.linspace(-0.5, 0.5, 100, endpoint=False)
X, Y = np.meshgrid(x, y)
# u_0 = np.random.rand(*X.shape) +1
# v_0 = np.random.rand(*X.shape) +2
# np.savez("initial_conditions.npz", u_0=u_0, v_0=v_0)
dX = X - 0.5*np.pi
dY = Y - 0.5*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
u_0 = np.exp(-dX**2 - dY**2)
dX = X - np.pi
dY = Y - np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
u_0 += np.exp(-dX**2 - dY**2)
dX = X - 1.5*np.pi
dY = Y - 1.5*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
u_0 += np.exp(-dX**2 - dY**2)


dX = X - 0.4*np.pi
dY = Y - 0.4*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
v_0 = np.exp(-dX**2 - dY**2)
dX = X - 0.8*np.pi
dY = Y - 0.8*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
v_0 += np.exp(-dX**2 - dY**2)
dX = X - 1.2*np.pi
dY = Y - 1.2*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
v_0 += np.exp(-dX**2 - dY**2)

dX = X - 1.6*np.pi
dY = Y - 1.6*np.pi
dX = np.where(dX < -np.pi, dX + 2 * np.pi, dX)
dY = np.where(dY < -np.pi, dY + 2 * np.pi, dY)
dX = np.where(dX > np.pi, dX - 2 * np.pi, dX)
dY = np.where(dY > np.pi, dY - 2 * np.pi, dY)
v_0 += np.exp(-dX**2 - dY**2)


dt =  1e-3
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
    du = laplace_u -  4*(1-u**2)/((1+u**2)**2)*nabla_u_x * nabla_v_x - 4*(1-u**2)/((1+u**2)**2)*nabla_u_y * nabla_v_y  - 4*(u/(1+u**2))*laplace_v + u*(1-u) 
    dv = laplace_v + u - v
    u_n = u + dt * du
    v_n = v + dt * dv
    return u_n, v_n

T = 1
dt = 1e-4
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