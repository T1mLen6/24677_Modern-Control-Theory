
import numpy as np
from scipy.signal import StateSpace, lsim
import matplotlib.pyplot as plt


m = 1 # kg
gam = 3
alpha = 1.2
sig = 0.1

G = np.asarray([[1., 0.2, 0.1],
                [0.1, 2., 0.1],
                [0.3, 0.1, 3.]])

A = alpha*gam*np.asarray([[0.,            G[0,1]/G[0,0], G[0,2]/G[0,0]],
                          [G[1,0]/G[1,1], 0.,            G[1,2]/G[1,1]],
                          [G[2,0]/G[2,2], G[2,1]/G[2,2], 0.          ]])

condition = 1 # change this to 2 if using second conditon


p = np.asarray([[0.1],
                [0.1],
                [0.1]]) 

if condition == 2:
    p = np.asarray([[0.1],
                    [0.01],
                    [0.02]]) 

B = alpha*gam*np.asarray([[1/G[0,0]],
                          [1/G[1,1]],
                          [1/G[2,2]]]) 

sigma_sq = np.square(np.asarray([[sig]]))
C = np.asarray([[0.]])
D = np.asarray([[0.]])

# define the continouse time linear system
cart_sys = StateSpace(A, B, C, D)

# define simulation steps in time
t = np.arange(0, 10, 1e-3)
# define control input
F = t**2
# simulate the system
_, y, x = lsim(cart_sys, F, t, X0=[0., 0.])
# plot
plt.figure(dpi=100)
plt.plot(t, y)
plt.ylabel('p [m]')
plt.xlabel('t [s]')
plt.show()

