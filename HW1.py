
import numpy as np
from scipy.signal import StateSpace, lsim
import matplotlib.pyplot as plt


m = 1 # kg
gam = 5 # or gamma = 5 as question asked
alpha = 1.2
sig = 0.1

G = np.asarray([[1., 0.2, 0.1],
                [0.1, 2., 0.1],
                [0.3, 0.1, 3.]])

A = alpha*gam*np.asarray([[0.,            G[0,1]/G[0,0], G[0,2]/G[0,0]],
                          [G[1,0]/G[1,1], 0.,            G[1,2]/G[1,1]],
                          [G[2,0]/G[2,2], G[2,1]/G[2,2], 0.          ]])

condition = 2 # change this to 2 if using second conditon


p = np.asarray([[0.1],
                [0.1],
                [0.1]]) 


if condition == 2:
    p = np.asarray([[0.1],
                    [0.01],
                    [0.02]]) 
initial_p = p

B = alpha*gam*np.asarray([[1/G[0,0]],
                          [1/G[1,1]],
                          [1/G[2,2]]]) 

si = np.array([G[0,0]*p[0], G[1,1]*p[1], G[2,2]*p[2]])
qi = np.array([sig**2 + G[0,1]*p[1] + G[0,2]*p[2], sig**2 + G[1,0]*p[0] + G[1,2]*p[2], sig**2 + G[2,0]*p[0] + G[2,1]*p[1]])
Si = si/qi


p_array = p
s_array = Si
t_array = []
target_val = np.asarray([alpha * gam])


for step in range(25):
    p = np.dot(A, p) + B * sig**2
    p_array = np.hstack((p_array, p))
    si = np.array([G[0,0]*p[0], G[1,1]*p[1], G[2,2]*p[2]])
    qi = np.array([sig**2 + G[0,1]*p[1] + G[0,2]*p[2], sig**2 + G[1,0]*p[0] + G[1,2]*p[2], sig**2 + G[2,0]*p[0] + G[2,1]*p[1]])
    Si = si/qi
    s_array = np.hstack((s_array, Si))
    t_array = np.append(t_array, step)
    target_val = np.append(target_val, alpha * gam)

t_array = np.append(t_array, t_array[-1]+1)


# plot
fig, axs = plt.subplots(2)
#plt.figure(dpi=100)
axs[0].plot(t_array, p_array[0,:], label = "p1")
axs[0].plot(t_array, p_array[1,:], label = "p2")
axs[0].plot(t_array, p_array[2,:], label = "p3")
axs[0].set_title('Transmitter Power vs. Time (p = '+str(initial_p.reshape(1,3)) + ", gamma = " + str(gam) + ")")
axs[0].set_xlabel('Time step (s)')
axs[0].set_ylabel('Transmitter Power')
axs[0].grid()
axs[0].legend(loc='upper right')

axs[1].plot(t_array, s_array[0,:], label = "s1")
axs[1].plot(t_array, s_array[1,:], label = "s2")
axs[1].plot(t_array, s_array[2,:], label = "s3")
axs[1].plot(t_array, target_val, label = "target", color = "red", linestyle = '--')
axs[1].set_title('SNIR vs. Time (p = '+str(initial_p.reshape(1,3)) + ", gamma = " + str(gam) + ")")
axs[1].set_xlabel('Time step (s)')
axs[1].set_ylabel('SNIR')
axs[1].grid()
axs[1].legend(loc='upper right')

fig.tight_layout()

#plt.legend()
plt.savefig("p = "+str(initial_p.reshape(1,3)) + ", gamma = " + str(gam) + ".png", dpi = 600)
plt.show()

