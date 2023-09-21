import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

A = np.array([[0., 1.], 
             [-2., -2.]])

B = np.array([[1.],[1.]])

C = np.array([2., 3.])
D = np.array([0.])


#y(5), CT system
t = np.linspace(0, 5)
u = np.ones(len(t))


CT_system = signal.StateSpace(A,B,C,D)
t_ct, y_ct, x_ct = signal.lsim(CT_system,u,t)


#y(5), DT system
Ad = np.array([[ 0.508,  0.309], 
               [-0.619, -0.111]])

Bd = np.array([[1.047],[-0.1821]])

y_dt = []
sum = 0

for k in range(6):
    for m in range(k):
        sum += np.dot(np.dot(C, np.linalg.matrix_power(Ad, k-m-1)), Bd)
    y_dt = np.append(y_dt, sum)
    sum = 0
    
print(y_dt)  
fig, ax = plt.subplots()
#plt.figure(dpi=100)
ax.plot(t_ct, y_ct, label = 'CT')
ax.plot(range(6), y_dt, label = 'DT')

ax.set_title('CT vs. DT')
ax.set_xlabel('Time step (s)')
ax.set_ylabel('System Response')
ax.grid()
ax.legend()

plt.show()

    




