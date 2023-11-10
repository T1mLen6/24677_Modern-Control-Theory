import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

m = 1888.6
lr = 1.39
lf = 1.55
c_alpha = 20000
Iz = 25854
f = 0.019
delT = 0.032

velocity = 8#change to 5 and 8
xdot = velocity

A = np.array([[0,   1,                            0,                      0                          ],
              [0,   -4*c_alpha/(m*xdot),          4*c_alpha/m,            -2*c_alpha*(lf-lr)/(m*xdot)],
              [0,   0,                            0,                      1                          ],
              [0,   -2*c_alpha*(lf-lr)/(Iz*xdot), 2*c_alpha*(lf-lr)/(Iz), -2*c_alpha*(lf**2+lr**2)/(Iz*xdot)]])

B = np.array([[0,               0],
              [2*c_alpha/m,     0],
              [0,               0],
              [2*c_alpha*lf/Iz, 0]])

C = np.identity(4)

P = np.hstack((B,
              np.dot(A, B),
              np.dot(np.linalg.matrix_power(A, 2), B),
              np.dot(np.linalg.matrix_power(A, 3), B)))

Q = np.vstack((C,
              np.dot(C, A),
              np.dot(C, np.linalg.matrix_power(A, 2)),
              np.dot(C, np.linalg.matrix_power(A, 3))))


#Q1.1---------------------------------------------------------------------------------
rank_P = np.linalg.matrix_rank(P)
rank_Q = np.linalg.matrix_rank(Q)

print('For velocity = ', velocity, ", The controllability and observability of the system are:")
print('Rank of P is:', rank_P)
#print('Rank of Q is:', rank_P)
if rank_P != 4:
    print('The system is non-controllable')
else:
    print('The system is controllable')
    
print('Rank of Q is:', rank_Q)
if rank_Q != 4:
    print('The system is non-observable')
else:
    print('The system is observable')
    
#Q1.2---------------------------------------------------------------------------------

lon_v = np.linspace(1, 40, 80)
log_sig_arr = []
pole_arr = np.empty([4,1])
#print(pole_arr)
i = 0

#print(lon_v)
for xdot in lon_v:
    A = np.array([[0,   1,                            0,                      0                          ],
                  [0,   -4*c_alpha/(m*xdot),          4*c_alpha/m,            -2*c_alpha*(lf-lr)/(m*xdot)],
                  [0,   0,                            0,                      1                          ],
                  [0,   -2*c_alpha*(lf-lr)/(Iz*xdot), 2*c_alpha*(lf-lr)/(Iz), -2*c_alpha*(lf**2+lr**2)/(Iz*xdot)]])

    B = np.array([[0,               0],
                  [2*c_alpha/m,     0],
                  [0,               0],
                  [2*c_alpha*lf/Iz, 0]])

    C = np.identity(4)

    P = np.hstack((B,
                np.dot(A, B),
                np.dot(np.linalg.matrix_power(A, 2), B),
                np.dot(np.linalg.matrix_power(A, 3), B)))
    
    # logs-------------------------------------------------------------------------------
    
    [_, sig, _] = np.linalg.svd(P)
    sig1 = sig[0]
    sign = sig[-1]
    log_sig = np.log10(sig1/sign)
    #print('11111',log_sig)
    log_sig_arr.append(log_sig)
    #print('22222',log_sig_arr)
    
    # poles------------------------------------------------------------------------------
    
    eigs, _ = np.linalg.eig(A)
    #There are 4 eigen values
    #for j in range(4):
        #print(int(eigs.real[0]))
    pole_arr = np.hstack((pole_arr, eigs.real.reshape(-1,1)))
    
        
    i = i + 1
pole_arr = pole_arr[:,1:]
#print(pole_arr)

# plotting-----------------------------------
plt.figure(1)
plt.title('log10(alpha_1/alpha_n) vs v(m/s)')
plt.plot(lon_v, log_sig_arr)
plt.ylabel('log10(alpha_1/alpha_n)')
plt.xlabel('velocity (m/s)')
plt.grid()
plt.show()

plt.figure(2)
#plt.title('log10(alpha_1/alpha_n) vs v(m/s)')
plt.subplot(2,2,1)
plt.plot(lon_v, pole_arr[0,:])
plt.ylabel('Re_p1')
plt.xlabel('velocity (m/s)')
plt.grid()

plt.subplot(2,2,2)
plt.plot(lon_v, pole_arr[1,:])
plt.ylabel('Re_p2')
plt.xlabel('velocity (m/s)')
plt.grid()

plt.subplot(2,2,3)
plt.plot(lon_v, pole_arr[2,:])
plt.ylabel('Re_p3')
plt.xlabel('velocity (m/s)')
plt.grid()

plt.subplot(2,2,4)
plt.plot(lon_v, pole_arr[3,:])
plt.ylabel('Re_p4')
plt.xlabel('velocity (m/s)')
plt.grid()

plt.tight_layout()
plt.show()
    
    
    
    
    

