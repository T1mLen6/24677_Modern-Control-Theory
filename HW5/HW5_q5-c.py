import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the orginal system of differential equations
def system_org(t, y):
    x1, x2 = y
    x1_dot = x2 - x1 * x2**2
    x2_dot = -x1**3
    return [x1_dot, x2_dot]

# Define the linearized system of differential equations
def system_lin(t, y):
    x1, x2 = y
    x1_dot = x2
    x2_dot = 0
    return [x1_dot, x2_dot]

x1 = np.linspace(-3, 3, 20)
x2 = np.linspace(-3, 3, 20)
X1, X2 = np.meshgrid(x1, x2)

X1_dot, X2_dot = np.zeros_like(X1), np.zeros_like(X2)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x1_val, x2_val = X1[i, j], X2[i, j]
        ####################################################
        ### Change the func name for org or lin system #####
        derivatives = system_org(0, [x1_val, x2_val])
        ####################################################
        
        X1_dot[i, j] = derivatives[0]
        X2_dot[i, j] = derivatives[1]

plt.figure()
plt.quiver(X1, X2, X1_dot, X2_dot, color='r')
plt.xlabel('x1')
plt.ylabel('x2')
#plt.title('Phase Portrait for the Original System')
plt.title('Phase Portrait for the Linearized System')
plt.grid()
plt.show()