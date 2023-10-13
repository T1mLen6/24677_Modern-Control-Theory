import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the system of differential equations
def system_V(t, y):
    x1, x2 = y
    v_dot = -4 * (x1**4) * (x2**2)  
    return [v_dot]


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

V_dot = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x1_val, x2_val = X1[i, j], X2[i, j]
        ####################################################
        ### #####
        v_dot = system_V(0, [x1_val, x2_val])
        ####################################################
        
        V_dot[i, j] = int(v_dot[0])

#print(np.shape(X1),np.shape(V_dot)) 
#print(V_dot)       
# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, V_dot, cmap='viridis')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Variation of V_dot')

plt.show()