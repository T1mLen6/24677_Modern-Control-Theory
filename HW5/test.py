import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# Defining a range of inputs x1 and x2
X1=np.arange(-100,100)
X2=np.arange(-100,100)
# Computing a meshgrid from the 2 inputs
x1,x2=np.meshgrid(X1,X2)
# Computing V_dot=-4*(x1^4)*(x2^2)
v=-4*np.power(x1,4)*np.power(x2,2)
# Plottting v,x1,x2
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x1,x2,v,cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()