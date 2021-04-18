#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

np.random.seed(42)

fig,ax=plt.subplots(subplot_kw={'projection':'3d'})

#Make data
#X=np.arange(-5,5,0.25)
#Y=np.arange(-5,5,0.25)
X=np.array([0,0.05,0.1,0.15,0.2])
Y=np.array([0,45,90])
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)
Z=np.random.normal(-1,1,X.shape)
print ('Z:',Z)


#plot the surface
surf=ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)

#customize Z
ax.set_zlim(-3,3)
ax.zaxis.set_major_locator(LinearLocator(10))
#A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:0.2f}')

#Add a colorbar that maps values into colors
fig.colorbar(surf,shrink=0.5,aspect=5)

plt.show()
