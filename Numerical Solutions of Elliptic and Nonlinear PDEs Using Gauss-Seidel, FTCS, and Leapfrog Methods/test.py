import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100) # does not include zero
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
R1 = ((X-1)**2 + Y**2)**.5 # 1st charge located at x=+1, y=0
R2 = ((X+1)**2 + Y**2)**.5 # 2nd charge located at x=-1, y=0
V = 1./R1 - 1./R2 # two equal-and-opposite charges
Ey, Ex = np.gradient(-V, y, x) # careful about order
fig = plt.figure(figsize=(6, 3))
strm = plt.streamplot(X, Y, Ex, Ey, color=V, linewidth=2, cmap='autumn')
cbar = fig.colorbar(strm.lines)
cbar.set_label('Potential $V$')
plt.title('Electric field lines')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')
plt.tight_layout()
plt.show()