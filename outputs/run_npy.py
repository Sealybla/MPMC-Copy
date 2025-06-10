#prints points

import numpy as np
import matplotlib.pyplot as plt 
#change path 
data = np.load(r'D:\work\Research\IITSURE\MPMC\outputs\dim_2\nsamples_64\nhid_64\LfL2dis.npy')

plt.scatter(data[0,:,0], data[0,:,1], color = 'blue') 
plt.title("2D Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()