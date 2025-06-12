#prints points

import numpy as np
import matplotlib.pyplot as plt 
import torch 
#change path 
xs = torch.rand(64, 2)
# data = np.load(r'D:\work\Research\IITSURE\coding\MPMC\outputs\dim_2\nsamples_64\nhid_64\LfL2per.npy')

# plt.scatter(data[0,:,0], data[0,:,1], color = 'blue') 
plt.scatter(xs[:,0], xs[:,1], color = 'blue') 
plt.title("2D Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.show()