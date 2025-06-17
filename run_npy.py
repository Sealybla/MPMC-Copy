#prints points
import numpy as np
import matplotlib.pyplot as plt 
import torch 
from dispersion import dispersion
from pathlib import Path
from qmcpy import Sobol


#change path 
xs = torch.rand(64, 2)
data = np.load(r'D:\work\Research\IITSURE\coding\MPMC\outputs\dim_2\nsamples_16\nhid_64\LfL2dis.npy')

# plt.scatter(data[0,:,0], data[0,:,1], color = 'blue') 
# plt.scatter(xs[:,0], xs[:,1], color = 'blue') 
# plt.title("2D Points")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.axis('equal')
# plt.show()

#calculate dispersion:
 
# losses = ['LfL2ext']
# configs = [[16, 32], [32, 32], [64,32], [128,64]]
# for loss in losses:
#     for nsamples, nhid in configs:
#         path = fr'D:\work\Research\IITSURE\coding\MPMC_Copy\outputs\dim_2\nsamples_{nsamples}\nhid_{nhid}\{loss}.npy'
#         data = np.load(path)
#         Path('dispersion/data').mkdir(parents=True, exist_ok=True)
#         Path('dispersion/pics').mkdir(parents=True, exist_ok=True)
#         pic_path = f'dispersion/pics/{loss}/plot_nsamples{nsamples}.png'
#         max_vol, best_box = dispersion(data.squeeze(), save_path = pic_path)
        
#         f = open('dispersion/data/' + loss + '.txt', 'a')
#         f.write(f"nsamples, nhid: ({nsamples},{nhid})\n")
#         f.write(f"Dispersion: {max_vol}\n")
#         f.write(f"Box: {best_box}\n\n")

#         f.close()


#calculate l_extreme for random points (16):
# print(Linfextr(torch.from_numpy(data)))

#Sobol
points = [32, 64, 128]
sobol_gen = Sobol(dimension = 2, randomize = None)
for n in points:
    data = sobol_gen.gen_samples(n)
    pic_path = f'dispersion/Sobol/pics/plot_nsamples{n}.png'
    max_vol, best_box = dispersion(data.squeeze(), save_path = pic_path)
    f = open('dispersion/Sobol/data.txt', 'a')
    f.write(f"nsamples: ({n})\n")
    f.write(f"Dispersion: {max_vol}\n")
    f.write(f"Box: {best_box}\n\n")
    f.close()

