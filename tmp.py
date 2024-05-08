import numpy as np 
import torch
import scipy.io

# load npy
# dir = '/home/zzzichen/GraphPDE/data/data/coarse_valid/0.npy'
# dict = np.load(dir, allow_pickle=True).item()
dict = scipy.io.loadmat('/scratch/jjparkcv_root/jjparkcv1/shared_data/diffusionPDE/NS/ns/ns_1000_128_128_10_45.mat')

print(dict.keys())
print()

# print the shapes of each component
for key in dict.keys():
    value = dict[key]
    if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        print(f"Shape of {key}: {value.shape}")
    else:
        print(f"Value of {key}: {value}")
    

# # print(dict['gt_observation'])

# # save observation 
# observation_name = dir.replace('dict_0.npy', 'observation.txt')
# np.savetxt(observation_name, dict['gt_observation'].cpu())
