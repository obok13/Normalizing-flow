#%%

# library

import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from normalizing_flow import *

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# target distribution

def log_target(x):

    return -x[:,0:1]**2/2 - 2*(x[:,1:2] - x[:,0:1]**2)**2 - np.log(np.pi)

# training

nn_iaf = IAF(in_features = 2, MADE_shape = [16,16], act = torch.sin, n_of_MADE = 3, device = device).to(device)
optimizer = torch.optim.Adam(nn_iaf.parameters(), lr=0.001)
nn_iaf.total_training(optimizer = optimizer, log_x = log_target, KL=False, iteration = 10000)

# sample

z_exact = torch.randn([3000,2]).to(device)
x_iaf = nn_iaf.forward(z_exact)
x1 = z_exact[:,0:1]
x2 = z_exact[:,1:2]*.5 + z_exact[:,0:1]**2
x_exact = torch.cat((x1,x2), -1).to(device)
z_iaf = nn_iaf.reverse(x_exact)

z_exact = z_exact.cpu().detach().numpy()
x_exact = x_exact.cpu().detach().numpy()
x_iaf = x_iaf.cpu().detach().numpy()
z_iaf = z_iaf.cpu().detach().numpy()

# plot

plt.figure(figsize = (12,8))

plt.subplot(2,2,1)
plt.plot(x_iaf[:,0], x_iaf[:,1], 'b.' , label = "x_iaf" , alpha = .5)
plt.xlim([-4,4])
plt.ylim([-2,10])
plt.legend()

plt.subplot(2,2,2)
plt.plot(x_exact[:,0], x_exact[:,1], 'b.' , label = "x_exact" , alpha = .5)
plt.xlim([-4,4])
plt.ylim([-2,10])
plt.legend()

plt.subplot(2,2,3)
plt.plot(z_iaf[:,0], z_iaf[:,1], 'b.' , label = "z_iaf" , alpha = .5)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()

plt.subplot(2,2,4)
plt.plot(z_exact[:,0], z_exact[:,1], 'b.' , label = 'z_exact' , alpha = .5)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()

plt.show()

# %%
