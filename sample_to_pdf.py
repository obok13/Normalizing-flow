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

# training data

z = torch.randn([3000,2]).to(device)
x1 = z[:,0:1]
x2 = z[:,1:2]*.5 + z[:,0:1]**2
train_data = torch.cat((x1,x2), -1).to(device)

# training

nn_maf = MAF(in_features = 2, MADE_shape = [16,16], act = torch.sin, n_of_MADE = 3, device = device).to(device)
optimizer = torch.optim.Adam(nn_maf.parameters(), lr=0.001)
nn_maf.total_training(optimizer = optimizer, train_data = train_data, KL=False, iteration = 10000)

# pdf

M = 101
X1 = np.linspace(-4,4,M, dtype=np.float64)
X2 = np.linspace(-2,10,M, dtype=np.float64)
X = torch.Tensor(np.array([point for point in itertools.product(X1,X2)])).to(device)
PDF_exact = -X[:,0:1]**2/2 - 2*(X[:,1:2] - X[:,0:1]**2)**2 - np.log(np.pi)
PDF_iaf = nn_maf.log_x_pred(X)

PDF_exact = PDF_exact.cpu().detach().numpy()
PDF_iaf = PDF_iaf.cpu().detach().numpy()
PDF_exact = np.exp(PDF_exact)
PDF_iaf = np.exp(PDF_iaf)

# plot

fig, axes = plt.subplots(1,2,figsize=(12, 6))
plt.rc('font', size=15)

im = axes.flat[0].imshow(PDF_iaf.reshape(M,M).T[::-1,:], extent=[-4,4,-2,10], cmap='Blues', vmin=PDF_exact.min(), vmax=PDF_exact.max())

im = axes.flat[1].imshow(PDF_exact.reshape(M,M).T[::-1,:], extent=[-4,4,-2,10], cmap='Blues', vmin=PDF_exact.min(), vmax=PDF_exact.max())

fig.colorbar(im, ax=axes.ravel().tolist())



#%%
