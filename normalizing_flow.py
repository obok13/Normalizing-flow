import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

class masked_dense(nn.Module):

    def __init__(self, in_dim, out_dim, mask, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.register_buffer("mask", mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.linear.weight, self.linear.bias)

class MADE(nn.Module):
    def __init__(self, in_dim, hidden_shape = [16,16], act = nn.Tanh()):
        super().__init__()

        self.act = act
        m_list = [] # save m^l
        m_list.append((np.arange(in_dim) + 1).reshape(-1,1))
        for i in range(len(hidden_shape)):
            m_l0 = m_list[-1]
            m_list.append(
                np.array([np.random.permutation(np.arange(np.min(m_l0) , in_dim))[0] for j in range(hidden_shape[i])]).reshape(-1,1) # allocate m^l
            )

        self.Mask_list = [] # save mask matrix
        for i in range(len(hidden_shape)) :
            m_l0 = m_list[i]
            m_l1 = m_list[i+1]
            self.Mask_list.append(torch.Tensor(1 - (m_l0.T > m_l1)))
        self.Mask_list.append(torch.Tensor(0 + (m_list[-1].T < m_list[0])))

        layers = []
        layers.append(masked_dense(in_dim = in_dim, out_dim = hidden_shape[0], mask = self.Mask_list[0]))
        for i in range(1,len(hidden_shape)) :
            layers.append(masked_dense(in_dim = hidden_shape[i-1], out_dim = hidden_shape[i], mask = self.Mask_list[i]))
        layers.append(masked_dense(in_dim = hidden_shape[-1], out_dim = in_dim, mask = self.Mask_list[-1]))

        self.layers = nn.ModuleList(layers)


    def forward(self, input):

        x = input 

        for layer in self.layers[:-1] :
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)

        return x


class MAF(nn.Module):
    def __init__(self, in_dim, made_hidden_shape=[16,16], made_act = nn.Tanh(), n_made = 3, shuffle = True, bd_sig = [0.1,2], device = 'cpu'):
        super().__init__()

        self.in_dim = in_dim
        self.n_made = n_made
        self.bd_sig = bd_sig
        self.shuffle_list = []
        self.shuffle_inv = []
        MADE_list_mu = []
        MADE_list_alp = []
        self.device = device

        for i in range(self.n_made):
            MADE_list_mu.append(MADE(in_dim = in_dim, hidden_shape = made_hidden_shape, act = made_act))
            MADE_list_alp.append(MADE(in_dim = in_dim, hidden_shape = made_hidden_shape, act = made_act))

            if shuffle:
                self.shuffle_list.append(list(np.random.permutation(np.arange(in_dim))))
            else:
                self.shuffle_list.append(list(np.arange(in_dim)))
            self.shuffle_inv.append(list(np.argsort(self.shuffle_list[i])))

        self.MADE_list_mu = nn.ModuleList(MADE_list_mu)
        self.MADE_list_alp = nn.ModuleList(MADE_list_alp)

    def forward(self, u_input): # u -> x

        x = u_input
        for i in range(self.n_made) :
            x = torch.cat([x[:,self.shuffle_list[i][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
            for j in range(self.in_dim) :                
                mu_i = self.MADE_list_mu[i](x) # x[:,0:j] has been computed
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0] # prevent blowup, being close to 0

                x_ = mu_i[:,j:j+1] + sig_i[:,j:j+1] * x[:,j:j+1] # update jth component
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)
        
        return x

    def reverse(self, x_input): # x -> u

        u = x_input
        for i in range(self.n_made) :
            mu_i = self.MADE_list_mu[-i-1](u)
            sig_i = self.MADE_list_alp[-i-1](u)
            sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
            u = (u - mu_i) / sig_i
            u = torch.cat([u[:,self.shuffle_inv[-i-1][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
        
        return u

    def log_u(self, u_input):
        return - .5 * torch.sum(u_input**2, 1, keepdim = True) - .5 * self.in_dim * np.log(2 * np.pi)

    def log_x_pred(self, x_input):

        u = x_input
        log_d = 0.
        for i in range(self.n_made) :
            mu_i = self.MADE_list_mu[-i-1](u)
            sig_i = self.MADE_list_alp[-i-1](u)
            sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
            log_d = log_d - torch.sum(torch.log(sig_i), 1, keepdim=True)
            u = (u - mu_i) / sig_i
            u = torch.cat([u[:,self.shuffle_inv[-i-1][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
            
        log_d = log_d + self.log_u(u)
        return log_d

    def log_u_pred(self, u_input, log_x) :

        x = u_input
        log_d = 0.
        for i in range(self.n_made) :
            x = torch.cat([x[:,self.shuffle_list[i][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
            for j in range(self.in_dim) :                
                mu_i = self.MADE_list_mu[i](x)
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
                log_d = log_d + torch.log(sig_i[:,j:j+1])
                x_ = mu_i[:,j:j+1] + sig_i[:,j:j+1] * x[:,j:j+1] # update jth component
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)

        log_d = log_d + log_x(x)
        return log_d

    def KL_x(self, u_input, log_x):

        x = self.forward(u_input)
        return torch.mean(self.log_x_pred(x) - log_x(x))

    def loss(self, train_data, log_x=None, KL=False):
        if log_x is None: # x sample given, x pdf not given -> train_data is x
            if KL:
                raise NotImplementedError()
            else:
                return - self.log_x_pred(train_data).mean()
        else: # x sample not given, x pdf given -> train_data is u
            if KL:
                return self.KL_x(train_data, log_x)
            else:
                return - self.log_u_pred(train_data, log_x).mean()

    def total_training(self, optimizer, train_data = None, M=100, log_x=None, KL=False, iteration = 100) :
        
        start = time.time()
            
        for i in range(iteration):
            if log_x is not None: # x sample not given, x pdf given -> train_data is u
                train_data = torch.randn([M, self.in_dim]).to(self.device)
            loss = self.loss(train_data, log_x, KL)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                torch.save(self.state_dict(), f'nf_weights/maf')


class IAF(nn.Module):
    def __init__(self, in_dim, made_hidden_shape=[16,16], made_act = nn.Tanh(), n_made = 3, shuffle = True, bd_sig = [0.1,2], device = 'cpu'):
        super().__init__()

        self.in_dim = in_dim
        self.n_made = n_made
        self.bd_sig = bd_sig
        self.shuffle_list = []
        self.shuffle_inv = []
        MADE_list_mu = []
        MADE_list_alp = []
        self.device = device

        for i in range(self.n_made):
            MADE_list_mu.append(MADE(in_dim = in_dim , hidden_shape = made_hidden_shape, act = made_act))
            MADE_list_alp.append(MADE(in_dim = in_dim , hidden_shape = made_hidden_shape, act = made_act))

            if shuffle:
                self.shuffle_list.append(list(np.random.permutation(np.arange(in_dim))))
            else:
                self.shuffle_list.append(list(np.arange(in_dim)))
            self.shuffle_inv.append(list(np.argsort(self.shuffle_list[i])))

        self.MADE_list_mu = nn.ModuleList(MADE_list_mu)
        self.MADE_list_alp = nn.ModuleList(MADE_list_alp)

    def forward(self, u_input): # u -> x

        x = u_input
        for i in range(self.n_made) :
            x = torch.cat([x[:,self.shuffle_list[-i-1][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
            x = mu_i + sig_i * x

        return x

    def reverse(self, x_input): # x -> u

        u = x_input

        for i in range(self.n_made) :
            for j in range(self.in_dim) :
                mu_i = self.MADE_list_mu[i](u)
                sig_i = self.MADE_list_alp[i](u)
                sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
                u_ = (u[:,j:j+1] - mu_i[:,j:j+1]) / sig_i[:,j:j+1] # update jth component
                u = torch.cat([u[:,:j],u_,u[:,j+1:]], -1)
            u = torch.cat([u[:,self.shuffle_inv[i][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle
            
        return u

    def log_u(self, u_input):
        return - .5 * torch.sum(u_input**2, 1, keepdim=True) - .5 * self.in_dim * np.log(2 * np.pi)

    def log_x_pred(self, x_input):

        u = x_input
        log_d = 0.
        for i in range(self.n_made) :
            for j in range(self.in_dim) :
                mu_i = self.MADE_list_mu[i](u)
                sig_i = self.MADE_list_alp[i](u)
                sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
                log_d = log_d - torch.log(sig_i[:,j:j+1])
                u_ = (u[:,j:j+1] - mu_i[:,j:j+1]) / sig_i[:,j:j+1] # update jth component
                u = torch.cat([u[:,:j],u_,u[:,j+1:]], -1)
            u = torch.cat([u[:,self.shuffle_inv[i][d]].view(-1,1) for d in range(self.in_dim)], -1) # shuffle

        log_d = log_d + self.log_u(u)
        return log_d

    def log_u_pred(self , u_input, log_x):

        x = u_input
        log_d = 0.
        for i in range(self.n_made) :
            x = torch.cat([x[:,self.shuffle_list[-i-1][d]].view(-1,1) for d  in range(self.in_dim)] , axis = -1) # shuffle
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd_sig[1] - self.bd_sig[0]) + self.bd_sig[0]
            log_d = log_d + torch.sum(torch.log(sig_i), 1, keepdim=True)
            x = mu_i + sig_i * x

        log_d = log_d + log_x(x)
        return log_d

    def KL_x(self, u_input, log_x):

        x = self.forward(u_input)
        return torch.mean(self.log_x_pred(x) - log_x(x))

    def loss(self, train_data, log_x=None, KL=False):
        if log_x is None: # x sample given, x pdf not given -> train_data is x
            if KL:
                raise NotImplementedError()
            else:
                return - self.log_x_pred(train_data).mean()
        else: # x sample not given, x pdf given -> train_data is u
            if KL:
                return self.KL_x(train_data, log_x)
            else:
                return - self.log_u_pred(train_data, log_x).mean()

    def total_training(self, optimizer, train_data = None, M=100, log_x=None, KL=False, iteration = 100) :
        
        start = time.time()
    
        for i in range(iteration):
            if log_x is not None: # x sample not given, x pdf given -> train_data is u
                train_data = torch.randn([M, self.in_dim]).to(self.device)
            loss = self.loss(train_data, log_x, KL)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                torch.save(self.state_dict(), f'nf_weights/iaf')