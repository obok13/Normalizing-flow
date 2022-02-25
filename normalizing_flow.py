import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

class masked_dense(nn.Module):

    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.register_buffer("mask", mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.linear.weight, self.linear.bias)

class MADE(nn.Module):
    def __init__(self, in_features, hidden_shape = [16,16], act = nn.Tanh()):
        super().__init__()

        self.act = act
        m_list = [] # m^l값 저장
        m_list.append((np.arange(in_features) + 1).reshape(-1,1))
        for i in range(len(hidden_shape)):
            m_l0 = m_list[-1]
            m_list.append(
                np.array([np.random.permutation(np.arange(np.min(m_l0) , in_features))[0] for j in range(hidden_shape[i])]).reshape(-1,1) # m^l값 할당하기
            )

        self.Mask_list = [] # mask matrix 저장
        for i in range(len(hidden_shape)) :
            m_l0 = m_list[i]
            m_l1 = m_list[i+1]
            self.Mask_list.append(torch.Tensor(1 - (m_l0.T > m_l1)))
        self.Mask_list.append(torch.Tensor(0 + (m_list[-1].T < m_list[0])))

        layers = []
        layers.append(masked_dense(in_features = in_features, out_features = hidden_shape[0], mask = self.Mask_list[0]))
        for i in range(1,len(hidden_shape)) :
            layers.append(masked_dense(in_features = hidden_shape[i-1], out_features = hidden_shape[i], mask = self.Mask_list[i]))
        layers.append(masked_dense(in_features = hidden_shape[-1], out_features = in_features, mask = self.Mask_list[-1]))

        self.layers = nn.ModuleList(layers)


    def forward(self, input):

        x = input 

        for layer in self.layers[:-1] :
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)

        return x


class MAF(nn.Module):
    def __init__(self, in_features, MADE_shape=[16,16], act = nn.Tanh(), n_of_MADE = 3, shuffle = True, bd = [0.1,2], device = 'cpu'):
        super().__init__()

        self.D = in_features
        self.L = n_of_MADE
        self.bd = bd
        self.shuffle_list = []
        self.shuffle_inv = []
        MADE_list_mu = []
        MADE_list_alp = []
        self.device = device

        for i in range(self.L):
            MADE_list_mu.append(MADE(in_features = in_features , hidden_shape = MADE_shape, act = act))
            MADE_list_alp.append(MADE(in_features = in_features , hidden_shape = MADE_shape, act = act))

            if shuffle:
                self.shuffle_list.append(list(np.random.permutation(np.arange(in_features))))
            else:
                self.shuffle_list.append(list(np.arange(in_features)))
            self.shuffle_inv.append(list(np.argsort(self.shuffle_list[i])))

        self.MADE_list_mu = nn.ModuleList(MADE_list_mu)
        self.MADE_list_alp = nn.ModuleList(MADE_list_alp)

    def forward(self, input): # u -> x

        x = input
        for i in range(self.L) :
            x = torch.cat([x[:,self.shuffle_list[i][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
            for j in range(self.D) :                
                mu_i = self.MADE_list_mu[i](x) # x[:,0:j]까지만 구해진 상태
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0] # blowup 방지, 너무 작아지는 것 방지

                x_ = mu_i[:,j:j+1] + sig_i[:,j:j+1] * x[:,j:j+1] # j번째 x만 계산
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)
        
        return x

    def reverse(self, input): # x -> u

        x = input
        for i in range(self.L) :
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
            x = (x - mu_i) / sig_i
            x = torch.cat([x[:,self.shuffle_inv[-i-1][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
        
        return x

    def log_u(self, input):
        return - .5 * torch.sum(input**2, 1, keepdim = True) - .5 * self.D * np.log(2 * np.pi)

    def log_x_pred(self, input) :

        x = input
        log_d = 0.
        for i in range(self.L) :
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
            log_d = log_d - torch.sum(torch.log(sig_i), 1, keepdim=True)
            x = (x - mu_i) / sig_i
            x = torch.cat([x[:,self.shuffle_inv[-i-1][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
            
        log_d = log_d + self.log_u(x)
        return log_d

    def log_u_pred(self , input, log_x) :

        x = input
        log_d = 0.
        for i in range(self.L) :
            x = torch.cat([x[:,self.shuffle_list[i][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
            for j in range(self.D) :                
                mu_i = self.MADE_list_mu[i](x)
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
                log_d = log_d + torch.log(sig_i[:,j:j+1])
                x_ = mu_i[:,j:j+1] + sig_i[:,j:j+1] * x[:,j:j+1] # j번째 x만 계산
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)

        log_d = log_d + log_x(x)
        return log_d

    def KL_x(self, input, log_x):

        x = self.forward(input)
        return torch.mean(self.log_x_pred(x) - log_x(x))

    def KL_u(self, input):

        x = self.reverse(input)
        return torch.mean(self.log_u_pred(x) - self.log_u(x))

    def loss(self, input, log_x=None, KL=False):
        if KL:
            if log_x is None:
                return self.KL_u(input)
            else:
                return self.KL_x(input, log_x)
        else:
            if log_x is None:
                return - self.log_x_pred(input).mean()
            else:
                return - self.log_u_pred(input, log_x).mean()

    def total_training(self, optimizer, train_data = None, M=100, log_x=None, KL=False, iteration = 100) :
        
        start = time.time()

        if train_data is None: # log_x가 주어지고 데이터는 없는 경우

            for i in range(iteration):
                train_data = torch.randn([M, self.D]).to(self.device)
                loss = self.loss(train_data, log_x, KL)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                    torch.save(self.state_dict(), f'nf_weights/maf')
        
        else:
            for i in range(iteration):
                loss = self.loss(train_data, log_x, KL)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                    torch.save(self.state_dict(), f'nf_weights/maf')


class IAF(nn.Module):
    def __init__(self, in_features, MADE_shape=[16,16], act = nn.Tanh(), n_of_MADE = 3, shuffle = True, bd = [0.1,2], device = 'cpu'):
        super().__init__()

        self.D = in_features
        self.L = n_of_MADE
        self.bd = bd
        self.shuffle_list = []
        self.shuffle_inv = []
        MADE_list_mu = []
        MADE_list_alp = []
        self.device = device

        for i in range(self.L):
            MADE_list_mu.append(MADE(in_features = in_features , hidden_shape = MADE_shape, act = act))
            MADE_list_alp.append(MADE(in_features = in_features , hidden_shape = MADE_shape, act = act))

            if shuffle:
                self.shuffle_list.append(list(np.random.permutation(np.arange(in_features))))
            else:
                self.shuffle_list.append(list(np.arange(in_features)))
            self.shuffle_inv.append(list(np.argsort(self.shuffle_list[i])))

        self.MADE_list_mu = nn.ModuleList(MADE_list_mu)
        self.MADE_list_alp = nn.ModuleList(MADE_list_alp)

    def forward(self, input): # u -> x

        x = input
        for i in range(self.L) :
            x = torch.cat([x[:,self.shuffle_list[-i-1][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
            x = mu_i + sig_i * x

        return x

    def reverse(self, input): # x -> u

        x = input

        for i in range(self.L) :
            for j in range(self.D) :
                mu_i = self.MADE_list_mu[i](x)
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
                x_ = (x[:,j:j+1] - mu_i[:,j:j+1]) / sig_i[:,j:j+1] # j번째 x만 계산
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)
            x = torch.cat([x[:,self.shuffle_inv[i][d]].view(-1,1) for d in range(self.D)], -1) # shuffle
            
        return x

    def log_u(self, input):
        return - .5 * torch.sum(input**2, 1, keepdim=True) - .5 * self.D * np.log(2 * np.pi)

    def log_x_pred(self, input):

        x = input
        log_d = 0.
        for i in range(self.L) :
            for j in range(self.D) :
                mu_i = self.MADE_list_mu[i](x)
                sig_i = self.MADE_list_alp[i](x)
                sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
                log_d = log_d - torch.log(sig_i[:,j:j+1])
                x_ = (x[:,j:j+1] - mu_i[:,j:j+1]) / sig_i[:,j:j+1] # j번째 x만 계산
                x = torch.cat([x[:,:j],x_,x[:,j+1:]], -1)
            x = torch.cat([x[:,self.shuffle_inv[i][d]].view(-1,1) for d in range(self.D)], -1) # shuffle

        log_d = log_d + self.log_u(x)
        return log_d

    def log_u_pred(self , input, log_x):

        x = input
        log_d = 0.
        for i in range(self.L) :
            x = torch.cat([x[:,self.shuffle_list[-i-1][d]].view(-1,1) for d  in range(self.D)] , axis = -1) # shuffle
            mu_i = self.MADE_list_mu[-i-1](x)
            sig_i = self.MADE_list_alp[-i-1](x)
            sig_i = torch.sigmoid(sig_i) * (self.bd[1] - self.bd[0]) + self.bd[0]
            log_d = log_d + torch.sum(torch.log(sig_i), 1, keepdim=True)
            x = mu_i + sig_i * x

        log_d = log_d + log_x(x)
        return log_d

    def KL_x(self, input, log_x):

        x = self.forward(input)
        return torch.mean(self.log_x_pred(x) - log_x(x))

    def KL_u(self, input):

        x = self.reverse(input)
        return torch.mean(self.log_u_pred(x) - torch.sum(self.log_u(x)))

    def loss(self, input, log_x=None, KL=False):
        if KL:
            if log_x is None:
                return self.KL_u(input)
            else:
                return self.KL_x(input, log_x)
        else:
            if log_x is None:
                return - self.log_x_pred(input).mean()
            else:
                return - self.log_u_pred(input, log_x).mean()

    def total_training(self, optimizer, train_data = None, M=100, log_x=None, KL=False, iteration = 100) :

        start = time.time()

        if train_data is None: # log_x가 주어지고 데이터는 없는 경우
            
            for i in range(iteration):
                train_data = torch.randn([M, self.D]).to(self.device)
                loss = self.loss(train_data, log_x, KL)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                    torch.save(self.state_dict(), f'nf_weights/iaf')
        
        else:
            for i in range(iteration):
                loss = self.loss(train_data, log_x, KL)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'train loss : {loss.item()}, Iteration : {i+1} / {iteration}, Time Lapsed : {time.time() - start}')
                    torch.save(self.state_dict(), f'nf_weights/iaf')