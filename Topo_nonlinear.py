import time
import torch.optim.lr_scheduler

import torch
import torch.nn as nn
import numpy as np
from copy import copy
import scipy.linalg as slin
import torch.nn.functional as F
from Topo_utils import create_Z,create_new_topo,threshold_W,find_idx_set_updated
import utils

def conditional_print(*args, **kwargs):
        """Prints only if PRINT_ENABLED is True."""
        if PRINT_ENABLED:
            print(*args, **kwargs)



class TopoMLP(nn.Module):

    def __init__(self, dims):
        super(TopoMLP, self).__init__()
        assert len(dims) >= 2 and dims[-1] == 1, "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.d1 = dims[1]
        self.dims = dims
        self.depth = len(dims)
        self.layerlist = nn.ModuleList()
        self._create_layerlist()

    def _create_layerlist(self):

        # Initialize the first layer separately due to its unique structure
        layer0 = nn.ModuleList([nn.Linear(1, self.d1, bias=False) for _ in range(self.d * self.d)])
        self.layerlist.append(layer0)
        # For subsequent layers, use a more streamlined approach
        for i in range(1, self.depth - 1):
            layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1], bias=False) for _ in range(self.d)])
            self.layerlist.append(layers)


    def set_all_para_grad_True(self):
        for param in self.parameters():
            param.requires_grad = True

    # set zero entry and set gradient non-updated for layer0!!!
    def reset_by_topo(self, topo):
        self.set_all_para_grad_True()
        Z = create_Z(topo)
        edge_abs_idx = np.argwhere(Z)
        with torch.no_grad():
            for idx in edge_abs_idx:
                linear_idx = int(idx[0] + self.d * idx[1])
                self.layerlist[0][linear_idx].weight.fill_(0)
                self.layerlist[0][linear_idx].weight.requires_grad = False



    def _forward_i(self, x, ith):
        # Improved forward pass to reduce complexity
        
        layer0_weights = torch.cat([self.layerlist[0][ll].weight for ll in range(self.d * ith, self.d * (ith + 1))],
                                       dim=1).T
        x = torch.mm(x, layer0_weights)
        for ii in range(1, self.depth - 1):
            x = F.sigmoid(x)  # Consider using F.relu(x) for ReLU activation
            x = self.layerlist[ii][ith](x)
        return x

    #

    def forward(self, x):  # [n,d] ->[n,d]
        output = [self._forward_i(x, ii) for ii in range(self.d)]
        return torch.cat(output, dim=1)


    @torch.no_grad()
    def freeze_grad_f_i(self,ith):
        # freeze all the gradient of all the parameters related to f_i
        for k in range(self.d):
            self.layerlist[0][int(k+self.d*ith)].weight.requires_grad = False
        for i in range(1, self.depth - 1):
            self.layerlist[i][ith].weight.requires_grad = False


    def update_nn_by_topo(self, topo, index):
        # update the zero constraint and freeze corresponding gradient update
        i, j = index
        wherei, wherej = topo.index(i), topo.index(j)
        topo0 = create_new_topo(copy(topo), index, opt=1)

        self.reset_by_topo(topo = topo0)
        freeze_idx = [oo for oo in range(self.d) if oo not in topo0[wherej:(wherei + 1)]]
        if freeze_idx:
            for ith in freeze_idx:
                self.freeze_grad_f_i(ith)





    def layer0_l1_reg(self):
        return sum(torch.sum(torch.abs(vec.weight)) for vec in self.layerlist[0])

    def l2_reg(self):
        return sum(torch.sum(vec.weight ** 2) for layer in self.layerlist for vec in layer)



    @torch.no_grad()
    def get_gradient_F(self):

        G_grad = torch.zeros(self.d ** 2, device='cpu')
        for count, vec in enumerate(self.layer0):
            G_grad[count] = torch.norm(vec.weight.grad, p=2)
        G_grad = torch.reshape(G_grad, (self.d, self.d)).t()
        return G_grad.numpy()

    @torch.no_grad()
    def layer0_to_adj(self):


        W = torch.zeros((self.d * self.d), device='cpu')
        for count, vec in enumerate(self.layerlist[0]):
            # W[count] = torch.sqrt(torch.sum(vec.weight ** 2))
            W[count] = torch.norm(vec.weight.data, p=2)
        W = torch.reshape(W, (self.d, self.d)).t()

        return W.numpy()



    def _h(self, W):
        I = np.eye(self.d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T

        return h, G_h


class TOPO_Nonlinear:

    def __init__(self,
            model,
            X,
             lambda1 = 0.01,
             lambda2 = 0.01,
             learning_rate = None,
             num_iter = 50,
             opti = 'LBFGS',
             loss_type = 'l2',
             lr_decay = False,
             tol = 0.01,
           dtype: torch.dtype = torch.double ):
        self.model = model
        self.X = X
        self.n, self.d = X.shape
        assert opti in ['LBFGS', 'Adam'], "Invalid optimizer"
        if learning_rate == None:
            if opti == 'LBFGS':
                self.learning_rate_lbfgs = 1
            else:
                self.learning_rate_adam = 0.01
        else:
            self.learning_rate_lbfgs = learning_rate
            self.learning_rate_adam = learning_rate
        self.lambda1 = torch.tensor(lambda1)
        self.lambda2 = torch.tensor(lambda2)
        self.num_iter = num_iter
        self.opti = opti
        self.X_torch = torch.from_numpy(X)
        self.loss_type = loss_type
        self.lr_decay = lr_decay
        self.tol = tol

    def squared_loss(self, output, target):
        return 0.5 / self.n * torch.sum((output - target) ** 2)

    def log_loss(self, output, target):
        return  0.5 * self.d * torch.log(1 / self.n * torch.sum((output - target) ** 2))
    
    def train_iter(self, model, optimizer):
        def closure():
            optimizer.zero_grad()
            X_hat = model(self.X_torch)
            if self.loss_type == 'l2':
                loss = self.squared_loss(X_hat, self.X_torch)
            elif self.loss_type == 'logl2':
                loss = self.log_loss(X_hat, self.X_torch)
            l2_reg = 0.5 * self.lambda2 * model.l2_reg()
            l1_reg = self.lambda1 * model.layer0_l1_reg()
            total_loss = loss + l2_reg + l1_reg
            total_loss.backward()
            return total_loss


        loss = optimizer.step(closure)
        # if scheduler is not None:
        #     scheduler.step()
        return loss.item()

    def train(self, model):
        loss_progress = []
        if self.opti == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate_adam,
                                         betas = (.99, .999))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8)

        if self.opti == 'LBFGS':
            optimizer = torch.optim.LBFGS(model.parameters(), lr=self.learning_rate_lbfgs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
        
        loss = self.train_iter(model=model, optimizer=optimizer)
        loss_progress.append(loss)

        for i in range(0,int(self.num_iter)+1):
            
            loss = self.train_iter(model=model, optimizer=optimizer)
            loss_progress.append(loss)
            if self.opti == 'Adam' and (i+1)%50 ==0:
                
                if abs((loss - loss_progress[-1])/max(loss_progress[-1],loss,1))<self.tol:
                    break
                
            if self.opti == 'LBFGS' and (i+1)%8 ==0:
                
                if abs((loss - loss_progress[-1])/max(loss_progress[-1],loss,1))<self.tol:
                    
                    break
                
            if self.lr_decay and self.opti == 'Adam' and (i+1)%500 == 0:
                scheduler.step()
            elif self.lr_decay and self.opti == 'LBFGS' and (i+1)%10 == 0:
                scheduler.step()

        return loss


    # def _loss(self, model):
    #     with torch.no_grad():
    #         X_hat = model(self.X_torch)
    #         if self.loss_type == 'l2':
    #             loss = self.squared_loss(X_hat, self.X_torch)
    #         elif self.loss_type == 'logl2':
    #             loss = self.log_loss(X_hat, self.X_torch)
    #     return loss.item()

    
    

    # @staticmethod
    # def check_gradient_info(model):
    #     for name, param in model.named_parameters():
    #         print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    @staticmethod
    def copy_model(model, model_clone):
        model_clone.load_state_dict(model.state_dict())

        
        for param_target, (name_source, param_source) in zip(model_clone.parameters(), model.named_parameters()):
            
            param_target.requires_grad = param_source.requires_grad

        return model_clone
    def fit(self,
            topo: list,
            no_large_search,
            size_small,
            size_large,
            ):

        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo
        self.Z = create_Z(self.topo)
        # training model according to initial topological sort.
        self.model.reset_by_topo(topo = self.topo)
        self.loss = self.train(model = self.model)
        conditional_print(f"The initial model, current loss {self.loss}")
        self.W_adj = self.model.layer0_to_adj()
        self.h, self.G_h = self.model._h(W=self.W_adj)
        # let gradient of F play no role in update
        self.G_loss = np.ones((self.d,self.d))

        idx_set_small, idx_set_large = find_idx_set_updated(G_h=self.G_h, G_loss=self.G_loss, Z=self.Z, size_small=size_small,
                                        size_large=size_large)
        idx_set = list(idx_set_small)

        while bool(idx_set):
            idx_len = len(idx_set)
            indicator_improve = False
           
            model_clone = type(self.model)(self.model.dims)
            
            for i in range(idx_len):
                model_clone = self.copy_model(model = self.model,model_clone = model_clone)
                topo_clone = create_new_topo(self.topo, idx_set[i], opt=1)
                model_clone.update_nn_by_topo(topo = self.topo, index = idx_set[i])
            
                loss_clone = self.train(model = model_clone)

                conditional_print(f"working with topological sort:{topo_clone}, current loss {loss_clone}")

                model_clone.reset_by_topo(topo = topo_clone)

                if loss_clone<self.loss:
                    indicator_improve = True
                    # model_clone is successful, and we get copy of it
                    conditional_print(f"better loss found, topological sort: {topo_clone}, and loss: {loss_clone}")
                    self.model = self.copy_model(model = model_clone, model_clone = self.model)
                    self.topo = topo_clone
                    self.Z = create_Z(topo_clone)
                    self.loss = loss_clone
                    self.W_adj = self.model.layer0_to_adj()
                    self.h, self.G_h = self.model._h(W=self.W_adj)
                    break

            if not indicator_improve:
                if large_space_used < no_large_search:
                    indicator_improve_large = False
                    # print('++++++++++++++++++++++++++++++++++++++++++++')
                    conditional_print(f"start to use large search space for {large_space_used + 1} times")
                    # print('++++++++++++++++++++++++++++++++++++++++++++')
                    idx_set = list(set(idx_set_large) - set(idx_set_small))
                    idx_len = len(idx_set)
                    
                    for i in range(idx_len):
                        model_clone = self.copy_model(model=self.model, model_clone=model_clone)
                        topo_clone = create_new_topo(self.topo, idx_set[i], opt=1)
                        model_clone.update_nn_by_topo(topo=self.topo, index=idx_set[i])
                        
                        loss_clone = self.train(model=model_clone)
                        conditional_print(f"working with topological sort:{topo_clone}, current loss {loss_clone}")
                        model_clone.reset_by_topo(topo=topo_clone)


                        if loss_clone<self.loss:
                            indicator_improve_large = True
                            self.model = self.copy_model(model=model_clone, model_clone=self.model)
                            self.topo = topo_clone
                            self.Z = create_Z(topo_clone)
                            self.loss = loss_clone
                            self.W_adj = self.model.layer0_to_adj()
                            self.h, self.G_h = self.model._h(W=self.W_adj)
                            conditional_print(f"better loss found, topological sort: {topo_clone}, and loss: {loss_clone}")
                             
                            break
                    if not indicator_improve_large:
                        conditional_print("Using larger search space, but we cannot find better loss")
                        break
                    large_space_used =large_space_used+ 1 
                else:
                    conditional_print("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break

             
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=self.G_h, G_loss=self.G_loss, Z=self.Z,
                                                        size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1
        return self.W_adj, self.topo, self.loss, self.model



if __name__ == '__main__':
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    rd_int = int(np.random.randint(10000, size=1)[0])
    rd_int = 4321
    utils.set_random_seed(rd_int)
    torch.manual_seed(rd_int)

    n, d, s0, graph_type, sem_type = 1000, 10, 10, 'ER', 'mlp'
    B_true = utils.simulate_dag(d, s0, graph_type)
    assert utils.is_dag(B_true)
    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
    topo_random = list(np.random.permutation(range(d)))
    
    # Whether to print the intermediate results
    PRINT_ENABLED = True # you can set it to False to suppress the intermediate results

    # set up the model
    dims = [d, 40, 1]
    no_large_search = 1
    size_small = 10
    size_large = 40
    learning_rate = None 
    num_iter = 1e4
    lambda1 = 0.01,
    lambda2 = 0.01,
    loss_type = 'l2'
    opti = 'LBFGS' # 'Adam'
    lr_decay = True

    Topo_mlp = TopoMLP(dims = dims)
    Topo_nonlinear = TOPO_Nonlinear(X = X, model = Topo_mlp,num_iter = num_iter,
                                    lambda1 = lambda1,lambda2 = lambda2,loss_type = loss_type,
                                    opti = opti, lr_decay = lr_decay)
    time_start = time.time()
    W,topo,loss, model = Topo_nonlinear.fit(topo = topo_random, size_large = size_large,size_small = size_small,
                       no_large_search = no_large_search)
    time_end = time.time()
    
    W_thres = threshold_W(W,threshold= 0.5)
    acc = utils.count_accuracy(B_true, W_thres != 0)
    print(acc)
    print(f"running time is {time_end - time_start}")
    print("...")

