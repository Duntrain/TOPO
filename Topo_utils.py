import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
import scipy.linalg as slin
from copy import copy
import torch
def threshold_W(W, threshold=0.3):
    """
    :param W: adjacent matrix
    :param threshold:
    :return: a threshed adjacent matrix
    """
    W_new = np.zeros_like(W)
    W_new[:] = W
    W_new[np.abs(W_new) < threshold] = 0
    return W_new

def init_Wstar(X, Z, tau=0, method='Linear'):
    """ Return Optimal solution to problem
    min_{W} \|X-XW\|_2^2+tau \|W\|_1  s.t. W_{ij}=0 (i,j)\in  \mathcal{Z}

    :param X: data
    :param Z: edge absence constraints
    :param tau: coefficient of L1 regression
    :param method: Lasso/Lars/linear regression
    :return: W^*(\mathcal{Z})
    """
    d = X.shape[1]
    W = np.zeros((d, d))
    if method == 'Lasso':
        reg = Lasso(alpha=tau, fit_intercept=False, max_iter=10000)
    elif method == 'Lars':
        reg = LassoLars(alpha=tau, fit_intercept=False, max_iter=10000)
    elif method == 'Linear':
        reg = LinearRegression(fit_intercept=False)
    else:
        raise ValueError('unknown method')
    for j in range(d):

        if (~Z[:, j]).any():
            reg.fit(X[:, ~Z[:, j]], X[:, j])
            W[~Z[:, j], j] = reg.coef_
        else:
            W[:, j] = 0
    return W

def create_Z(ordering):
    """
    Create edge absence constraints \mathcal{Z} corresponding to topological ordering
    :param ordering: topological sort
    :return: bool matrix

    create_Z([0,1,2,3])
    Out:
    array([[ True, False, False, False],
       [ True,  True, False, False],
       [ True,  True,  True, False],
       [ True,  True,  True,  True]])

    """
    d = len(ordering)
    Z = np.ones((d, d), dtype=bool)
    for i in range(d - 1):
        Z[ordering[i], ordering[i + 1:]] = False
    return Z

def obj_loss(X, W, equal_variances=True, weighted_matrix=None):
    """
    Calculate sample loss and gradient.
    :param W: estimated W
    :param X: data
    :param equal_variances: whether is it equal variance
    :param weighted_matrix: if variance is unequal, weighted_matrix
    :return: loss and gradient
    """
    M = X @ W
    if equal_variances:
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ R
    else:
        if weighted_matrix is None:
            raise ValueError('Weighted matrix is required')
        else:
            R = (X - M) @ weighted_matrix
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = -1 / X.shape[0] * X.T @ R @ weighted_matrix
    return loss, G_loss

def h_func(W, h_type='log_det'):
    """Evaluate value and gradient of acyclicity constraint.
    Option 1: h(W) = Tr(I+|W|/d)^d-d
    """
    h_types = ['poly','exp_abs','exp_square','log_det']
    assert h_type in h_type, f"acyclicity function should be one of {h_types}"
    d = W.shape[0]
    if h_type == 'poly':
        A = np.abs(W)
        E = np.eye(d) + A / d
        G_h = np.linalg.matrix_power(E, d - 1).T
        h = (G_h * E).sum() - d

    elif h_type == 'exp_abs':
        A = np.abs(W)
        E = slin.expm(A)
        h = np.trace(E) - d
        G_h = E.T

    elif h_type == 'exp_square':
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * 2

    elif h_type == 'log_det':
        """
        h(W) = -log det(sI-W*W) + d log (s)
        nabla h(W) = 2 (sI-W*W)^{-T}*W
        """
        I = np.eye(d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + d * np.log(s)
        G_h =  slin.inv(M).T


    return h, G_h


def create_new_topo(topo, idx, opt=1):
    '''
    Args:
        topo: topological
        index: (i,j)
        opt: 1: how to swap position of i and j
    Returns:
        a new topological sort
    '''

    topo0 = copy(topo)
    topo0 = list(topo0)
    i, j = idx
    i_pos,j_pos = topo0.index(i),topo0.index(j)
    if opt == 1:
        topo0[i_pos] = idx[1]
        topo0[j_pos] = idx[0]
    elif opt == 2:
        topo0.remove(j)
        topo0.insert(i_pos, j)
    else:
        topo0.remove(i)
        topo0.insert(j_pos, i)
    return topo0


def find_hgrad_index(G_h, Z, thres=1e-2):
    """
    Find where {(i.j)| i\neq j, (G_h)_{ij}<thres, Z[i,j] = True }

    :param G_h: gradient of h
    :param Z: edge absence constaints
    :param thres: threshold for gradient of h
    :return: set {(i.j)| i\neq j, (G_h)_{ij}<thres, Z[i,j] = True }
    """
    G_h0 = copy(G_h)
    index = np.transpose(np.where(np.logical_and(G_h0 <= thres, Z)))
    index0 = index[index[:, 1] != index[:, 0],]
    return index0


def find_Fgrad_index(G_loss, Z, thres=1e-3):
    """
    Find where {(i,j)| G_loss(i,j) not = 0 and Z(i,j)= True}

    :param G_loss: gradient of Loss function
    :param Z: edge absence constaints
    :param thres:
    :return: set {(i.j)| i\neq j, |(G_F)_{ij}|>=thres, Z[i,j] = True }
    """
    not0grad = np.logical_or(G_loss <= (-thres), G_loss >= thres)
    index = np.transpose(np.where(np.logical_and(not0grad, Z)))
    index0 = index[index[:, 1] != index[:, 0],]
    return index0


def find_common(indx1, indx2):
    """
    find the intersection between indx1 and indx2

    :param indx1: index set A
    :param indx2: index set B
    :return: return A\cap B
    """
    A = list(zip(indx1[:, 0], indx1[:, 1]))
    B = list(zip(indx2[:, 0], indx2[:, 1]))
    return set(A).intersection(B)


def find_idx_set(G_h, G_loss, Z, size_small, size_large):
    r"""
    Implement Algorithm 2 in Paper, find

    index_set_small = \mathcal{Y}(W,\tau_*,\xi^*) s.t. |index_set_small| = size1
    index_set_large = \mathcal{Y}(W,\tau^*,\xi_*) s.t. |index_set_large| = size2

    :param G_h: gradient of h
    :param G_loss: gradient of loss
    :param Z: edge absence constraints
    :param size1: size of \mathcal{Y}(W,\tau_*,\xi^*)
    :param size2: size of \mathcal{Y}(W,\tau^*,\xi_*)
    :return: index_set_small, index_set_large
    """
    gFs = [0]
    # gFs =  [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4]
    # gFs = [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]
    ghs = sorted([40, 30, 20, 10, 5, 2, 1, 0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04,
                  0.03, 0.025, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00005, 1e-5, 8e-6, 6e-6, 4e-6, 2e-6, 1e-6, 1e-7, 0])

    M = np.zeros([len(ghs), len(gFs)])
    for count_gF, gF in enumerate(gFs):
        for count_gh, gh in enumerate(ghs):
            indx1 = find_hgrad_index(G_h, Z=Z, thres=gh)
            # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
            indx2 = find_Fgrad_index(G_loss, Z=Z, thres=gF)
            index_set = find_common(indx1, indx2)
            M[count_gh, count_gF] = len(index_set)

    i1, j1 = np.unravel_index(np.argmin(np.abs(M - size_small), axis=None), M.shape)
    i2, j2 = np.unravel_index(np.argmin(np.abs(M - size_large), axis=None), M.shape)

    indx1_small = find_hgrad_index(G_h, Z=Z, thres=ghs[i1])
    # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
    indx2_small = find_Fgrad_index(G_loss, Z=Z, thres=gFs[j1])
    index_set_small = find_common(indx1_small, indx2_small)

    if len(index_set_small) > size_small + 20 and ghs[i1] == 0:
        size1_th_largest = np.partition(np.abs(G_loss[(indx1_small[:, 0], indx1_small[:, 1])]), -1 * size_small)[-1 * size_small]
        indx2_small_v = find_Fgrad_index(G_loss, Z=Z, thres=size1_th_largest)
        index_set_small = find_common(indx1_small, indx2_small_v)

    indx1_large = find_hgrad_index(G_h, Z=Z, thres=ghs[i2])
    # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
    indx2_large = find_Fgrad_index(G_loss, Z=Z, thres=gFs[j2])
    index_set_large = find_common(indx1_large, indx2_large)

    if len(index_set_large) < (size_large - 100):
        indx2_large = find_Fgrad_index(G_loss, Z=Z, thres=0)
        size2 = min(size_large, len(indx2_large))
        size2_th_smallest = np.partition(G_h[(indx2_large[:, 0], indx2_large[:, 1])], size2 - 2)[size2 - 2]
        indx1_large_v = find_hgrad_index(G_h, Z=Z, thres=size2_th_smallest)
        index_set_large = find_common(indx1_large_v, indx2_large)

    return index_set_small, index_set_large


def init_Wstar_slice(X, index_y, index_x, tau=0, method="Linear"):
    if method == 'Lasso':
        reg = Lasso(alpha=tau, fit_intercept=False, max_iter=10000)
    elif method == 'Lars':
        reg = LassoLars(alpha=tau, fit_intercept=False, max_iter=10000)
    elif method == 'Linear':
        reg = LinearRegression(fit_intercept=False)
    else:
        raise ValueError('unknown method')
    reg.fit(X[:, index_x], X[:, index_y])
    return reg.coef_


def update_topo_linear(X, topo_order, W_Z, index, tau=0, method='Linear', opt=1):
    """
    create a new topological sort based on current topological sort
    and pair (i,j) and return W based on such new topological sort

    :param X: data
    :param top_order: topological sort
    :param W_Z: W_{\pi} based on top_order
    :param index: (i,j) in top_order their relative position are like [-----,j,-----,i,------]
    :param tau: coefficient of l_1 penalty
    :param method: Linear/Lass/Lars
    :param opt: how to swap pair (i,j)
    :return:
    """

    topo_order0 = copy(topo_order)
    l = len(topo_order0)
    W_new = np.zeros_like(W_Z)
    i = index[0]
    j = index[1]
    wherei = topo_order.index(i)
    wherej = topo_order.index(j)
    dist = wherei - wherej

    # W_new[:, top_order[:wherej]] = W_Z[:, top_order[:wherej]]
    # W_new[:, top_order[(wherei + 1):]] = W_Z[:, top_order[(wherei + 1):]]
    # top_order0 = create_new_topo(top_order0, index, opt=opt)
    # for k in range(wherej, wherei + 1):
    #     W_new[top_order0[:k], top_order0[k]] = init_Wstar_slice(X, top_order0[k], top_order0[:k], tau=tau,
    #                                                             method=method)

    if wherej >= 1:
        if wherei <= (l - 2):
            if (wherej + 1) != wherei:
                W_new[:, topo_order[:wherej]] = W_Z[:, topo_order[:wherej]]
                W_new[:, topo_order[(wherei + 1):]] = W_Z[:, topo_order[(wherei + 1):]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                for k in range(wherej, wherei + 1):
                    W_new[topo_order0[:k], topo_order0[k]] = init_Wstar_slice(X, topo_order0[k], topo_order0[:k], tau=tau,
                                                                            method=method)
            else:
                W_new[:, topo_order[:wherej]] = W_Z[:, topo_order[:wherej]]
                W_new[:, topo_order[(wherei + 1):]] = W_Z[:, topo_order[(wherei + 1):]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                W_new[topo_order0[:wherej], topo_order0[wherej]] = init_Wstar_slice(X, topo_order0[wherej],
                                                                                  topo_order0[:wherej], tau=tau,
                                                                                  method=method)
                W_new[topo_order0[:wherei], topo_order0[wherei]] = init_Wstar_slice(X, topo_order0[wherei],
                                                                                  topo_order0[:wherei], tau=tau,
                                                                                  method=method)

        else:
            if (wherej + 1) != wherei:
                W_new[:, topo_order[:wherej]] = W_Z[:, topo_order[:wherej]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                for k in range(wherej, wherei + 1):
                    W_new[topo_order0[:k], topo_order0[k]] = init_Wstar_slice(X, topo_order0[k], topo_order0[:k], tau=tau,
                                                                            method=method)
            else:
                W_new[:, topo_order[:wherej]] = W_Z[:, topo_order[:wherej]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                W_new[topo_order0[:wherej], topo_order0[wherej]] = init_Wstar_slice(X, topo_order0[wherej],
                                                                                  topo_order0[:wherej], tau=tau,
                                                                                  method=method)
                W_new[topo_order0[:wherei], topo_order0[wherei]] = init_Wstar_slice(X, topo_order0[wherei],
                                                                                  topo_order0[:wherei], tau=tau,
                                                                                  method=method)

    else:
        if wherei <= (l - 2):
            if (wherej + 1) != wherei:
                W_new[:, topo_order[(wherei + 1):]] = W_Z[:, topo_order[(wherei + 1):]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                for k in range(wherej + 1, wherei + 1):
                    W_new[topo_order0[:k], topo_order0[k]] = init_Wstar_slice(X, topo_order0[k], topo_order0[:k], tau=tau,
                                                                            method=method)

            else:
                W_new[:, topo_order[(wherei + 1):]] = W_Z[:, topo_order[(wherei + 1):]]
                topo_order0 = create_new_topo(topo_order0, index, opt=opt)
                W_new[topo_order0[:wherei], topo_order0[wherei]] = init_Wstar_slice(X, topo_order0[wherei],
                                                                                  topo_order0[:wherei], tau=tau,
                                                                                  method=method)
        else:
            topo_order0 = create_new_topo(topo_order0, index, opt=opt)
            Z_G = create_Z(topo_order0)
            W_new = init_Wstar(X, Z_G, tau=tau, method=method)
    return W_new, topo_order0, create_Z(topo_order0), dist


def argwhereedges(W_true):
    dd = np.argwhere(np.abs(W_true) > 0.01)
    edges = list(zip(dd[:, 0], dd[:, 1]))
    return edges

def assign_negative(i,j,topo):
    succ = False
    if np.size(np.where(topo == i)) and np.size(np.where(topo == j)):
        pos_i = np.where(topo == i)
        pos_j = np.where(topo == j)
        if not np.any(topo[pos_j[0][0]:(pos_i[0][0] + 1)] == -1):
            topo[pos_j[0][0]:(pos_i[0][0] + 1)] = -1
            succ = True

    return topo, succ


def create_new_topo_greedy(topo,loss_collections,idx_set,loss,opt = 1):
    topo0 = np.array(copy(topo)).astype(int)
    loss_table = np.concatenate((np.array(list(idx_set)), loss_collections.reshape(-1, 1)), axis=1)
    loss_table_good = loss_table[np.where(loss_collections < loss)]
    sorted_loss_table_good = loss_table_good[loss_table_good[:,2].argsort()]
    len_loss_table_good = sorted_loss_table_good.shape[0]
    for k in range(len_loss_table_good):
        i,j = sorted_loss_table_good[k,0:2]
        i,j = int(i),int(j)
        topo0, succ = assign_negative(i, j, topo0)
        if succ:
            topo = create_new_topo(topo= topo,idx =(i,j),opt = opt)
    return topo

def gradient_l1(W, A, lambda1):
    grad = np.zeros_like(W)
    pos_W = W > 0
    neg_W = W < 0
    zero_W = ~np.logical_or(pos_W, neg_W)
    grad[pos_W] = lambda1
    grad[neg_W] = (-lambda1)
    pos_A = A > lambda1
    neg_A = A < -lambda1
    zero_A = ~np.logical_or(pos_A, neg_A)
    grad[zero_W & pos_A] = (-lambda1)
    grad[zero_W & neg_A] = lambda1
    grad[zero_W & zero_A] = 0

    return grad

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss
