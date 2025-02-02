import numpy as np
from tensorly import fold, unfold
import matplotlib.cm as cm
from scipy.io import loadmat, savemat


def left_unfold(input_tensor):
    '''
    Left unfold a 3-way tensor of size n1 * n2 * n3 to (n1n2) * n3

    Input Parameters:
        input_tensor: a 3-way tensor of shape (n1,n2,n3)

    Output: 2-way tensor of shape (n1n2,n3)
    '''
    if not input_tensor.ndim == 3:
        raise Exception("left unfolding not on 3-way tensor!")

    return unfold(input_tensor, mode=2).T


def left_unfold_rev(input_tensor, shape):
    '''
    Reverse the left unfolding operation.

    Input Parameters:
        input_tensor: a 2-way tensor of shape (n1*n2,n3)
        shape: a 3-tuple specifying the original tensor shape (n1,n2,n3)

    Output: a 3-way tensor of shape (n1,n2,n3)
    '''
    return fold(input_tensor.T, mode=2, shape=shape)


def right_unfold(input_tensor):
    '''
    Right unfold a 3-way tensor of size n1 * n2 * n3 to n1 * (n2*n3)

    Input Parameters:
        input_tensor: a 3-way tensor of shape (n1,n2,n3)

    Output: 2-way tensor of shape (n1,n2*n3)
    '''
    if not input_tensor.ndim == 3:
        raise Exception("right unfolding not on 3-way tensor!")

    return unfold(input_tensor, mode=0)


def right_unfold_rev(input_tensor, shape):
    '''
    Reverse the right unfolding operation.

    Input Parameters:
        input_tensor: a 2-way tensor of shape (n1,n2*n3)
        shape: a 3-tuple specifying the original tensor shape (n1,n2,n3)

    Output: a 3-way tensor of shape (n1,n2,n3)
    '''
    return fold(input_tensor, mode=0, shape=shape)


def tensor_sep(input_tensor, mode):
    '''
    Compute the k-th mode separation of a tensor-train tensor

    params:
        input_tensor: a K-mode tensor
        mode: the mode along which to do separation
    '''
    d = input_tensor.shape
    K = input_tensor.ndim
    n = np.prod(d)
    nrows = 1 if mode == -1 else np.prod(d[:(mode + 1)])
    return np.reshape(input_tensor, (nrows, -1), order="F")


def tensor_product(tensor1, tensor2):
    '''
    Compute the product of two tensors.

    params:
        tensor1: tensor of size d_1 * d_2 * ... * d_i * r
        tensor2: tensor of size r * d_{i+1} * ... * d_j

    return: tensor of size d_1 * ... * d_i * d_{i+1} * ... * d_j
    '''
    d1, d2 = tensor1.shape, tensor2.shape
    d = d1[:-1] + d2[1:]  # resulting shape
    if not d1[-1] == d2[0]:
        raise Exception("Cannot take product of two tensors with incompatible shape.")
    r = d1[-1]
    tensor1 = tensor1.reshape((-1, r), order='F')
    tensor2 = tensor2.reshape((r, -1), order='F')
    return np.reshape(tensor1 @ tensor2, d, order='F')


def tt_left_part(tt_tensor, mode):
    '''
    Compute the left part of a tensor-train tensor.

    params:
        tt_tensor: a tensor-train object from tensorly, shape = (d_1,...,d_K), tt-rank = (r_1,...,r_{K-1})
        mode: the mode at which one extracts the left part (use zero-indexing, so 0 is the first mode)

    return: matrix of size (d_1*...d_{mode+1}) * r_{mode+1}
    '''
    d = tt_tensor.shape
    K = len(d)

    if mode >= K:
        raise Exception("Cannot extract left part with mode higher than the tensor mode.")

    if mode < -1:
        raise Exception("Cannot extract left part with mode lower than -1.")

    if mode == -1:
        return np.array([[1]])

    L = np.reshape(tt_tensor[0], (d[0], -1), order='F')
    if mode == 0:
        return L
    else:
        for k in range(1, mode + 1):
            L = tensor_product(L, tt_tensor[k])
            L = L.reshape((-1, L.shape[-1]), order='F')
        return L


def tt_right_part(tt_tensor, mode):
    '''
    Compute the right part of a tensor-train tensor.

    params:
        tt_tensor: a tensor-train object from tensorly, shape = (d_1,...,d_K), tt-rank = (r_1,...,r_{K-1})
        mode: the mode at which one extracts the right part (use zero-indexing, so 0 is the first mode)

    return: matrix of size r_{mode} * (d_{mode+1}*...*d_K)
    '''
    d = tt_tensor.shape
    K = len(d)

    if mode > K:
        raise Exception("Cannot extract right part with mode higher than the tensor mode.")

    if mode <= -1:
        raise Exception("Cannot extract right part with mode lower than -1.")

    if mode == K:
        return np.array([[1]])

    R = np.reshape(tt_tensor[-1], (-1, d[-1]), order='F')
    if mode == K - 1:
        return R
    else:
        for k in range(K - 2, mode - 1, -1):
            R = tensor_product(tt_tensor[k], R)
            R = R.reshape((R.shape[0], -1), order='F')
        return R

def RMSE(A, B):
    return np.sqrt(np.mean((A-B)**2))

def RSE(A, B):
    return np.sqrt(np.mean((A-B)**2) / np.mean(B**2))

def tensor_cmap(tensor, cmap="RdBu_r", vmin=-2, vmax=2):
    '''
    Transform an input tensor to an RGB tensor for plotting
    '''

    colormap = cm.get_cmap(cmap)
    vmin = np.amin(tensor) if vmin is None else vmin
    vmax = np.amax(tensor) if vmax is None else vmax
    color_tensor = colormap((tensor - vmin) / (vmax - vmin))[:, :, :, :-1]
    return color_tensor

def npz2mat(filename):
    data = np.load(filename)
    filename = filename.replace(".npz", ".mat")
    savemat(filename, {"B": data["B"], "sample": data["sample"], "train": data["train"]})