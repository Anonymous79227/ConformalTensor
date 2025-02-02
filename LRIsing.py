import itertools, copy
import numpy as np
from math import exp, log
from scipy.stats import norm
from scipy.ndimage import convolve
from numpy.linalg import inv, qr, svd
from tensorly.decomposition import tensor_train, tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tt_tensor import tt_to_tensor, tt_to_unfolded
from tensorly.tenalg import mode_dot, multi_mode_dot, kronecker
from tensorly import fold, unfold
from utils import *


# new ver.
class TensorIsing:
    '''
    Tensor Ising Model class
    '''

    def __init__(self, g_func="product", h_func="logit", g_params=None, h_params=None, nb="1-nb"):
        '''
        params:
            g_func: selection of the bi-variate function g (values from {"zero", "linear", "product", "sigmoid"})
            h_func: selection of the uni-variate function h (values from {"zero", "logit", "probit"})
            g_params: parameters of g function (optional)
            h_params: parameters of h function (optional)
            nb: neighboring structure (values from {"1-nb", "1-nb-spatial", "cube-nb"})
        '''

        if g_func == "linear":
            # g(x,y) = x+y
            self.g = lambda x, y: x + y
            self.gp = lambda x, y: 1
        elif g_func == "product":
            # g(x,y) = xy
            self.g = lambda x, y: x * y / g_params['temp']
            self.gp = lambda x, y: y / g_params['temp']
        elif g_func == "sigmoid":
            # g(x,y) = (sigmoid(x*y)-1/2)
            self.g = lambda x, y: np.exp(x * y) / (1 + np.exp(x * y)) - 0.5
            self.gp = lambda x, y: y * np.exp(x * y) / ((1 + np.exp(x * y)) ** 2)
        elif g_func == "zero":
            # g(x,y) = 0
            self.g = lambda x, y: 0
            self.gp = lambda x, y: 0
        else:
            raise Exception("Unknown g function. Please choose from 'zero', 'linear', 'product', 'sigmoid'.")

        if h_func == "logit":
            # h(x) = x/2
            self.h = lambda x: x / 2
            self.hp = lambda x: 1 / 2
        elif h_func == "probit":
            self.h = lambda x: 0.5 * log(norm.cdf(x) / (1 - norm.cdf(x)))
            self.hp = lambda x: 0.5 * norm.pdf(x) / (norm.cdf(x) * (1 - norm.cdf(x)))
        elif h_func == "zero":
            self.h = lambda x: 0
            self.hp = lambda x: 0
        else:
            raise Exception("Unknown h function. Please choose from 'zero', 'logit', 'probit'.")

        self.g_func = g_func
        self.h_func = h_func
        self.g_params = g_params
        self.h_params = h_params
        self.nb = nb
        self.kernel = None

    def LocalConv(self, W, B):
        # compute sum of g(B_i,B_j) * W_j for each entry i
        d = W.shape
        K = W.ndim
        T = np.zeros_like(W).astype(float)

        if not self.g_func == "zero":
            if self.g_func in ["sigmoid"]:
                slices = (slice(1, -1),) * K
                pW, pB = np.pad(W, 1), np.pad(B, 1)
                if self.nb in ["1-nb", "1-nb-spatial"]:
                    In = np.zeros((*d, 2 * K))
                    neighbor_mode = K if self.nb == "1-nb" else K - 1
                    for k in range(neighbor_mode):
                        for i, dx in enumerate([-1, 1]):
                            pB_nb = np.roll(pB, shift=dx, axis=k)
                            pW_nb = np.roll(pW, shift=dx, axis=k)
                            g = self.g(B, pB_nb[slices])  # pairwise interaction with g function
                            In[..., 2 * k + i] = g * pW_nb[slices]
                elif self.nb == "cube-nb":
                    In = np.zeros((*d, 3 ** K - 1))
                    arr = np.array([-1, 0, 1])
                    combs = itertools.product(*([arr] * K))
                    dx = [np.array(comb) for comb in combs]
                    for i, m in enumerate(dx):
                        if not np.all(m == 0):
                            pB_nb = np.roll(pB, shift=m)
                            pW_nb = np.roll(pW, shift=m)
                            g = self.g(B, pB_nb[slices])  # pairwise interaction with g function
                            In[..., i] = g * pW_nb[slices]
                T += np.sum(In, axis=-1)
            else:
                # set convolution kernel
                if self.nb == "cube-nb":
                    kernel = np.ones((3,) * K)
                    kernel[(1,) * K] = 0
                elif self.nb in ["1-nb", "1-nb-spatial"]:
                    neighbor_mode = K if self.nb == "1-nb" else K - 1
                    kernel = np.zeros((3,) * K)
                    for k in range(neighbor_mode):
                        center = np.ones(K, dtype=int)
                        center[k] = 0
                        kernel[tuple(center)] = 1
                        center[k] = 2
                        kernel[tuple(center)] = 1
                self.kernel = kernel

                # compute conditional probability with convolution
                if self.g_func == "linear":
                    T += B * convolve(W, weights=self.kernel, mode='constant') + convolve(W * B, weights=self.kernel,
                                                                                          mode='constant')
                elif self.g_func == "product":
                    T += B * convolve(W * B, weights=self.kernel, mode='constant')

        return T

    def CondProb(self, W, B, T=None, invtemp=None):
        '''
        Evaluate the conditional probability that a certain entry equals to 1.
        params:
            W: K-mode binary tensor (values from {-1,1})
            B: K-mode real-valued tensor parameter
            invtemp: scaling factor of the g function
        '''
        d = W.shape
        K = W.ndim
        invtemp = 1 if 'temp' not in self.g_params else 1 / self.g_params['temp']
        if T is None:
            T = self.LocalConv(W, B) * invtemp
        else:
            T = T * invtemp
        T += self.h(B)

        # trim T if necessary for numerical stability
        T = np.minimum(2 * T, 700)
        P = np.exp(T) / (1 + np.exp(T))
        P = np.clip(P, 1e-4, 1 - 1e-4)
        return P

    def Pseudo_NLL(self, W, B, P=None, invtemp=None, q=1):
        '''
        Evaluate the pseudo negative log-likelihood.
        '''
        invtemp = 1 if 'temp' not in self.g_params else 1 / self.g_params['temp']
        if P is None:
            P = self.CondProb(W, B, invtemp=invtemp)
        P = np.clip(P, 1e-4, 1 - 1e-4)  # clip for numerical stability
        NLL = (-np.log(q * P) * (W / 2 + 1 / 2) - np.log(1 - q * P) * (1 / 2 - W / 2)).sum()
        return NLL

    def Grad(self, W, B, P=None, invtemp=None, q=1):
        '''
        Compute the vanilla gradient of the tensor Ising model.
        '''
        invtemp = 1 if 'temp' not in self.g_params else 1 / self.g_params['temp']
        
        if P is None:
            P = self.CondProb(W, B, invtemp=invtemp)
        V = (1 - P) * (q * P - (W / 2 + 1 / 2)) / (1 - q * P)

        d = W.shape
        K = W.ndim
        G = 2 * self.hp(B) * V  # output gradient tensor

        if not self.g_func == "zero":
            if self.g_func in ["sigmoid"]:
                slices = (slice(1, -1),) * K
                # compute (V_i*W_j + V_j*W_i)*g(B_i,B_j)
                pW, pB, pV = np.pad(W, 1), np.pad(B, 1), np.pad(V, 1)
                if self.nb in ["1-nb", "1-nb-spatial"]:
                    In = np.zeros((*d, 2 * K))
                    neighbor_mode = K if self.nb == "1-nb" else K - 1
                    for k in range(neighbor_mode):
                        for i, dx in enumerate([-1, 1]):
                            pB_nb = np.roll(pB, shift=dx, axis=k)
                            pV_nb = np.roll(pV, shift=dx, axis=k)
                            pW_nb = np.roll(pW, shift=dx, axis=k)
                            gx = self.gp(B, pB_nb[slices])  # pairwise interaction with g_x function
                            In[..., 2 * k + i] = gx * (V * pW_nb[slices] + pV_nb[slices] * W)
                elif self.nb in ["cube-nb"]:
                    In = np.zeros((*d, 3 ** K - 1))
                    arr = np.array([-1, 0, 1])
                    combs = itertools.product(*([arr] * K))
                    dx = [np.array(comb) for comb in combs]
                    for i, m in enumerate(dx):
                        if not np.all(m == 0):
                            pB_nb = np.roll(pB, shift=m)
                            pV_nb = np.roll(pV, shift=m)
                            pW_nb = np.roll(pW, shift=m)
                            gx = self.gp(B, pB_nb[slices])  # pairwise interaction with g_x function
                            In[..., i] = gx * (V * pW_nb[slices] + pV_nb[slices] * W)
                G += 2 * np.sum(In, axis=-1) * invtemp
            else:
                if self.g_func == "linear":
                    G += 2 * (V * convolve(W, weights=self.kernel, mode="constant") + W * convolve(V,
                                                                                                   weights=self.kernel,
                                                                                                   mode="constant")) * invtemp
                elif self.g_func == "product":
                    G += 2 * (V * convolve(W * B, weights=self.kernel, mode="constant") + W * convolve(V * B,
                                                                                                       weights=self.kernel,
                                                                                                       mode="constant")) * invtemp

        return G
        

class LRIsing:
    '''
    Low-rank Ising Model Estimator
    '''

    def __init__(self, g_func="product", h_func="logit", g_params=None, h_params=None, nb="1-nb", true_B=None,
                 true_temp=None):
        '''
        params:
            g_func: selection of the bi-variate function g (values from {"zero", "linear", "product", "sigmoid"})
            h_func: selection of the uni-variate function h (values from {"zero", "logit", "probit"})
            g_params: parameters of g function (optional)
            h_params: parameters of h function (optional)
            true_B: true tensor parameter (optional, only used in simulation)
        '''
        self.TenIsing = TensorIsing(g_func=g_func, h_func=h_func, g_params=g_params, h_params=h_params, nb=nb)
        self.true_B = true_B
        self.true_temp = true_temp

    def fit(self, W, rank, q=1, method="tensor-train", config=None):
        '''
        params:
            W: K-mode binary tensor (values from {-1,1})
            rank: rank of parameter tensor
            q: the train-calibration split ratio
            method: rank notion (values from {"tensor-train", "tucker"})
            config: algorithm configuration (e.g. maximum iteration, convergence threshold, etc.)
        '''

        if method == "tensor-train" and not W.ndim == len(rank) + 1:
            raise Exception(
                "Wrong tensor-train rank! Please provide a rank parameter with length equal to the number of modes minus 1.")

        if method == "tucker" and not W.ndim == len(rank):
            raise Exception(
                "Wrong tucker rank! Please provide a rank parameter with length equal to the number of modes.")

        if method == "tensor-train":
            self.fit_TT(W, rank, q, config)
        elif method == "tucker":
            self.fit_TK(W, rank, q, config)
        else:
            raise ValueError("Unknown method! Please choose from 'tensor-train' or 'tucker'.")

    def fit_TT(self, W, rank, q,
               config={"max_iter": 2000,
                       "thres": 1e-3,
                       "beta": 0.1,
                       "verbose": True,
                       "spikiness": 3.0,
                       "rse_est": 1.0,
                       "backtracking": False,
                       "armijo_threshold": 1e-4, 
                       "projected_grad": False}):
        '''
        Fit low tensor-train rank MPLE for Ising model
        params:
            W: K-mode binary tensor (values from {-1,1})
            rank: rank of parameter tensor
            q: the train-calibration split ratio
            config: algorithm configuration (e.g. maximum iteration, convergence threshold, etc.)
        '''

        d = W.shape
        K = W.ndim
        if K < 3:
            raise TypeError("Input tensor should contain at least three modes!")
        tt_rank = [1] + list(rank) + [1]  # full tensor-train rank

        # ----- Initialization ----- #
        if "seed" in config:
            np.random.seed(config['seed'])
        T = tensor_train(W + np.random.normal(loc=0, scale=0.2, size=d), rank=tt_rank)  # T is a tensorly tensor-train object
        B = tt_to_tensor(T)
        invtemp = 1 / self.TenIsing.g_params['temp']

        # evaluate initial conditional probability
        P = self.TenIsing.CondProb(W, B, invtemp=invtemp)
        NLL = self.TenIsing.Pseudo_NLL(W, B, P=P, invtemp=invtemp, q=q)
        NLL_hist = [NLL]

        # ----- Riemannian Gradient Descent ----- #
        beta = 0.1 if "beta" not in config else config["beta"]  # step size
        l, delta = 1, 1
        l_max = 500 if "max_iter" not in config else config["max_iter"]
        threshold = 1e-3 if "thres" not in config else config["thres"]
        verbose = False if "verbose" not in config else config["verbose"]
        print_frequency = 100 if "print_frequency" not in config else config["print_frequency"]
        spikiness = 2.0 if "spikiness" not in config else config["spikiness"]
        rse_est = 1.0 if "rse_est" not in config else config["rse_est"]
        backtracking = False if "backtracking" not in config else config['backtracking']
        armijo_threshold = 1e-4 if "armijo_threshold" not in config else config['armijo_threshold']    
        projected_grad = False if "projected_grad" not in config else config["projected_grad"]
        RSE_hist = []
        step_hist = []

        if self.true_B is not None:
            RSE_B = RSE(B, self.true_B)
            RSE_hist.append(RSE_B)

        while (l <= l_max) and (delta > threshold):
            if l % print_frequency == 0 and l > 0 and verbose:
                print(f"iter {l}...")

            # Step I: compute vanilla gradient
            G = self.TenIsing.Grad(W, B, P=P, invtemp=invtemp, q=q)

            if projected_grad:
                projG = G
            else:
                # Step II: tangent space projected gradient descent
                if K > 3:
                    # compute projected gradient
                    projG = np.zeros_like(G)
    
                    # compute first K-1 projection components
                    for k in range(K - 1):
                        TT_factors = [f for f in T]
                        LT = left_unfold(TT_factors[k])
                        orthogonal_component = np.eye(tt_rank[k] * d[k]) - LT @ LT.T
                        left_part, right_part = tt_left_part(T, k - 1), tt_right_part(T, k + 1)
                        LY = orthogonal_component @ np.kron(left_part, np.eye(d[k])).T @ tensor_sep(G,k) @ right_part.T @ np.linalg.inv(right_part @ right_part.T)
                        TT_factors[k] = left_unfold_rev(LY, TT_factors[k].shape)
                        projG += tt_to_tensor(TT_factors)
    
                    # compute the last projection components
                    TT_factors = [f for f in T]
                    LY = np.kron(tt_left_part(T, K - 2), np.eye(d[K - 1])).T @ tensor_sep(G, K - 1)
                    TT_factors[-1] = left_unfold_rev(LY, TT_factors[-1].shape)
                    projG += tt_to_tensor(TT_factors)
                elif K == 3:
                    # compute projected gradient faster for 3-mode tensors
                    T1, T2, T3 = T[0].squeeze(), T[1], T[2].squeeze()
                    T3_T3T = T3 @ T3.T
    
                    # update Y1
                    T2_1 = unfold(T2, mode=0)
                    C = unfold(mode_dot(G, T3, mode=2), mode=0)
                    U1 = unfold(mode_dot(T2, T3.T, mode=2), mode=0)
                    Y1 = T1.T @ C
                    Y1 = C - T1 @ Y1
                    Y1 = Y1 @ T2_1.T @ inv(U1 @ U1.T)
    
                    # update Y2
                    T2_3 = unfold(T2, mode=2)
                    C = unfold(multi_mode_dot(G, [T1.T, T3], modes=[0, 2]), mode=2).T
                    Y2 = T2_3 @ C
                    Y2 = C - T2_3.T @ Y2
                    Y2 = Y2 @ inv(T3_T3T)
                    Y2 = np.reshape(Y2, (rank[0], d[1], rank[1]))
    
                    # update Y3
                    u, s, v = np.linalg.svd(T1, full_matrices=False)
                    left_tensor = mode_dot(T2, (v.T @ np.diag(s)).T, mode=0)
                    right_tensor = mode_dot(G, u.T, mode=0)
                    Y3 = unfold(left_tensor, mode=2) @ unfold(right_tensor, mode=2).T
                    projG = multi_mode_dot(T2, [Y1, T3.T], modes=[0, 2]) + multi_mode_dot(Y2, [T1, T3.T], modes=[0, 2]) + multi_mode_dot(T2,[T1,Y3.T],modes=[0,2])            
                
            # track convergence
            if backtracking:
                stepsize = beta
                
                # descent along the projected gradient direction             
                B_tilde = B - stepsize * projG
                
                # Step III: retraction
                T = tensor_train(B_tilde, rank=tt_rank)
                B_new = tt_to_tensor(T)
                
                prev_NLL = NLL_hist[-1]
                P = self.TenIsing.CondProb(W, B_new, invtemp=invtemp)
                NLL = self.TenIsing.Pseudo_NLL(W, B_new, P=P, invtemp=invtemp, q=q)
                backtracking_threshold = armijo_threshold * (np.linalg.norm(projG) ** 2)                
                
                while ((prev_NLL - NLL) < stepsize * backtracking_threshold) and (stepsize >= 1e-3):
                    # print('armijo condition not met')
                    stepsize /= 2
                    B_tilde = B - stepsize * projG            
                    T = tensor_train(B_tilde, rank=tt_rank)
                    B_new = tt_to_tensor(T)
                    P = self.TenIsing.CondProb(W, B_new, invtemp=invtemp)
                    NLL = self.TenIsing.Pseudo_NLL(W, B_new, P=P, invtemp=invtemp, q=q)  

                step_hist.append(stepsize)
            else:
                # descent along the projected gradient direction             
                B_tilde = B - beta * projG
                
                # Step III: retraction
                T = tensor_train(B_tilde, rank=tt_rank)
                B_new = tt_to_tensor(T)
                step_hist.append(beta)
            
            delta = np.sqrt(((B_new - B) ** 2).sum() / (B ** 2).sum())
            B = np.copy(B_new)
            l += 1
            if self.true_B is not None:
                RSE_B = RSE(B, self.true_B)
                RSE_hist.append(RSE_B)

            # evaluate negative log pseudo-likelihood
            P = self.TenIsing.CondProb(W, B, invtemp=invtemp)
            NLL = self.TenIsing.Pseudo_NLL(W, B, P=P, invtemp=invtemp, q=q)
            NLL_hist.append(NLL)

        if verbose:
            print(f"RGrad terminates at iter {l}.")

        # compute the NLL at the true parameter, if available
        self.NLL_true = None
        if not self.true_B is None and not self.true_temp is None:
            P_true = self.TenIsing.CondProb(W, self.true_B, invtemp=1 / self.true_temp)
            self.NLL_true = self.TenIsing.Pseudo_NLL(W, self.true_B, P=P_true, invtemp=1 / self.true_temp, q=q)

        # return model parameters & fitting history
        self.B = B
        self.P = P
        self.T = T
        self.l = l
        self.projected_grad = projected_grad
        self.invtemp = invtemp
        self.temp = 1 / invtemp
        self.NLL_hist = NLL_hist
        self.RSE_hist = RSE_hist
        self.step_hist = step_hist

        # compute model BIC
        n_param = 0
        for k in range(K):
            n_param += d[k] * tt_rank[k] * tt_rank[k + 1]
            if k < K - 1:
                n_param -= tt_rank[k + 1] ** 2  # adjust for left-orthogonality
        self.BIC = 2 * NLL_hist[-1] + n_param * np.log(np.prod(d))
        self.AIC = 2 * NLL_hist[-1] + 2 * n_param

    def fit_TK(self, W, rank, q,
               config={"max_iter": 2000,
                       "thres": 1e-3,
                       "beta": 0.1,
                       "verbose": True,
                       "spikiness": 3.0,
                       "rse_est": 1.0,
                       "backtracking": False,
                       "armijo_threshold": 1e-4,
                       "projected_grad": False}):
        '''
        Fit low Tucker rank MPLE for Ising model (replicating Cai et al. (2022 JASA))
        params:
            W: K-mode binary tensor (values from {-1,1})
            rank: rank of parameter tensor
            config: algorithm configuration (e.g. maximum iteration, convergence threshold, etc.)
        '''

        d = W.shape
        K = W.ndim
        if K < 3:
            raise TypeError("Input tensor should contain at least three modes!")
        tk_rank = list(rank)  # tucker rank

        # ----- Initialization ----- #
        if "seed" in config:
            np.random.seed(config['seed'])
        C, U = tucker(W + np.random.normal(loc=0, scale=0.2, size=d), rank=tk_rank)
        B = tucker_to_tensor((C, U))
        invtemp = invtemp = 1 / self.TenIsing.g_params['temp']

        # evaluate initial conditional probability
        P = self.TenIsing.CondProb(W, B, invtemp=invtemp)
        NLL = self.TenIsing.Pseudo_NLL(W, B, P=P, invtemp=invtemp, q=q)
        NLL_hist = [NLL]

        # ----- Riemannian Gradient Descent ----- #
        beta = 0.1 if "beta" not in config else config["beta"]  # step size
        l, delta = 1, 1
        l_max = 500 if "max_iter" not in config else config["max_iter"]
        threshold = 1e-3 if "thres" not in config else config["thres"]
        verbose = False if "verbose" not in config else config["verbose"]
        print_frequency = 100 if "print_frequency" not in config else config["print_frequency"]
        spikiness = 2.0 if "spikiness" not in config else config["spikiness"]
        rse_est = 1.0 if "rse_est" not in config else config["rse_est"]
        backtracking = False if "backtracking" not in config else config['backtracking']
        armijo_threshold = 1e-4 if "armijo_threshold" not in config else config['armijo_threshold']  
        projected_grad = False if "projected_grad" not in config else config["projected_grad"]
        RSE_hist = []
        step_hist = []
        
        while (l <= l_max) and (delta > threshold):
            if l % 100 == 0 and l > 0 and verbose:
                print(f"iter {l}...")

            # Step I: compute vanilla gradient
            G = self.TenIsing.Grad(W, B, P=P, invtemp=invtemp, q=q)

            if projected_grad:
                projG = G
            else:
                # Step II: tangent space projected gradient descent
                UT = [mat.T for mat in U]
                G_core = tucker_to_tensor((G, UT))
                projG = tucker_to_tensor((G_core, U))
        
                for k in range(K):
                    Uk = copy.deepcopy(U)
                    Uk[k] = np.eye(d[k])
                    UkT = [mat.T for mat in Uk]
                    C_k = unfold(C, k)
                    C_pinv = C_k.T @ inv(C_k @ C_k.T)
                    Uk[k] = (unfold(tucker_to_tensor((G, UkT)), k) - U[k] @ unfold(G_core, k)) @ C_pinv
                    projG += tucker_to_tensor((C, Uk))
            
            if backtracking:
                stepsize = beta
                # descent along the projected gradient direction
                B_tilde = B - stepsize * projG
    
                # Step III: retraction
                C, U = tucker(B_tilde, rank=tk_rank)
                B_new = tucker_to_tensor((C, U))
                
                prev_NLL = NLL_hist[-1]
                P = self.TenIsing.CondProb(W, B_new, invtemp=invtemp)
                NLL = self.TenIsing.Pseudo_NLL(W, B_new, P=P, invtemp=invtemp, q=q)
                backtracking_threshold = armijo_threshold * (np.linalg.norm(projG) ** 2)                
                
                while ((prev_NLL - NLL) < stepsize * backtracking_threshold) and (stepsize >= 1e-3):
                    stepsize /= 2
                    B_tilde = B - stepsize * projG
                    C, U = tucker(B_tilde, rank=tk_rank)
                    B_new = tucker_to_tensor((C, U))
                    P = self.TenIsing.CondProb(W, B_new, invtemp=invtemp)
                    NLL = self.TenIsing.Pseudo_NLL(W, B_new, P=P, invtemp=invtemp, q=q)  

                step_hist.append(stepsize)
            else:
                # descent along the projected gradient direction
                B_tilde = B - beta * projG
    
                # Step III: retraction
                C, U = tucker(B_tilde, rank=tk_rank)
                B_new = tucker_to_tensor((C, U))
                step_hist.append(beta)

            # track convergence
            delta = np.sqrt(((B_new - B) ** 2).sum() / (B ** 2).sum())
            B = np.copy(B_new)
            l += 1
            if self.true_B is not None:
                RSE_B = RSE(B, self.true_B)
                RSE_hist.append(RSE_B)

            # evaluate negative log pseudo-likelihood
            P = self.TenIsing.CondProb(W, B, invtemp=invtemp)
            NLL = self.TenIsing.Pseudo_NLL(W, B, P=P, invtemp=invtemp, q=q)
            NLL_hist.append(NLL)

        if verbose:
            print(f"RGrad terminates at iter {l}.")

        # compute the NLL at the true parameter, if available
        self.NLL_true = None
        if not self.true_B is None and not self.true_temp is None:
            P_true = self.TenIsing.CondProb(W, self.true_B, invtemp=1 / self.true_temp)
            self.NLL_true = self.TenIsing.Pseudo_NLL(W, self.true_B, P=P_true, invtemp=1 / self.true_temp, q=q)

        # return model parameters & fitting history
        self.B = B
        self.P = P
        self.l = l
        self.projected_grad = projected_grad
        self.invtemp = invtemp
        self.temp = 1 / invtemp
        self.NLL_hist = NLL_hist
        self.RSE_hist = RSE_hist
        self.step_hist = step_hist

        # compute model BIC
        n_param = np.prod(np.array(tk_rank))
        for k in range(K):
            n_param += (d[k] - tk_rank[k]) * tk_rank[k]
        self.BIC = 2 * NLL_hist[-1] + n_param * np.log(np.prod(d))
        self.AIC = 2 * NLL_hist[-1] + 2 * n_param