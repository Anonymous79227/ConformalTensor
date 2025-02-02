from LRIsing import *
from tensorly.decomposition import tensor_train, tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tt_tensor import tt_to_tensor, tt_to_unfolded


class SimIsing:
    '''
    Simulate from low-rank Ising Model
    '''

    def __init__(self, d=(100, 100, 100), g_func="product", h_func="logit", g_params=None, h_params=None, nb="1-nb"):
        self.d = d
        self.K = len(d)
        self.TenIsing = TensorIsing(g_func=g_func, h_func=h_func, g_params=g_params, h_params=h_params, nb=nb)
        self.B = None

    def sim_B(self, rank=(3, 3), method="tensor-train", seed=42, tensor_type="checkerbox"):
        '''
        Randomly simulate low-rank tensor parameter B

        params:
            rank: simulate rank for B
            method: rank notion of B, values from {"tensor-train", "tucker"}
            seed: simulation random seed
            tensor_type: type of the random tensor B, values from {"checkerbox", "random"}
        '''
        np.random.seed(seed)
        if tensor_type == "random":
            if method == "tensor-train":
                if not len(rank) == self.K - 1:
                    raise Exception("Rank for the tensor-train tensor should have length equals to the mode minus one.")

                T = []
                tt_rank = [1] + list(rank) + [1]
                for k in range(self.K):
                    T.append(np.random.normal(size=(tt_rank[k], self.d[k], tt_rank[k + 1])))
                B = tt_to_tensor(T)
            elif method == "tucker":
                if not len(rank) == self.K:
                    raise Exception("Rank for the tucker tensor should have equal length as the tensor mode.")
                core_tensor = np.random.normal(size=rank)
                factor_tensor = []
                for k in range(self.K):
                    factor_tensor.append(np.random.normal(size=(self.d[k], rank[k])))
                B = tucker_to_tensor((core_tensor, factor_tensor))
        elif tensor_type == "checkerbox":
            # generate checkerbox type tensor with tensor block model (TBM)
            if not len(rank) == self.K:
                # raise Exception("When generating checkerbox type of tensor, please provide a rank with the length equal to the number of modes.")
                rank = list(rank) + [rank[-1]]
            else:
                rank = list(rank)

            core_tensor = np.random.normal(loc=1.0, scale=0.5, size=rank)
            mixture = np.random.uniform(size=rank) <= 0.5
            core_tensor = mixture * core_tensor + (1-mixture) * np.random.normal(loc=-1.0, scale=0.5, size=rank)
            factor_tensor = []
            for k in range(self.K):
                # generate binary cluster indicator
                U = np.zeros((self.d[k], rank[k]))
                # clust_prop = np.random.dirichlet(alpha=np.ones(rank[k]))
                clust_prop = np.ones(rank[k]) / rank[k]
                start_index = 0
                for cl in range(rank[k] - 1):
                    cl_size = int(np.floor(self.d[k] * clust_prop[cl]))
                    if cl_size > 0:
                        U[start_index:(start_index + cl_size), cl] = 1
                    start_index += cl_size
                U[start_index:, -1] = 1
                factor_tensor.append(U)
            mean_tensor = tucker_to_tensor((core_tensor, factor_tensor))
            B = mean_tensor + np.random.normal(scale=0.2, size=self.d)

        self.B = 2 * B / np.amax(np.abs(B))  # restrict the maximum magnitude of B to be 2
        self.B_method = method
        self.B_rank = rank
        return self.B

    def sim_sample(self, method="MH", method_params=None, seed=42):
        if self.B is None:
            raise Exception("Please simulate the underlying true tensor parameter first!")

        np.random.seed(seed)
        if method == "MH":
            sample = self.one_spin_MCMC(**method_params)
        # elif method == "SW":
        #     sample = self.sw_MCMC(**method_params)
        elif method == "Block-Gibbs":
            sample = self.block_Gibbs(**method_params)

        return sample

    def one_spin_MCMC(self, max_iter=10000, burn_in=5000, N_sample=10):
        '''
        Standard Metropolis Algorithm
        '''
        t = 0
        sample = []

        # generate initial sample with independent Bernoulli
        T = 2 * self.TenIsing.h(self.B)
        P = np.exp(T) / (1 + np.exp(T))
        if not self.TenIsing.g_func == "zero":
            W = (np.random.uniform(size=self.d) <= P).astype(int)
            W[W == 0] = -1
        else:
            for _ in range(N_sample):
                W = (np.random.uniform(size=self.d) <= P).astype(int)
                W = 2 * W - 1
                sample.append(W)
            return np.array(sample)

        # Metropolis-Hastings Algorithm
        sample_freq = (max_iter - burn_in) // N_sample
        arr = np.array([-1, 0, 1])
        combs = itertools.product(*([arr] * self.K))
        dy = [np.array(comb) for comb in combs]
        dx = [dd for dd in dy if not (dd == 0).all()]

        while t < max_iter:
            # propose a new sample
            random_index = np.random.choice(np.prod(self.d))
            index = np.unravel_index(random_index, self.d)
            W_prop = np.copy(W)
            W_prop[index] = -1 * W_prop[index]  # single spin flip proposal

            # evaluate likelihood ratio
            ll_exp = 0.5 * T[index]  # exponent of the likelihood ratio
            if self.TenIsing.nb in ["1-nb", "1-nb-spatial"]:
                neighbor_mode = self.K if self.TenIsing.nb == "1-nb" else self.K - 1
                for k in range(neighbor_mode):
                    if index[k] > 0:
                        loc = np.copy(index)
                        loc[k] -= 1
                        loc = tuple(loc)
                        ll_exp += self.TenIsing.g(B[index], B[loc]) * W_prop[loc]
                    if index[k] < self.d[k] - 1:
                        loc = np.copy(index)
                        loc[k] += 1
                        loc = tuple(loc)
                        ll_exp += self.TenIsing.g(B[index], B[loc]) * W_prop[loc]
            elif self.TenIsing.nb == "cube-nb":
                for dd in dx:
                    loc = np.array(index) + dd
                    loc = tuple(loc)
                    ll_exp += W_prop[loc] * self.TenIsing.g(B[index], B[loc])
            ll_exp = (W_prop[index] - W[index]) * ll_exp

            # accept or reject the proposed tensor
            u = np.random.uniform()
            if ll_exp >= 0:
                W = np.copy(W_prop)
            elif u <= np.exp(ll_exp):
                W = np.copy(W_prop)

                # take the sample if needed
            if t >= burn_in and (t - burn_in) % sample_freq == 0:
                sample.append(W)

            t += 1
        return np.array(sample)

    def block_Gibbs(self, max_iter=10000, burn_in=5000, N_sample=10):
        '''
        Sample from Ising model with Block Gibbs Algorithm
        '''
        t = 0
        sample = []
        sample_freq = (max_iter - burn_in) // N_sample

        # generate initial sample with independent Bernoulli
        B = self.B
        T = 2 * self.TenIsing.h(B)
        P = np.exp(T) / (1 + np.exp(T))
        if not self.TenIsing.g_func == "zero":
            W = (np.random.uniform(size=self.d) <= P).astype(int)
            W[W == 0] = -1
        else:
            for _ in range(N_sample):
                W = (np.random.uniform(size=self.d) <= P).astype(int)
                W = 2 * W - 1
                sample.append(W)
            return np.array(sample)

        # block-gibbs sampling
        if self.TenIsing.nb in ["1-nb", "1-nb-spatial"]:
            # gibbs sampling block structure
            blocks = np.zeros(self.d)
            for index in np.ndindex(self.d):
                if sum(index) % 2 == 0:
                    blocks[index] = 1

            while t < max_iter:
                # sample block-1
                P = self.TenIsing.CondProb(W, B)
                W0 = (np.random.uniform(size=self.d) <= P).astype(int)
                W0 = 2 * W0 - 1
                W = blocks * W0 + (1 - blocks) * W

                # sample block-0
                P = self.TenIsing.CondProb(W, B)
                W0 = (np.random.uniform(size=self.d) <= P).astype(int)
                W0 = 2 * W0 - 1
                W = (1 - blocks) * W0 + blocks * W

                # take the sample if needed
                if t >= burn_in and (t - burn_in) % sample_freq == 0:
                    sample.append(W)

                t += 1
            return np.array(sample)

        elif self.TenIsing.nb == "cube-nb":
            pass


class SimData:
    '''
    Simulate Tensor Data
    '''

    def __init__(self, d=(100, 100, 100)):
        self.d = d
        self.K = len(d)

    def sim_X(self, rank=(3, 3), method="tensor-train", seed=42, noise="constant", sigma=1.0, true_B=None, SNR=2.0,
              tensor_type="checkerbox"):
        '''
        Simulate fully-observed data tensor by additive model.

        params:
            rank: rank of the noiseless tensor
            method: method for simulating the noiseless tensor, values from {"tensor-train", "tucker"}
            seed: random seed
            noise: method for generating noise tensor, values from {"constant", "adversarial"}
            sigma: standard deviation of the noise tensor, if noise == "constant"
            SNR: signal-to-noise ratio, i.e. the ratio of the max-norm of the true tensor and noise tensor
            tensor_type: type of simulated tensor, values from {"random", "checkerbox"}
        '''

        if method == "tensor_train" and not len(rank) == self.K - 1:
            raise Exception("The length of rank should equal to number of modes minus 1 while using tensor train rank!")

        if method == "tucker" and not len(rank) == self.K - 1:
            raise Exception("The length of rank should equal to number of modes while using tucker rank!")

        np.random.seed(seed)
        if tensor_type == "random":
            if method == "tensor-train":
                T = []
                tt_rank = [1] + list(rank) + [1]
                for k in range(self.K):
                    T.append(np.random.normal(loc=0.0, scale=1.0, size=(tt_rank[k], self.d[k], tt_rank[k + 1])))
                X_true = tt_to_tensor(T)
            elif method == "tucker":
                core_tensor = np.random.normal(loc=0.0, scale=1.0, size=rank)
                factor_matrices = []
                for k in range(self.K):
                    factor_matrices.append(np.random.normal(size=(self.d[k], rank[k])))
                X_true = tucker_to_tensor((core_tensor, factor_matrices))
        elif tensor_type == "checkerbox":
            # generate checkerbox type tensor with tensor block model (TBM)
            if not len(rank) == self.K:
                # raise Exception("When generating checkerbox type of tensor, please provide a rank with the length equal to the number of modes.")
                rank = list(rank) + [rank[-1]]
            else:
                rank = list(rank)

            core_tensor = np.random.normal(scale=0.5, size=rank)
            factor_tensor = []
            for k in range(self.K):
                # generate binary cluster indicator
                U = np.zeros((self.d[k], rank[k]))
                clust_prop = np.random.dirichlet(alpha=np.ones(rank[k]))
                start_index = 0
                for cl in range(rank[k] - 1):
                    cl_size = int(np.floor(self.d[k] * clust_prop[cl]))
                    if cl_size > 0:
                        U[start_index:(start_index + cl_size), cl] = 1
                    start_index += cl_size
                U[start_index:, -1] = 1
                factor_tensor.append(U)
            X_true = tucker_to_tensor((core_tensor, factor_tensor))
        X_true = 2 * X_true / np.amax(np.abs(X_true))  # rescale the magnitude of X_true to have max-norm 2

        # simulate noise tensor
        if noise == "constant":
            noise_tensor = np.random.normal(scale=sigma, size=self.d)
        elif noise == "adversarial":
            if true_B is None:
                raise Exception("Please provide the tensor parameter B under the adversarial noise setting!")
            P = np.exp(true_B) / (1 + np.exp(true_B))
            noise_tensor = np.random.normal(scale=1 / (2 * P))
        noise_tensor = noise_tensor * (2 / (SNR * np.amax(np.abs(noise_tensor))))

        return X_true, noise_tensor