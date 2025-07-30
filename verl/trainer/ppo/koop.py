import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal
import ot
from tqdm import tqdm

device = torch.device("cuda")

class KMD:
    def __init__(self, data, rank=None, lamb=0.,
            backend='numpy', 
            device='cpu',
            verbose=False
        ):
        """
        Base class for Koopman Mode Decomposition.
        
        Parameters:
        -----------
        data : array-like
            Input data to decompose
        backend : str, optional
            Computational backend to use ('numpy', 'pytorch', or 'cupy')
        device : str, optional
            Device to use for computation when using PyTorch backend
        verbose : bool, optional
            Whether to print verbose output
        send_to_cpu : bool, optional
            Whether to send results to CPU after computation
        """
        self.data = data
        self.backend = backend
        self.device = device
        self.rank = rank
        self.lamb = lamb
        self.verbose = verbose
        self.A_v, self.E, self.S, self.V, self.Vh, self.W, self.W_prime = None, None, None, None, None, None, None
 
        # TODO: Backends specification
        # if backend == 'numpy':
        #     self.xp = np
        # elif backend == 'cupy':
        #     self.xp = cp
        # elif backend == 'pytorch':
        #     self.xp = torch
        # else:
        #     raise ValueError(f"Unsupported backend: {backend}. Choose from 'numpy', 'pytorch', or 'cupy'")
        

    def init_data(self):
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data).to(self.device)
        # print("self.data", self.data.shape)
        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(0)  # Add trial dimension (1, timesteps, features)
        elif self.data.ndim == 3:
            pass  # Already in the correct format (trials, timesteps, features)
        else:
            raise ValueError(f"Invalid data shape: {self.data.shape}. Expected 2D (samples, features) or 3D (trials, samples, features)")
        
        self.n_trials, self.n_timesteps, self.n_features = self.data.shape
        
    def embed(self):
            
        raise NotImplementedError
        
    
    def compute_svd(self):
        """
        Compute the Singular Value Decomposition of the embedded data.
        
        Parameters:
        -----------
        rank : int, optional
            Truncation rank for SVD. If None, full SVD is computed.
            
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Flatten embedding across trials if 3D
        E = self.E.reshape(self.E.shape[0] * self.E.shape[1], self.E.shape[2]) if self.E.ndim == 3 else self.E
        
        U, self.S, self.Vh = torch.linalg.svd(E.T, full_matrices=False)
        self.V = self.Vh.T


    def reduced_rank(self):

        raise NotImplementedError

    
    def compute_dmd(self):
        """
        Compute the Dynamic Mode Decomposition.
        """
        if self.verbose:
            print("Computing DMD")

        if self.lamb != 0:
            regularization = self.lamb * torch.eye(self.rank).to(self.device)
        else:
            regularization = torch.zeros(self.rank, self.rank).to(self.device)

        self.A_v = (torch.linalg.inv(self.W.T @ self.W + regularization) @ self.W.T @ self.W_prime).T
        

    def fit(self):
        self.init_data()
        self.embed()
        self.compute_svd()
        if self.rank is not None:
            self.reduced_rank()
        self.compute_dmd()


class KoopSTD(KMD):
    def __init__(self, data, rank=15, lamb=0., win_len=8, hop_size=1,
            backend='numpy', 
            device='cpu',
            verbose=False
        ):
        super().__init__(data, rank, lamb, backend, device, verbose)
        self.win_len = win_len
        self.hop_size = hop_size
        self.rank = rank
        self.lamb = lamb

        if data.shape[0] < self.rank:
            padding_size = self.rank - data.shape[0] + 1
            last_dim_data = data[-1:].repeat(padding_size, 1)
            data = torch.cat([data, last_dim_data], dim=0)
        
        self.data = data

        self.backend = backend
        self.device = device
        self.verbose = verbose 

    def embed(self):
        # multivariate STFT
        stfts = []
        for i in range(self.n_features):
            stft = torch.stft(self.data[:, :, i], n_fft=self.win_len, hop_length=self.hop_size, return_complex=True, normalized=True)
            stfts.append(stft)
        stfts = torch.stack(stfts, dim=1)
        trial, _, _, time_frames = stfts.shape
        stfts = stfts.view(trial, time_frames, -1).real.to(torch.float32)
 
        self.E = stfts.to(self.device)
        if self.n_trials == 1:
            self.E = self.E.squeeze(0)

    def compute_svd(self):
        if self.E.ndim == 3: 
            E = self.E.reshape(self.E.shape[0] * self.E.shape[1], self.E.shape[2])
        else:
            E = self.E
        
        _, self.S, self.V = torch.linalg.svd(E.T, full_matrices=False)

        if E.shape[0] < E.shape[1]:  # T < N
            E = E[:, :E.shape[0]]
        self.E_minus = E[:-1]
        self.E_plus = E[1:]

    def reduced_rank(self):
        M = torch.matmul(self.E_minus.T.conj(), self.E_minus)
        N = torch.matmul(self.E_minus.T.conj(), self.E_plus)
        O = torch.matmul(self.E_plus.T.conj(), self.E_plus)
        
        egvalues, egvectors = self.S, self.V
        residuals = []
        for j, (eigenvalue, eigenvector) in enumerate(zip(egvalues, egvectors.T)):
            residual = self.compute_residuals(M, N, O, eigenvalue, eigenvector)
            residuals.append(residual)
        residuals = torch.tensor(residuals)
        topk_indices = torch.topk(-residuals, self.rank, largest=False).indices
        V = self.V.T

        if self.n_trials > 1:
            V = V.reshape(self.E.shape)
            V_rank = V[:, :, topk_indices]
            new_shape = (self.E.shape[0] * (self.E.shape[1] - 1), self.rank)
            V_minus_rank = V_rank[:, :-1].reshape(new_shape)
            V_plus_rank = V_rank[:, 1:].reshape(new_shape)
        else:
            V_rank = V[:, topk_indices]
            V_minus_rank = V_rank[:-1]
            V_plus_rank = V_rank[1:]
        
        self.W = V_minus_rank
        self.W_prime = V_plus_rank
        
    def residual_dmd(self):
        """
        Standard implementation of ResDMD, however, for the sake of efficiency, 
        we don't recommend it in large dataset comparison.
        """
        self.Vt_minus = self.V[:-1]
        self.Vt_plus = self.V[1:]

        X_X = torch.matmul(self.Vt_plus.T.conj(), self.Vt_plus)
        X_Y = torch.matmul(self.Vt_plus.T.conj(), self.Vt_minus)
        Y_Y = torch.matmul(self.Vt_minus.T.conj(), self.Vt_minus)

        A_full = torch.linalg.inv(self.Vt_minus.T @ self.Vt_minus) @ self.Vt_minus.T @ self.Vt_plus
        _, egvalues, egvectors = torch.linalg.svd(A_full, full_matrices=True)
        residuals = []
        for j, (eigenvalue, eigenvector) in enumerate(zip(egvalues, egvectors.T)):
            residual = self.compute_residuals(X_X, X_Y, Y_Y, eigenvalue, eigenvector)
            residuals.append(residual)
        residuals = torch.tensor(residuals)
        topk_indices = torch.topk(-residuals, self.rank, largest=False).indices
        self.A_v = egvalues[topk_indices].view(-1,1)  # direct eigenvalues
    
    def compute_residuals(self, X_X, X_Y, Y_Y, eigenvalue, eigenvector):
        numerator = torch.matmul(
            eigenvector.conj(),
            torch.matmul(
                Y_Y - eigenvalue * X_Y - torch.conj(eigenvalue) * X_Y.T.conj() + (eigenvalue.abs() ** 2) * X_X,
                eigenvector
            )
        )
        denominator = torch.matmul(eigenvector.conj(), torch.matmul(X_X, eigenvector))

        residual = torch.sqrt(torch.abs(numerator) / torch.abs(denominator))
        return residual
    

class HAVOK(KMD):
    def __init__(self, data, rank=15, lamb=0., n_delays=8, delay_interval=1,
            backend='numpy', 
            device='cpu',
            verbose=False
        ):
        """
            The DMD part implementation of HAVOK-based DSA (Ostrow et al., 2024). 
            Adapted from https://github.com/mitchellostrow/DSA.
        """
        super().__init__(data, rank, lamb, backend, device, verbose)
        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.lamb = lamb
        self.data = data

    def embed(self):
        # Hankel (delay) Embedding
        if self.data.shape[int(self.data.ndim==3)] - (self.n_delays - 1) * self.delay_interval < 1:
            raise ValueError("The number of delays is too large for the number of time points in the data!")
        
        if self.data.ndim == 3:
            embedding = torch.zeros((self.data.shape[0], self.data.shape[1] - (self.n_delays - 1) * self.delay_interval, self.data.shape[2] * self.n_delays))
        else:
            embedding = torch.zeros((self.data.shape[0] - (self.n_delays - 1) * self.delay_interval, self.data.shape[1] * self.n_delays))
        
        for d in range(self.n_delays):
            index = (self.n_delays - 1 - d) * self.delay_interval
            ddelay = d * self.delay_interval

            if self.data.ndim == 3:
                ddata = d * self.data.shape[2]
                embedding[:,:, ddata: ddata + self.data.shape[2]] = self.data[:,index:self.data.shape[1] - ddelay]
            else:
                ddata = d * self.data.shape[1]
                embedding[:, ddata:ddata + self.data.shape[1]] = self.data[index:self.data.shape[0] - ddelay]
    
        self.E = embedding.to(self.device)
        if self.n_trials == 1:
            self.E = self.E.squeeze(0)

    def reduced_rank(self):
        if self.n_trials > 1:
            V = self.V.reshape(self.E.shape)
            new_shape = (self.E.shape[0] * (self.E.shape[1] - 1), self.E.shape[2])
            V_minus = V[:, :-1].reshape(new_shape)
            V_plus = V[:, 1:].reshape(new_shape)
        else:
            V_minus = self.V[:-1]
            V_plus = self.V[1:]
        
        self.W = V_minus[:, :self.rank]
        self.W_prime = V_plus[:, :self.rank]


class KoopDSA:
    def __init__(self, data: list, kmd_method: str='koopstd', kmd_params: dict=None, device: str='cuda'):
        self.data = data
        self.kmd_params = kmd_params or {}
        self.kmd_params['device'] = device

        # Initialize KMD models
        self.kmds = []
        for d in self.data:
            if kmd_method == 'koopstd':
                self.kmds.append(KoopSTD(d, **self.kmd_params))
            elif kmd_method == 'havok':
                self.kmds.append(HAVOK(d, **self.kmd_params))

    def fit_dmds(self):
        """
        Fit DMDS models without calculating distance matrix.
        
        Returns:
        --------
        """
        # print("Fitting KMD models")
        for kmd in self.kmds:
            kmd.fit()


    
class DistCalculator:
    def __init__(self, dist_type: str):
        self.dist_type = dist_type
        if dist_type == "js":
            self.dist_func = self.js_divergence
        elif dist_type == "ws":
            self.dist_func = self.wasserstein_distance
        elif dist_type == "l2":
            self.dist_func = self.l2_distance
        elif dist_type == "cos":
            self.dist_func = self.cosine_distance
        elif dist_type == "corr":
            self.dist_func = self.linear_correlation

    def compute_loo_disperse(self, spectrums: dict):  # {key1: [b1_r1, b1_r2, ..., b1_r6], key2: [b2_r1, b2_r2, ..., b2_r6], ...}
        one_batch_dispersive_rewards = {}
        for key, value in spectrums.items():
            # print(key, len(value))
            one_question_loo_dispersive_rewards = []
            for i in range(len(value)):
                leave_one_total_distance = 0.0
                for j in range(len(value)):
                    if i == j:
                        continue
                    leave_one_total_distance += self.dist_func(value[i], value[j])
                average_distance = leave_one_total_distance / (len(value) - 1)
                one_question_loo_dispersive_rewards.append(average_distance.item())
            one_batch_dispersive_rewards[key] = one_question_loo_dispersive_rewards
        return one_batch_dispersive_rewards

    def spectra_distance(self, m1: torch.Tensor, m2: torch.Tensor):
        return self.dist_func(m1, m2)

    def js_divergence(self, m1: torch.Tensor, m2: torch.Tensor):
        """
        Jensen-Shannon divergence (differentiable version).
        """
        # Flatten & normalize to probability distributions
        p = m1.view(-1)
        q = m2.view(-1)
        eps = 1e-8

        p = p / (p.sum() + eps)
        q = q / (q.sum() + eps)

        m = 0.5 * (p + q)

        # Add eps to prevent log(0)
        p = p + eps
        q = q + eps
        m = m + eps

        # All operations are differentiable in PyTorch
        kl_pm = torch.sum(p * torch.log(p / m))
        kl_qm = torch.sum(q * torch.log(q / m))

        js = 0.5 * (kl_pm + kl_qm)
        normalized_js = js / torch.log(torch.tensor(2.0, device=js.device, dtype=js.dtype))
        
        return normalized_js

    def wasserstein_distance(self, m1: torch.Tensor, m2: torch.Tensor):
        """
        Wasserstein-p Distance (p=1 default), implementation of PyOT.
        """
        p = 1
        m1 = m1.view(-1, 1)
        m2 = m2.view(-1, 1)
        a = torch.ones(m1.shape[0], device=m1.device) / m1.shape[0]
        b = torch.ones(m2.shape[0], device=m2.device) / m2.shape[0]
        M = ot.dist(m1, m2, metric='euclidean')
        if p != 1:
            M = M ** (p / 2)
        dist = ot.emd2(a, b, M) 
        if p != 1:
            dist = dist ** (1.0 / p)
        return dist

    def l2_distance(self, m1: torch.Tensor, m2: torch.Tensor):
        """
        L2 distance.
        """
        # Flatten tensors to 1D vectors for proper L2 distance calculation
        m1_flat = m1.view(-1)
        m2_flat = m2.view(-1)
        
        # Compute L2 distance (Euclidean distance)
        l2_dist = torch.norm(m1_flat - m2_flat, p=2)
        return l2_dist.item()

    def cosine_distance(self, m1: torch.Tensor, m2: torch.Tensor):
        """
        Cosine distance (1 - cosine similarity).
        """
        # Flatten tensors to 1D vectors for proper cosine distance calculation
        m1_flat = m1.view(-1)
        m2_flat = m2.view(-1)
        
        # Compute cosine similarity and convert to distance
        cosine_sim = F.cosine_similarity(m1_flat, m2_flat, dim=0)
        cosine_dist = 1.0 - cosine_sim
        return cosine_dist.item()

    def linear_correlation(self, m1: torch.Tensor, m2: torch.Tensor):
        """
        Linear correlation.
        """
        corr = torch.corrcoef(torch.stack([m1.view(-1), m2.view(-1)]))[0,1]
        return 1 - corr.item()


class DynamicDispersiveReward:
    def __init__(self, l, s, r, lamb: float = 0.001, dist: str = "ws", mode: Literal["rollout", "prompt"] = "rollout", device: str = "cuda"):
        super(DynamicDispersiveReward, self).__init__()
        self.koopstd_params = {
            'win_len': l,
            'hop_size': s,
            'rank': r,
            'lamb': lamb
        }
        self.dist_calculator = DistCalculator(dist)
        self.mode = mode
        self.device = device

    def reward(self, batch_hidden_state_rollouts: list):
        if self.mode == "rollout":
            return self.reward_rollout_mode_disperse(batch_hidden_state_rollouts)
        elif self.mode == "sample":
            return self.reward_sample_mode_disperse(batch_hidden_state_rollouts)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def reward_sample_mode_disperse(self, batch_hidden_state_rollouts: list):
        """
        Multi-prompts sample mode: batch_hidden_state_rollouts is a list of lists of hidden states, 
                                   each hidden state is a torch tensor of shape (seq_len, hidden_dim).
        We directly compute the sum of the distance between all pairs of spectrums for each sample
        Output: a list of dispersive rewards of each sample. Each reward is a float.
        """
        sample_dispersive_rewards = []
        for sample_rollouts in batch_hidden_state_rollouts:
            spectrums = self.get_one_prompt_multi_rollouts_spectrums(sample_rollouts)
            sample_dispersive_rewards.append(self.compute_one_sample_multi_rollouts_sum_distance(spectrums).item())
        return sample_dispersive_rewards

    def reward_rollout_mode_disperse(self, batch_hidden_state_rollouts: list):
        """
        One-prompt rollout mode: batch_hidden_state_rollouts is a list with only one list of hidden states, 
                                 each hidden state is a torch tensor of shape (seq_len, hidden_dim).
        A leave-one-out strategy is used to compute the dispersive reward, for each rollout, it is defined as the average distance of all other rollouts to it.
        Output: a list of dispersive rewards of each rollout.
        """
        one_sample_multi_rollouts = batch_hidden_state_rollouts
        spectrums = self.get_one_prompt_multi_rollouts_spectrums(one_sample_multi_rollouts)
        rollout_dispersive_rewards = []
        for i in range(len(spectrums)):
            leave_one_total_distance = 0.0
            for j in range(len(spectrums)):
                if i == j:
                    continue
                leave_one_total_distance += self.dist_calculator.spectra_distance(spectrums[i], spectrums[j])
            average_distance = leave_one_total_distance / (len(spectrums) - 1)
            rollout_dispersive_rewards.append(average_distance.item())
        return rollout_dispersive_rewards

    def compute_one_sample_multi_rollouts_sum_distance(self, spectrums: list):
        """
        Calculate the sum of the distance between all pairs of spectrums.
        """
        total_distance = 0.0
        for i in range(len(spectrums)):
            for j in range(i + 1, len(spectrums)):
                distance = self.dist_calculator.spectra_distance(spectrums[i], spectrums[j])
                total_distance += distance  
        return total_distance

    def get_koopman_spectrums(self, hidden_state_rollouts: list):
        """
        Input: list of hidden states, each hidden state is a torch tensor of shape (seq_len, hidden_dim)
        Output: list of koopman spectrums, each spectrum is a torch tensor of shape (1, rank)
        """
        # TODO: parallel processing
        hidden_states = [hidden_state_rollouts[i].to(device) for i in range(len(hidden_state_rollouts))]
        koopstd = KoopDSA(
                data=hidden_states,
                kmd_method='koopstd',
                kmd_params=self.koopstd_params,
                device=self.device
            )
        koopstd.fit_dmds()
        spectrums = []
        for k in koopstd.kmds:
            A_v = k.A_v.float()
            spectrums.append(torch.svd(A_v).S.view(1, -1).to(device))
        return spectrums

