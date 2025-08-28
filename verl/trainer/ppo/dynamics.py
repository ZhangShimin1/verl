import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LearnableDictionary(nn.Module):
    def __init__(self, state_dim, koopman_dim):
        super(LearnableDictionary, self).__init__()
        self.state_dim = state_dim
        self.koopman_dim = koopman_dim
        self.dictionary = nn.Linear(state_dim, koopman_dim, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.dictionary(x))

def residual_loss(dict_net, batch_state_traj):
    snapshots = [batch_state_traj[:, :-1], batch_state_traj[:, 1:]]
    # B, T, D -> B * T, D
    psi_x = dict_net(snapshots[0]).reshape(-1, dict_net.koopman_dim)
    psi_y = dict_net(snapshots[1]).reshape(-1, dict_net.koopman_dim)
    psi_xT = psi_x.T
    # approximate K for a batch of state trajectories
    G_inv = torch.linalg.pinv(torch.matmul(psi_xT, psi_x))
    A = torch.matmul(psi_xT, psi_y)
    K = torch.matmul(G_inv, A)

    _, S, Vh = torch.linalg.svd(K)
    S_diag = torch.diag(S)

    psi_x_v = torch.matmul(psi_x, Vh)
    psi_x_v_k = torch.matmul(psi_x_v, S_diag)
    psi_y_v = torch.matmul(psi_y, Vh)

    J = torch.norm(psi_y_v - psi_x_v_k, p='fro')

    return J

def koopman_learning(hidden_states, koopman_dim):
    device, hidden_dim = hidden_states.device, hidden_states.shape[2]
    learnable_dict = LearnableDictionary(hidden_dim, koopman_dim).to(device)
    optimizer = torch.optim.Adam(learnable_dict.parameters(), lr=0.001)

    dataset = TensorDataset(hidden_states)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # train the dictionary network via Residual DMD
    best_net, min_loss, losses = None, np.inf, []
    for epoch in range(30):
        for idx, hs in enumerate(dataloader):
            optimizer.zero_grad()
            loss = residual_loss(learnable_dict, hs[0])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if np.mean(losses) < min_loss:
            min_loss = np.mean(losses)
            best_net = learnable_dict
    dict_matrix = best_net.dictionary.weight.detach()
    # Save the dict_matrix for later direct matrix multiplication
    torch.save(dict_matrix, "koopman_dict_matrix.pt")
    print("The Learnable Dictionary for Koopman Operator has been saved in koopman_dict_matrix.pt...")

    # To reuse this matrix for direct matrix multiplication (without loading the network):
    # loaded_matrix = torch.load("koopman_dict_matrix.pt")  # shape: [koopman_dim, hidden_dim]
    # result = torch.matmul(input_tensor, loaded_matrix.t())  # input_tensor: [..., hidden_dim]
    spectral_spreads = []
    for hs in hidden_states:
        # hs: [seq_len, hidden_dim]
        psi_x = torch.tanh(torch.matmul(hs[:-1], dict_matrix.t()))
        psi_y = torch.tanh(torch.matmul(hs[1:], dict_matrix.t()))
        psi_xT = psi_x.transpose(0, 1)  # [koopman_dim, seq_len-1]

        G = torch.matmul(psi_xT, psi_x)  # [koopman_dim, koopman_dim]
        G_inv = torch.linalg.pinv(G)
        A = torch.matmul(psi_xT, psi_y)
        K = torch.matmul(G_inv, A)  # [koopman_dim, koopman_dim]

        _, S, _ = torch.linalg.svd(K, full_matrices=False)
        radius = torch.abs(S)
        spread = torch.var(radius)
        spectral_spreads.append(spread)

    return torch.stack(spectral_spreads, dim=0)

