import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.risk import get_tau_hat_agent
import numpy as np

class RISKRNNQRAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RISKRNNQRAgent, self).__init__()
        self.args = args
        self.cumulative_density = th.tensor(
        (2 * np.arange(self.args.n_target_quantiles) + 1) / (2.0 * self.args.n_target_quantiles), dtype=th.float32).view(1, -1).to(
        self.args.device)
        self.quantile_embed_dim = args.quantile_embed_dim
        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_approx_quantiles = args.n_approx_quantiles

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * args.n_quantiles)

    def init_hidden(self):

        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        logits = self.fc2(h)
        shape = h.shape
        batch_size = shape[0]
        n_rnd_quantiles = self.n_target_quantiles
        q_vals = logits.view(batch_size, self.args.n_actions, n_rnd_quantiles)



        if getattr(self.args, "sort_quantiles", True):
            sorted_qvals, indices = th.sort(q_vals, dim=2)
            if getattr(self.args, "masked_out_quantiles", True):
                risk_type = self.args.risk_type
                risk_param = self.args.risk_param
                if risk_type == "cvar":
                    masks = (self.cumulative_density <= risk_param).float()
                    masks = masks.view(1, 1, n_rnd_quantiles)
                    if masks.shape[0] < sorted_qvals.shape[0]: 
                        masks = masks.repeat(sorted_qvals.shape[0], 1, 1)
                    sorted_qvals = sorted_qvals * masks
            return sorted_qvals, h
        else:
            return q_vals, h
