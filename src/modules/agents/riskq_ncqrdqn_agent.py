import torch as th
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from utils.risk import get_tau_hat_agent


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class RISKNCQRDQNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RISKNCQRDQNRNNAgent, self).__init__() 
        self.args = args

        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3
        self.num_support = self.args.n_quantiles
        self.quantile_embed_dim = 64

        self.constant = np.log(np.exp(1 - 1e-3) - 1)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.phi = nn.Linear(args.quantile_embed_dim, args.rnn_hidden_dim)
        self.fc_q = nn.Linear(self.num_support, self.num_support)
        self.alpha = nn.Linear(self.num_support, 1)
        self.beta = nn.Linear(self.num_support, 1)

        if self.args.no_xavier is True:
            self.rnn.apply(initialize_weights_xavier)
            self.fc2.apply(initialize_weights_xavier)
            self.phi.apply(initialize_weights_xavier)
            self.fc_q.apply(initialize_weights_xavier)
            self.alpha.apply(initialize_weights_xavier)
            self.beta.apply(initialize_weights_xavier)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, forward_type=None, rnd_value01=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h = self.rnn(x, h_in)

        del x, h_in
        shape = h.shape
        batch_size = shape[0]


        batch_size_grouped = batch_size // self.args.n_agents
        rnd_value01 = th.ones(batch_size_grouped, self.num_support).cuda() * (1/self.num_support)
        rnd_value01 = rnd_value01.reshape(-1)
        n_rnd_quantiles = self.num_support
        assert rnd_value01.shape == (batch_size_grouped * self.num_support,)
        if self.args.iqn_like is True:
            h2 = h.reshape(batch_size, 1, self.args.rnn_hidden_dim).expand(-1, n_rnd_quantiles, -1).reshape(-1,
                                                                                                            self.args.rnn_hidden_dim)
            quantiles = rnd_value01.view(batch_size_grouped * n_rnd_quantiles, 1).expand(-1, self.quantile_embed_dim)
            assert quantiles.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)

            feature_id = th.arange(0, self.quantile_embed_dim).cuda()  
            feature_id = feature_id.view(1, -1).expand(batch_size_grouped * n_rnd_quantiles, -1)
            assert feature_id.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
            t, tau_hat, p = get_tau_hat_agent(quantiles) 
            del t, p
            cos = th.cos(math.pi * feature_id * tau_hat)
            assert cos.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)

            q_phi = F.relu(self.phi(cos))
            assert q_phi.shape == (batch_size_grouped * n_rnd_quantiles, self.args.rnn_hidden_dim)
            if self.args.name != "diql":
                q_phi = q_phi.view(batch_size_grouped, n_rnd_quantiles, self.args.rnn_hidden_dim)
                q_phi = q_phi.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).contiguous().view(-1,
                                                                                                    self.args.rnn_hidden_dim)  
            assert q_phi.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)  
            q_vals = self.fc2(h2 * q_phi)
            q_vals = q_vals.view(-1, n_rnd_quantiles, self.args.n_actions)
            assert q_vals.shape == (batch_size, n_rnd_quantiles, self.args.n_actions)
            q_vals = q_vals.permute(0, 2, 1)
            assert q_vals.shape == (batch_size, self.args.n_actions, n_rnd_quantiles)
            del feature_id, q_phi, h2, cos
            x = F.relu(q_vals)
        else:
            t2 = self.fc2(h)
            x = t2.unsqueeze(2).expand(-1, -1, self.num_support)


        quant = self.fc_q(x)
        quant = torch.softmax(quant, dim=-1)
        quant = torch.cumsum(quant, dim=-1)


        scale_a = F.relu(self.alpha(x))
        scale_b = self.beta(x)

        V = scale_a * quant + scale_b

        V = V.view(batch_size, self.args.n_actions, self.num_support)
        del x, scale_a, quant, scale_b
        if self.args.iqn_like is True:
            del q_vals
        rnd_value01 = rnd_value01.view(batch_size_grouped, self.num_support)
        return V, h, rnd_value01.detach()
