import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class IQNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(IQNRNNAgent, self).__init__()
        self.args = args

        self.quantile_embed_dim = args.quantile_embed_dim
        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_approx_quantiles = args.n_approx_quantiles

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.phi = nn.Linear(args.quantile_embed_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):

        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, forward_type=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        if forward_type == "approx":   
            n_rnd_quantiles = self.n_approx_quantiles 
        elif forward_type == "policy":
            n_rnd_quantiles = self.n_quantiles          
        elif forward_type == "target":
            n_rnd_quantiles = self.n_target_quantiles   
        else:
            raise ValueError("Unknown forward_type")
        shape = h.shape
        batch_size = shape[0]
        h2 = h.reshape(batch_size, 1, self.args.rnn_hidden_dim).expand(-1, n_rnd_quantiles, -1).reshape(-1, self.args.rnn_hidden_dim)    
        assert h2.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)
        shape = h2.shape

        if self.args.name == "diql":
            rnd_quantiles = th.rand(batch_size * n_rnd_quantiles).cuda()
            batch_size_grouped = batch_size
        else:
            batch_size_grouped = batch_size // self.args.n_agents

            rnd_quantiles = th.rand(batch_size_grouped, 1, n_rnd_quantiles).cuda()
            rnd_quantiles = rnd_quantiles.reshape(-1)
        assert rnd_quantiles.shape == (batch_size_grouped * n_rnd_quantiles,)

        quantiles = rnd_quantiles.view(batch_size_grouped * n_rnd_quantiles, 1).expand(-1, self.quantile_embed_dim)
        assert quantiles.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)

        feature_id = th.arange(0, self.quantile_embed_dim).cuda() 
        feature_id = feature_id.view(1, -1).expand(batch_size_grouped * n_rnd_quantiles, -1)
        assert feature_id.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        cos = th.cos(math.pi * feature_id * quantiles)
        assert cos.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)

        q_phi = F.relu(self.phi(cos))
        assert q_phi.shape == (batch_size_grouped * n_rnd_quantiles, self.args.rnn_hidden_dim)
        if self.args.name != "diql":
            q_phi = q_phi.view(batch_size_grouped, n_rnd_quantiles, self.args.rnn_hidden_dim)
            q_phi = q_phi.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).contiguous().view(-1, self.args.rnn_hidden_dim)
        assert q_phi.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)
        q_vals = self.fc2(h2 * q_phi)
        q_vals = q_vals.view(-1, n_rnd_quantiles, self.args.n_actions)
        assert q_vals.shape == (batch_size, n_rnd_quantiles, self.args.n_actions)
        q_vals = q_vals.permute(0, 2, 1) 
        assert q_vals.shape == (batch_size, self.args.n_actions, n_rnd_quantiles)
        rnd_quantiles = rnd_quantiles.view(batch_size_grouped, n_rnd_quantiles)
        return q_vals, h, rnd_quantiles
