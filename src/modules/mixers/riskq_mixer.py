import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RiskqMixer(nn.Module):

    def __init__(self, args):
        super(RiskqMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.u_dim = int(np.prod(args.agent_own_state_size))

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = args.n_key_embedding_layer1
        self.n_head_embedding_layer1 = args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = args.n_head_embedding_layer2
        self.n_attention_head = args.n_attention_head
        self.n_constrant_value = args.n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.state_dim, self.n_query_embedding_layer1), nn.ReLU(), nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)))

        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1))

        self.scaled_product_value = np.sqrt(args.n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_head_embedding_layer1), nn.ReLU(), nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2))

        self.constrant_value_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_constrant_value), nn.ReLU(), nn.Linear(self.n_constrant_value, 1))

    def forward(self, agent_qs, states, target): 
        if target:  
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        bs = agent_qs.size(0)  
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents, n_rnd_quantiles)  

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            state_embedding = state_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        q_lambda_list = q_lambda_list.unsqueeze(3).expand(-1, -1, -1, n_rnd_quantiles)

        q_lambda_list = q_lambda_list.permute(0, 3, 2, 1)

        agent_qs = agent_qs.permute(0, 3, 1, 2)

        q_h = th.matmul(agent_qs, q_lambda_list)

        if self.args.type == 'weighted':
            w_h = th.abs(self.head_embedding_layer(states))
            w_h = w_h.unsqueeze(1).expand(-1, n_rnd_quantiles, self.n_head_embedding_layer2)
            w_h = w_h.reshape(-1, n_rnd_quantiles, self.n_head_embedding_layer2, 1)

            sum_q_h = th.matmul(q_h, w_h)
            sum_q_h = sum_q_h.reshape(-1, n_rnd_quantiles)
        else:
            sum_q_h = q_h.sum(-1)
            sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states).expand_as(sum_q_h)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1, n_rnd_quantiles)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.args.agent_own_state_size
        with th.no_grad():
            us = states[:, :agent_own_state_size * self.n_agents].reshape(-1, agent_own_state_size)
        return us