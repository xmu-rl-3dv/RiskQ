from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from utils.risk import get_risk_q
from torch.distributions import Categorical
import random


class RiskMAC:
    def __init__(self, scheme, groups, args):

        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.rnd01 = th.ones(1, self.args.n_target_quantiles).cuda() * (1/self.args.n_target_quantiles)

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.r_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        rnd01 = None
        if self.args.agent in ["qrdqn_rnn", "iqn_rnn", "risk_rnn", "ncqrdqn_rnn"]:
            agent_outputs, rnd01 = self.forward(ep_batch, t_ep, forward_type="approx")
        else:
            agent_outputs = self.forward(ep_batch, t_ep, forward_type=test_mode)
        if self.args.agent == "iqn_rnn":  
            agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
        if self.args.agent in ["risk_rnn", "ncqrdqn_rnn", "qrdqn_rnn"]:
            risk_type = self.args.risk_type
            risk_param = self.args.risk_param
            if rnd01 is None:
                rnd01 = self.rnd01

            ain = agent_outputs[bs]
            if ain.shape[0] == 1:
                ain = ain.squeeze(0)

            riskq_value, tau_hat = get_risk_q(risk_type, risk_param, rnd01, ain)

            riskq_value = riskq_value.squeeze(-1).unsqueeze(0)

            chosen_actions = self.action_selector.select_action(riskq_value, avail_actions[bs], t_env,
                                                                test_mode=test_mode)

            del tau_hat
            del riskq_value
            del rnd01

            
        else:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        if hasattr(self.args, 'noise_threshold') and  hasattr(self.args, 'noise_agent_num') and test_mode:
            noise_threshold = self.args.noise_threshold
            
            noise_agent_num = int(self.args.noise_agent_num)
            if  random.random() < noise_threshold:             
                chosen_actions[:, -noise_agent_num:] =  Categorical(avail_actions[bs][:, -noise_agent_num:].float()).sample().long()


        del agent_outputs
        return chosen_actions

    def forward(self, ep_batch, t, forward_type=None, rnd_value01=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.agent in ["iqn_rnn", "risk_rnn", "ncqrdqn_rnn"]:
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type, rnd_value01=rnd_value01)

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        del agent_inputs
        if self.args.agent in ["iqn_rnn", "risk_rnn", "ncqrdqn_rnn"]:
            return agent_outs, rnd_quantiles
        elif self.args.agent == "qrdqn_rnn":
            return agent_outs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, self.args.n_target_quantiles), None
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):

        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):

        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t]) 
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
