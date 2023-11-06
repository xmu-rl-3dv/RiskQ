from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class BaseCentralMAC:

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type 

        self.hidden_states = None

    def forward(self, ep_batch, t, forward_type=None, rnd_value01=None):

        agent_inputs = self._build_inputs(ep_batch, t)

        if self.args.agent in ["iqn_rnn", "ncqrdqn_rnn"]:
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type, rnd_value01=rnd_value01)

            return agent_outs, rnd_quantiles
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
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
        if self.args.central_agent == "central_rnn_big":
            inputs.append(batch["state"][:, t].unsqueeze(1).repeat(1, self.args.n_agents, 1))
        else:
            inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1]) 
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))  
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.central_agent == "central_rnn_big":
            input_shape += scheme["state"]["vshape"]
            input_shape -= scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] 
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
