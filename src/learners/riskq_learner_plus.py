import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.riskq_mixer import RiskqMixer
from modules.mixers.dresq_central_atten import CentralattenMixer
from modules.mixers.dresq_central import DRESCentralMixer
from utils.rl_utils import build_distributional_td_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from controllers import REGISTRY as mac_REGISTRY
from torch.optim import Adam
from utils.risk import get_risk_q,get_tau_hat3

class RiskQLearnerPlus:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac

        self.logger = logger


        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.mixer = None

        if args.mixer is not None:

            if args.mixer == "ddn":
                self.mixer = DDNMixer(args)
            elif args.mixer == "dmix":
                self.mixer = DMixer(args)
            elif args.mixer == "riskq":
                self.mixer = RiskqMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.params += list(self.mixer.parameters())

        self.init_res_central(args, scheme)


        self.last_target_update_episode = 0


        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

        self.grad_norm = 1
        self.last_qr_update_t = 1

    def init_res_central(self, args, scheme):
        self.res_mixer = getattr(args, 'res_mixer', None)
        if self.res_mixer is not None:
            if args.res_mixer == "atten":
                self.res_mixer = CentralattenMixer(args)
            elif args.res_mixer == "dmix":
                self.res_mixer = DMixer(args)
            elif args.res_mixer == "ddn":
                self.res_mixer = DDNMixer(args)
            elif args.res_mixer == "ff":
                self.res_mixer = DRESCentralMixer(args)
        else:
            self.res_mixer = CentralattenMixer(args) 
        self.params += list(self.res_mixer.parameters())

        self.central_mac = None
        assert args.central_mac == "base_central_mac"
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())

        self.central_mixer = DRESCentralMixer(args) 
        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

    def update_qr(self, t_env):
        if hasattr(self.args, "version") and self.args.version == "v2":
            if (t_env - self.last_qr_update_t) / self.args.qr_update_interval >= 1.0 and t_env > self.args.start_to_update_qr:
                return True
            else:
                return False
        return False

    def get_mac_q_values(self, mac, rnd_01, batch, actions, forward_type, episode_length):
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if not self.args.name.startswith("v1"):
                agent_outs, _ = mac.forward(batch, t=t, forward_type=forward_type)
                del _
            else:
                agent_outs, _ = mac.forward(batch, t=t, forward_type=forward_type, rnd_value01=rnd_01[:, t])
                del _
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1) 
        assert mac_out.shape == (
            batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        if forward_type == "target":
            return mac_out
        elif forward_type == "policy":
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  
            return mac_out, chosen_action_qvals

    def get_quantile_loss(self, targets, current, tau_hat, mask):

        condition = targets.detach() - current.permute(0, 1, 3, 2)
        quantile_weight = th.abs(tau_hat - condition.le(0.).float())

        quantile_loss = quantile_weight * F.smooth_l1_loss(condition, th.zeros(condition.shape).cuda(),
                                                           reduction='none')
        quantile_loss = quantile_loss.mean(dim=-1).sum(dim=-1)[..., None]
        quantile_loss = (quantile_loss * mask).sum() / mask.sum()
        return quantile_loss

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.mixer is not None:
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents


        mac_out = []
        rnd_01 = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)
            if agent_rnd_quantiles is not None:
                agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
                rnd_01.append(agent_rnd_quantiles)
        mac_out = th.stack(mac_out, dim=1) 
        assert mac_out.shape == (
        batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        if self.args.agent != "qrdqn_rnn":
            rnd_01 = th.stack(rnd_01, dim=1)  
            assert rnd_01.shape == (batch.batch_size, episode_length + 1, n_quantile_groups, self.n_quantiles)
  

        actions = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

 
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) 

        res_mac_out, res_chosen_action_qvals = self.get_mac_q_values(self.res_mac, rnd_01, batch, actions, "policy", episode_length)
        central_mac_out, central_chosen_action_qvals = self.get_mac_q_values(self.central_mac, rnd_01, batch, actions, "policy", episode_length)
        target_central_mac_out = self.get_mac_q_values(self.target_central_mac, rnd_01, batch, actions, "target", episode_length)

        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_central_mac_out[target_avail_actions[:, :] == 0] = -9999999

        if self.args.double_q: 

            risk_type = self.args.risk_type
            risk_param = self.args.risk_param

            mac_out_detach = mac_out.clone().detach()  
            mac_out_detach[target_avail_actions == 0] = -9999999
            if self.args.agent == "qrdqn_rnn":
                rnd_01 = th.ones(1, self.args.n_target_quantiles).cuda() * (1 / self.args.n_target_quantiles)
                rnd_01 = rnd_01[None, None, ...]
                rnd_01 = rnd_01.expand(mac_out.shape[0], mac_out.shape[1], 1, -1)
            riskq_value, tau_hat = get_risk_q(risk_type, risk_param, rnd_01, mac_out_detach) 
        
            riskq_value = riskq_value.squeeze(-1).detach()
            riskq_value[avail_actions == 0] = -9999999

            max_result = riskq_value[:, :].max(dim=3, keepdim=True)
            cur_max_actions = max_result[1]
            mean_risk = max_result[0].detach().mean()
            chosen_riskq_vals = max_result[0]
            del tavail_actions
            del mac_out_detach
            del max_result
            del tau_hat

            rnd_01 = rnd_01[:, :-1] 
        else:
            cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1] 

        cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)

        target_per_agent_max_qvals = th.gather(target_central_mac_out, 3, cur_max_actions).squeeze(3)


        if self.args.mixer == "risk_atten": 
            agent_input_obs = self.mixer._build_inputs(batch)
            Q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1], agent_input_obs[:, :-1], target=False)
            del agent_input_obs
        else:
            Q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)

        Q_r = self.res_mixer(res_chosen_action_qvals, batch["state"][:, :-1], target=False)
        if getattr(self.args, 'negative_abs', False):
            Q_r = -1 * Q_r.abs()

        central_chosen_qvals = self.central_mixer(central_chosen_action_qvals, batch["state"][:, :-1], target=False)  #
        target_max_qvals = self.target_central_mixer(target_per_agent_max_qvals, batch["state"], target=True)


        if self.args.td_lambda == 0:
            targets = rewards.unsqueeze(3) + (self.args.gamma * (1 - terminated)).unsqueeze(3) * target_max_qvals[:, 1:]
        else:
            targets = build_distributional_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)
        assert targets.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        del target_avail_actions
        del target_max_qvals
        del rewards

        _, tau_hat, presum = get_tau_hat3(rnd_01)
        del _
        mean_Q = Q_tot.mean()

        is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
        w_r = th.where(is_max_action, th.zeros_like(Q_r), th.ones_like(Q_r))
        w_to_use = w_r.mean().item()  
        Q_jt = Q_tot + w_r.detach() * Q_r

        del cur_max_actions


        quantile_loss1 = self.get_quantile_loss(targets, central_chosen_qvals, tau_hat, mask)
        quantile_loss2 = self.get_quantile_loss(central_chosen_qvals, Q_jt, tau_hat, mask)



        loss = quantile_loss1 + quantile_loss2


        if self.update_qr(t_env):

            qr_targets = chosen_riskq_vals[:,:-1] + self.args.gamma * (1 - terminated[...,None]) * target_per_agent_max_qvals[:,1:]

            qr_td_error = (chosen_action_qvals[..., None] - qr_targets[..., None, :].detach())

            condition2 = qr_td_error
            quantile_weight2 = th.abs(tau_hat[...,None] - condition2.le(0.).float())
            quantile_loss2 = quantile_weight2 * F.smooth_l1_loss(condition2, th.zeros(condition2.shape).cuda(),
                                                               reduction='none')
            quantile_loss2 = quantile_loss2.mean(dim=4).sum(dim=3)
            mask2 = mask.expand_as(quantile_loss2)
            quantile_loss2 = (quantile_loss2 * mask2).sum() / mask2.sum()
            self.last_qr_update_t = t_env
            del mask2

  
        self.optimiser.zero_grad()
        if self.update_qr(t_env):
            qr_learning_ratio = getattr(self.args, "qr_learning_ratio", 1)
            loss = loss + qr_learning_ratio * quantile_loss2
            self.last_qr_update_t = t_env 
        loss.backward()



        agent_norm = 0
        for p in self.mac_params:

            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item()**2
        agent_norm = agent_norm**(1. / 2)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("quantile_loss1", quantile_loss1.item(), t_env)
            self.logger.log_stat("quantile_loss2", quantile_loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            self.logger.log_stat("mean_Q", mean_Q.item(), t_env)
            self.logger.log_stat("mean_risk", mean_risk.item(), t_env)
            self.log_stats_t = t_env

        del riskq_value
        del target_per_agent_max_qvals
        del terminated
        del rnd_01
        del condition
        del quantile_loss
        del quantile_weight
    
        del targets
        del Q_jt
        del Q_tot

        del mean_Q
        del mean_risk
        del tau_hat

        del presum
        del mac_out
        del target_mac_out


    def _update_targets(self):
        if self.target_mac is not None:
            self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.res_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.res_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    # TODO:
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac = copy.deepcopy(self.mac)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer = copy.deepcopy(self.mixer)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
