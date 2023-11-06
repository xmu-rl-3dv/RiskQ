from envs.multiagentenv import MultiAgentEnv
import numpy as np
import torch as th
import random
import math
import copy

is_dist = True
class MultiAgentCar(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        self.n_agents = kwargs["n_agents"]
        self.map_name = kwargs["map_name"]
        if self.map_name in["macf_opt"]:
            self.n_agents = 2
            self.target_pos = 2.5 
            self.episode_limit = 200
            self.delta_time_per_step = 0.1 
            self.steps = 0
            self.lowest_accel = -1 
            self.highest_accel = 1 
            self.n_actions = 21
            self.accel_base = (self.highest_accel - self.lowest_accel)/self.n_actions
            self.start_pos = 0.0
            self.speed_limit = 1 
            self.speed_max = 1.8
            self.probability_crash = 0.2
            self.risk_delta = 0.4
            self.observation_range = 0.3 
            self.ofr_reward = 1
            self.normal_reward = -1 
            self.crash_reward = -1 
            self.termination_reward = 10 
            self.num_fails = 0 
            self.num_reached = 0
            self.over_speeds = 0
        self.reached = [False, False]
        self.total_pos = []
        self.pos = []
        self.vel = []
        for i in range(self.n_agents):
            self.pos.append(self.start_pos)
            self.vel.append(0)
            self.total_pos.append(self.start_pos)

    def reset(self):
        self.steps = 0
        self.num_fails = 0
        self.num_reached = 0
        self.over_speeds = 0
        self.pos = []
        self.vel = []
        self.total_pos = []
        self.reached = [False, False]
        for i in range(self.n_agents):
            self.pos.append(self.start_pos)
            self.vel.append(0)
            self.total_pos.append(self.start_pos)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        terminated = False
        self.steps = self.steps + 1
        for i in range(self.n_agents):
            if self.reached[i] == False :
                action = actions[i]
                accel = self.accel_base * action + self.lowest_accel 
                if (self.vel[i] + accel * self.delta_time_per_step) >= self.speed_max: 
                    t1 = (self.speed_max - self.vel[i]) / accel
                    if isinstance(t1, th.Tensor):
                        t1 = t1.detach().cpu().numpy()
                    t1 = t1 * 100
                    t1 = np.round(t1)
                    t1 = t1 / 100
                    t2 = self.delta_time_per_step - t1
                    self.pos[i] = self.pos[i] + (self.vel[i] * t1 + 0.5 * accel * t1 * t1) + (self.speed_max * t2)
                    self.vel[i] = self.speed_max
                else:
                    self.pos[i] = self.pos[i] + self.vel[i] * self.delta_time_per_step + 0.5 * accel * self.delta_time_per_step * self.delta_time_per_step
                    self.vel[i] = self.vel[i] + accel * self.delta_time_per_step
            else:
                self.pos[i] = 2.5
                self.vel[i] = 0.0
            
        reward = self.normal_reward 
        crash_rewards = 0 
        for i in range(self.n_agents):
            if isinstance(self.pos[i], th.Tensor):
                pos = self.pos[i].detach().cpu().numpy()
            else:
                pos = self.pos[i]
            reward -= abs(pos-self.target_pos)

            if self.reached[i] == False:
                if abs(pos - self.target_pos)<1e-2: 
                    self.reached[i] = True
                    self.num_reached += 1 
                    reward += 5

            if abs(self.vel[i]) > self.speed_limit:
                prob = random.random()
                self.over_speeds += 1
                risk_now = (self.vel[i] - self.speed_limit) * self.risk_delta
                if prob <= risk_now:
                    crash_rewards += self.crash_reward
                    self.num_fails +=1
            self.total_pos.append(pos)
            
        reward += crash_rewards
        
        pos_dis = abs(self.pos[0] - self.pos[1]) 
        if pos_dis <= self.observation_range:
            reward += self.ofr_reward

        if self.num_reached == self.n_agents:
            reward += self.termination_reward 
            terminated = True

        if self.steps >=self.episode_limit:
            terminated = True
        
        def average(lst):
            return sum(lst) / (len(lst)*1.0)
        
        info = {}
        info["num_fails"] = self.num_fails
        info["pos"] = average(self.total_pos)
        info["num_reached"] = self.num_reached
        info["num_overspeed"] = self.over_speeds
        
        return reward, terminated, info


    def get_obs(self):

        all = [self.get_obs_agent(i) for i in range(self.n_agents) ]

        return all


    def get_obs_agent(self, agent_id):
        my_pos = self.pos[agent_id]
        ret = []
        for i in range(self.n_agents):
            pos = self.pos[i]
            if abs(pos - my_pos)<=self.observation_range:
                ret.append(pos)
                ret.append(self.vel[i])
            else:
                ret.append(0)
                ret.append(0)
        return ret

    def get_obs_size(self):
        return 2 * self.n_agents 

    def get_state(self):
        ret = []
        for i in range(self.n_agents):
            ret.append(self.pos[i])
            ret.append(self.vel[i])
        return ret

    def get_state_size(self):

        return 2 * self.n_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):

        return np.ones(self.n_actions) 

    def get_total_actions(self):

        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError