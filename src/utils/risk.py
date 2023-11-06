
import copy

import numpy as np
import torch
import math


def get_risk_q(risk_type, risk_param, rnd_01, z_values):
    if len(rnd_01.shape) == 4 and len(z_values.shape) == 5:
        batch_size = z_values.shape[0]
        episode_length = z_values.shape[1]

        n_quantiles = z_values.shape[4]
        n_agents = z_values.shape[2]
        n_actions = z_values.shape[3]
        tau_tmp = rnd_01.expand(-1, -1, n_agents, -1)
        tau_tmp = tau_tmp.unsqueeze(3).expand(-1, -1, -1, n_actions, -1)
        tau, tau_hat, presum_tau = get_tau_hat(tau_tmp)
        risk_q_value = get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, z_values)
        del tau_tmp
    if len(rnd_01.shape) == 2 and len(z_values.shape) ==3: 
        n_agents = z_values.shape[0]
        n_actions = z_values.shape[1]
        n_quantiles = z_values.shape[2]
        tau_tmp = rnd_01.expand(n_agents, -1, -1)
        tau_tmp = tau_tmp.expand(-1,n_actions,-1)
        tau, tau_hat, presum_tau = get_tau_hat2(tau_tmp)
        risk_q_value = get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, z_values)
    del tau, presum_tau
    return risk_q_value, tau_hat

def normal_cdf(value, loc=0., scale=1.):
    return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))


def normal_icdf(value, loc=0., scale=1.):
    return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)


def normal_pdf(value, loc=0., scale=1.):
    return torch.exp(-(value - loc)**2 / (2 * scale**2)) / scale / np.sqrt(2 * np.pi)


def get_tau_hat(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)  
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:,:,:, 0:1] = tau[:,:,:,:, 0:1] / 2.
        tau_hat[:,:,:,:, 1:] = (tau[:,:,:,:, 1:] + tau[:,:,:,:, :-1]) / 2.
    return tau, tau_hat, presum_tau



def get_tau_hat2(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True) 
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:, 0:1] = tau[:,:, 0:1] / 2.
        tau_hat[:,:, 1:] = (tau[:,:, 1:] + tau[:,:, :-1]) / 2.
    return tau, tau_hat, presum_tau

def get_tau_hat3(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:,:, 0:1] = tau[:,:,:, 0:1] / 2.
        tau_hat[:,:,:, 1:] = (tau[:,:,:, 1:] + tau[:,:,:, :-1]) / 2.
    return tau, tau_hat, presum_tau

def get_tau_hat_agent(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True) 
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,0:1] = tau[:, 0:1] / 2.
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2. 
    return tau, tau_hat, presum_tau



def get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, quantile_values):
    if risk_type == "var":
        n_quantiles = quantile_values.shape[-1]
        idx = -2
        if risk_param - 1.0/(n_quantiles * 2) >= 0:
            idx = math.floor((risk_param - 1.0/(n_quantiles * 2)) / (1.0/n_quantiles))
        with torch.no_grad():
            mask = torch.zeros_like(tau_hat)
        mask[...,idx] = 1
        risk_q_value = torch.sum(mask * quantile_values, dim=-1, keepdims=True)
        return risk_q_value
    else:
        with torch.no_grad():
            risk_weights = distortion_de(tau_hat, risk_type, risk_param)
        risk_q_value = torch.sum(risk_weights * presum_tau * quantile_values, dim=-1, keepdims=True)
    return risk_q_value

def distortion_fn(tau, mode="neutral", param=0.):
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = tau
        elif mode == "wang":
            tau_ = normal_cdf(normal_icdf(tau) + param)
        elif mode == "cvar":
            tau_ = (1. / param) * tau
        elif mode == "cpw":
            tau_ = tau**param / (tau**param + (1. - tau)**param)**(1. / param)
        return tau_.clamp(0., 1.)
    else:
        return 1 - distortion_fn(1 - tau, mode, -param)


def distortion_de(tau, mode="neutral", param=0., eps=1e-8):
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = torch.ones_like(tau)
        elif mode == "wang":
            tau_ = normal_pdf(normal_icdf(tau) + param) / (normal_pdf(normal_icdf(tau)) + eps)
        elif mode == "cvar":
            tau_ = (1. / param) * (tau < param)
        elif mode == "cpw":
            g = tau**param
            h = (tau**param + (1 - tau)**param)**(1 / param)
            g_ = param * tau**(param - 1)
            h_ = (tau**param + (1 - tau)**param)**(1 / param - 1) * (tau**(param - 1) - (1 - tau)**(param - 1))
            tau_ = (g_ * h - g * h_) / (h**2 + eps)
        return tau_.clamp(0., 5.)

    else:
        return distortion_de(1 - tau, mode, -param)
