import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import io
from collections import defaultdict, deque

def mw45_task_names():
    return ['pick-out-of-hole', 'shelf-place', 'peg-unplug-side', 'drawer-open', 'pick-place', 'button-press-wall', 'assembly', 'door-open', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'door-close', 'drawer-close', 'faucet-open', 'faucet-close', 'hammer', 'handle-press-side', 'handle-press', 'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'reach', 'push-back', 'push', 'plate-slide', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'stick-push', 'push-wall', 'reach-wall','sweep-into', 'sweep', 'window-open', 'window-close', 'door-unlock', 'door-lock', 'bin-picking', 'basketball', 'soccer']

def generate_causal_mask(time_step, num_modalities):
    """
    Generates a causal mask for a sequence where each time step contains multiple modalities.
    The mask allows the model to look at all modalities within the current time step,
    but not at future time steps.

    Parameters:
    time_step (int): The number of time steps in the sequence.
    num_modalities (int): The number of modalities per time step.

    Returns:
    torch.Tensor: The causal mask.
    """
    # Size of the full sequence
    full_size = time_step * num_modalities

    # Creating a mask for the full sequence
    mask = torch.full((full_size, full_size), float('-inf'))

    # Filling in the mask
    for t in range(time_step):
        start_idx = t * num_modalities
        end_idx = (t + 1) * num_modalities
        mask[start_idx:end_idx, :end_idx] = 0

    return mask

### Update the latent embedding with new timestep
def update_z_history(z_history, z_new, pad_idx, pad_mask):
    batch_size, timestep, feature_dim = z_history.shape
    pad_update_mask = (pad_idx < timestep - 1) ### (batch_size, )
    
    ### Case I: the example is not padded 
    z_history_append = torch.cat((z_history[:, 1:].clone(), z_new.unsqueeze(1)), dim=1)
    ### Case II: the example is padded 
    z_history[pad_update_mask, pad_idx[pad_update_mask] + 1] = z_new[pad_update_mask]
    
    mask = pad_update_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,timestep,feature_dim)
    updated_z_history = torch.where(mask, z_history, z_history_append)

    # Update pad_mask and pad_idx for padded sequences
    new_pad_mask = pad_mask.clone()
    new_pad_idx = pad_idx.clone()
    new_pad_mask[pad_update_mask, pad_idx[pad_update_mask] + 1] = False
    new_pad_idx[pad_update_mask] += 1

    return updated_z_history, new_pad_idx, new_pad_mask

### code_pred: code sequence predicted by the meta-policy
### code_target: target code sequence provided by the tokenizer
### code_dist: pre-computed distance matrix between codes
def cal_tok_dist(code_pred, code_actual, code_dist):
    len_match = min(len(code_pred), len(code_actual))
    dist  = 0.
    for i in range(len_match):
        dist += code_dist[code_pred[i]][code_actual[i]]
    if len(code_pred) < len(code_actual):
        dist += np.max(code_dist) * (len(code_actual) - len(code_pred))
    return dist

def tokenize_vocab(traj_tok, vocab_lookup, merges):
    traj_vocab = []
    for i in range(len(traj_tok)):
        tok = traj_tok[i]
        idx = i
        while idx < len(traj_tok) - 1 and (tok, traj_tok[idx+1]) in merges:
            tok += traj_tok[idx+1]
        traj_vocab.append(vocab_lookup[tok])
    return traj_vocab

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())
    

def dynamics_loss(f_x1s, f_x2s):
    f_x1 = F.normalize(f_x1s, p=2., dim=-1, eps=1e-3)
    f_x2 = F.normalize(f_x2s, p=2., dim=-1, eps=1e-3)
    loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
    return loss
    
class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, input):
        # Create output tensor
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Sum up the gradients from all outputs
        grad_input = sum(grad_outputs)
        return grad_input

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def set_requires_grad(net, value=False):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)

### input shape: (batch_size, length, action_dim)
### output shape: (batch_size, action_dim)
class ActionEncoding(nn.Module):
    def __init__(self, action_dim, latent_action_dim, multistep):
        super().__init__()
        self.action_dim = action_dim
        self.action_tokenizer = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64), 
            nn.Tanh(), 
            nn.Linear(64, latent_action_dim)
        )
        self.action_seq_tokenizer = nn.Sequential(
            nn.Linear(latent_action_dim*multistep, latent_action_dim*multistep),
            nn.LayerNorm(latent_action_dim*multistep), nn.Tanh()
        )
        self.apply(weight_init)
        
    def forward(self, action, seq=False):
        if seq:
            batch_size = action.shape[0]
            action = self.action_tokenizer(action) #(batch_size, length, action_dim)
            action = action.reshape(batch_size, -1)
            return self.action_seq_tokenizer(action)
        else:
            return self.action_tokenizer(action)



class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def choose(traj_list, max_traj):
    # NOTE: this assumes that random's seed has been set.
    random.shuffle(traj_list)
    return (traj_list if max_traj < 0 else traj_list[:max_traj])


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

        
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def to_torch_distribute(xs):
    return tuple(torch.as_tensor(x).cuda() for x in xs)

def encode_multiple(encoder, xs, detach_lst):
    length = [x.shape[0] for x in xs]
    xs, xs_lst = torch.cat(xs, dim=0), []
    xs = encoder(xs)
    start = 0
    for i in range(len(detach_lst)):
        x = xs[start:start+length[i], :]
        if detach_lst[i]:
            x = x.detach()
        xs_lst.append(x)
        start += length[i]
    return xs_lst
    
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


# class ActionQuantizer(nn.Module):
#     def __init__(self, agent):
#         super().__init__()
#         self.agent = agent
#         set_requires_grad(self.agent, False)
        
#     def forward(self, action):
#         action_en = self.agent.TACO.proj_aseq(action)
#         _, _, _, _, min_encoding_indices = self.agent.TACO.quantizer(action_en)
#         return min_encoding_indices
        
    
def compute_traj_latent_embedding(episode, device, nstep_history):
    with torch.no_grad():
        obs = episode['observation'][:-1]
        state     = episode['state'][:-1]
        if state.shape[-1] == 39:
            state = np.hstack((state[:, :4], state[:, 18 : 18 + 4]))
        action    = episode['action'][1:]
        obs, state, action = to_torch((obs, state, action), device=device)


        obs_buffer = deque(maxlen=nstep_history)
        state_buffer = deque(maxlen=nstep_history)
        obs_episode, state_episode = [], []

        for t in range(obs.shape[0]):
            ### corner case (prefill the queue in the initial timestep)
            if len(obs_buffer) == 0:
                for i in range(nstep_history):
                    obs_buffer.append(obs[0].unsqueeze(0)) ### (1,3,128,128)
                    state_buffer.append(state[0].float().unsqueeze(0)) ### (1, state_dim)
            else:
                obs_buffer.append(obs[t].unsqueeze(0)) ### (1,3,128,128)
                state_buffer.append(state[t].float().unsqueeze(0)) ### (1,state_dim)

            obs_history = torch.concatenate(list(obs_buffer), dim=0) ### (10,3,128,128)
            state_history = torch.concatenate(list(state_buffer), dim=0) ### (10,state_dim)
    
            obs_episode.append(obs_history.unsqueeze(0))
            state_episode.append(state_history.unsqueeze(0))


        obs_episode = torch.concatenate(obs_episode, dim=0)
        state_episode = torch.concatenate(state_episode, dim=0)
        obs_history = (obs_episode,
                       state_episode)
    return obs_history