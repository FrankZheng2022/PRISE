import datetime
import io
import random
import traceback
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
import utils.misc as utils


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_traj_per_task, max_size, num_workers, nstep,
                 nstep_history, fetch_every, save_snapshot,
                 rank=None, world_size=None,
                 n_code=None, vocab_size=None,
                 min_frequency=None, max_token_length=None):
        self._replay_dir = replay_dir if type(replay_dir) == list else [replay_dir]
        self._max_traj_per_task = max_traj_per_task
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._nstep_history = nstep_history
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self.n_code    = n_code
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        print('Loading Data into CPU Memory')
        self._preload()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def __len__(self):
        return self._size

    def _store_episode(self, eps_fn):
        episode = load_episode(eps_fn)
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        return True

    def _preload(self):
        eps_fns = []
        for replay_dir in self._replay_dir:
            eps_fns.extend(utils.choose(sorted(replay_dir.glob('*.npz'), reverse=True), self._max_traj_per_task))
        if len(eps_fns)==0:
            raise ValueError('No episodes found in {}'.format(self._replay_dir))
        for eps_idx, eps_fn in enumerate(eps_fns):
            if self.rank is not None and eps_idx % self.world_size != self.rank:
                continue
            else:
                self._store_episode(eps_fn)
        print(f'Process {self.rank} Loaded {len(self._episode_fns)} Trajectories')
    
    
    def _sample(self):
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        task_embedding = episode['task_embedding'].astype(np.float32)
        
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        action     = episode['action'][idx].astype(np.float32)
        action_seq = [episode['action'][idx+i].astype(np.float32) for i in range(self._nstep)]

        ############### Prepare future observations ###############
        next_obs_agent, next_obs_wrist, next_state, next_task_embedding = [], [], [], []
        for i in range(self._nstep):
            next_obs_agent.append(episode['observation'][idx + i][None,:])
            next_obs_wrist.append(episode['observation_wrist'][idx + i][None,:])
            next_state.append(episode['state'][idx + i][None,:])
            next_task_embedding.append(task_embedding[None,:])

        next_obs_agent = np.vstack(next_obs_agent)
        next_obs_wrist = np.vstack(next_obs_wrist)
        next_state = np.vstack(next_state).astype(np.float32)
        next_task_embedding = np.vstack(next_task_embedding)                           
        next_obs = (next_obs_agent, next_obs_wrist, next_state, next_task_embedding)
        
        ############### Prepare historical observations ###############
        obs_agent_history, obs_wrist_history, state_history, task_embedding_history = [], [], [], []
        timestep = idx - 1
        ### obs_history: (o_{t-3}, o_{t-2}, o_{t-1}, o_{t}, 0, 0 ...)
        while timestep >= 0 and len(obs_agent_history)<self._nstep_history:
            obs_agent_history = [episode['observation'][timestep][None,:]] + obs_agent_history
            obs_wrist_history = [episode['observation_wrist'][timestep][None,:]] + obs_wrist_history
            state_history     = [episode['state'][timestep][None, :]] + state_history 
            task_embedding_history = [task_embedding[None, :]] + task_embedding_history
            timestep -= 1
            
        ### pad the missing steps when the chosen timestep is smaller than nstep_history
        pad_step = self._nstep_history - len(obs_agent_history)
        obs_agent_history = [episode['observation'][0][None,:] for i in range(pad_step)] + obs_agent_history
        obs_wrist_history = [episode['observation_wrist'][0][None,:] for i in range(pad_step)] + obs_wrist_history
        state_history     = [episode['state'][0][None, :] for i in range(pad_step)] + state_history 
        task_embedding_history = [task_embedding[None, :] for i in range(pad_step)] + task_embedding_history
        
        obs_agent_history = np.vstack(obs_agent_history) ### (10, feature_dim)
        obs_wrist_history = np.vstack(obs_wrist_history) ### (10, feature_dim)
        state_history = np.vstack(state_history).astype(np.float32) ### (10, feature_dim)
        task_embedding_history = np.vstack(task_embedding_history)  ### (10, feature_dim)         
        obs_history = (obs_agent_history, obs_wrist_history, state_history, task_embedding_history)
                                 
        if 'token' in episode.keys():
            tok = episode['token'][idx - 1]
            return (obs_history, action, tok, action_seq, next_obs)
        else:
            return (obs_history, action, action_seq, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader_dist(replay_dir, max_traj_per_task, max_size, batch_size, num_workers,
                       save_snapshot, nstep, nstep_history, rank, world_size,
                        n_code=None, vocab_size=None, min_frequency=None,
                        max_token_length=None):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_traj_per_task,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            nstep_history,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            rank=rank,
                            world_size=world_size,
                            n_code=n_code,
                            vocab_size=vocab_size,
                            min_frequency=min_frequency,
                            max_token_length=max_token_length)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=False,
                                         worker_init_fn=_worker_init_fn)
    return loader
