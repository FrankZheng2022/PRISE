import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import copy
import pickle
import io
import distutils.dir_util
import hydra
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group, gather
from libero.libero import benchmark
import utils.misc as utils
import utils.libero_wrapper as libero_wrapper
from utils.logger import Logger
from replay_buffer import make_replay_loader_dist
import torch.multiprocessing as mp
from collections import defaultdict, deque
from tokenizer_api import Tokenizer

torch.backends.cudnn.benchmark = True

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{}".format(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def make_agent(obs_shape, action_dim, rank, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_dim = action_dim
    device_ids = list(range(torch.cuda.device_count()))
    cfg.device = device_ids[rank]
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, rank, world_size):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.rank = int(rank)
        self.world_size = world_size
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        device_ids = list(range(torch.cuda.device_count()))
        self.device = device_ids[rank]
        self.results_dir = Path(self.cfg.results_dir)
        self.eval_dir = self.results_dir / 'eval_libero90' / self.cfg.downstream_exp_name
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        

        a_dim = self.cfg.action_dim
        obs_shape = [3]+list(self.cfg.img_res)
        self.agent = make_agent(obs_shape,
                                a_dim,
                                rank,
                                self.cfg.agent)
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.cfg.downstream_task_suite]()
        task = task_suite.get_task(int(self.cfg.downstream_task_name))
        self.eval_env = libero_wrapper.make(self.cfg.downstream_task_name, 
                                            self.cfg.downstream_task_suite, seed=self.cfg.seed)
        self.eval_env.task_name = task.name
        self.eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)

    def act(self, env, obs, code_buffer, z_history_buffer):
        obs_agent = obs.agentview
        obs_wrist = obs.wristview
        state     = obs.state 
        task_embedding = env.task_embedding
        
        ### Encode the current timestep
        task_embedding = torch.torch.as_tensor(task_embedding, device=self.device)
        obs_agent = torch.torch.as_tensor(obs_agent.copy(), device=self.device).unsqueeze(0)
        obs_wrist = torch.torch.as_tensor(obs_wrist.copy(), device=self.device).unsqueeze(0)
        state     = torch.torch.as_tensor(state, device=self.device).unsqueeze(0)
        z = self.agent.encode_obs((obs_agent, obs_wrist, state, task_embedding), aug=False)
        
        ### Prefill the buffer at the initial timestep
        if len(z_history_buffer) == 0:
            for i in range(self.cfg.nstep_history):
                z_history_buffer.append(z)
        ### Otherwise, append to the observation embedding buffer
        else:
            z_history_buffer.append(z) ### (1,1,4,feature_dim)
        z_history = torch.concatenate(list(z_history_buffer), dim=1) 
        z_history = self.agent.compute_transformer_embedding(z_history)
        
        ### Query the skill token policy again if the code buffer is empty
        if len(code_buffer) == 0:
            meta_action = self.agent.PRISE.module.meta_policy(z_history).max(-1)[1]
            tok = self.idx_to_tok[int(meta_action.item())]
            try:
                code_buffer = self.tokenizer.decode([tok], verbose=False)
            except:
                print('Error occured when choosing meta action:{}'.format(meta_action))
                assert False
        
        ### Pop the first code and decode the raw action
        code_selected = code_buffer.pop(0)
        learned_code  = self.agent.PRISE.module.a_quantizer.embedding.weight
        u = learned_code[code_selected, :]
        action = self.agent.PRISE.module.decode(z_history, u, decoder_type=self.cfg.decoder_type)
        return code_buffer, action.detach().cpu().numpy()[0]
    
       
    def evaluate_libero90(self):
        self.agent.train(False)
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        num_tasks_per_gpu = (90//self.world_size) + 1
        
        for i in range(self.rank*num_tasks_per_gpu, min((self.rank+1)*num_tasks_per_gpu, 90)):
            
            ### Setup eval environment
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[self.cfg.downstream_task_suite]()
            task = task_suite.get_task(i)
            eval_env = libero_wrapper.make(i, self.cfg.downstream_task_suite, 
                                           seed=self.cfg.seed)
            eval_env.task_name = task.name
            eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)
            task_name = task.name
            
            eval_start_time = time.time()
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
            counter, episode, success = 0, 0, 0
            while eval_until_episode(episode):
                time_step = eval_env.reset()
                step, code_buffer = 0, []
                z_history_buffer = deque(maxlen=self.cfg.nstep_history)
                while step < self.cfg.eval_max_steps:
                    if time_step['done']:
                        success += 1
                        break
                    with torch.no_grad():
                        code_buffer, action = self.act(eval_env, time_step, code_buffer, z_history_buffer)
                    time_step = eval_env.step(action)
                    step += 1
                episode += 1

            print(f'Task:{task_name} Evaluation Time:{time.time()-eval_start_time}s Success Rate:{success/self.cfg.num_eval_episodes*100}%', flush=True)
            
            ### Save the evaluated success rate
            try:
                with open(self.eval_dir / '{}.pkl'.format(self.cfg.downstream_exp_name), 'rb') as f:
                    performance = pickle.load(f)
            except:
                performance = {}
            with open(self.eval_dir / '{}.pkl'.format(self.cfg.downstream_exp_name), 'wb') as f:
                performance[task_name] = success/self.cfg.num_eval_episodes*100
                pickle.dump(performance, f)
            eval_env.close()
        print(f'=====================Process {self.rank} End Evaluation=====================')

    
    def load_snapshot(self):
        snapshot = self.results_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=torch.device(f'cuda:{self.device}'))
        self.__dict__['agent'] = payload['agent']
        self.agent.device = self.device
        self.agent.PRISE.device = self.device
        self.agent.PRISE.to(self.device)
        self.idx_to_tok = self.agent.idx_to_tok
        self.tok_to_idx = self.agent.tok_to_idx
        self.tokenizer = self.agent.tokenizer
        print('Resuming Snapshopt')



RANK = None
WORLD_SIZE = None

@hydra.main(config_path='cfgs', config_name='prise_config')
def main(cfg):
    global RANK, WORLD_SIZE
    ddp_setup(RANK, WORLD_SIZE, cfg.port)
    from eval_libero90 import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, RANK, WORLD_SIZE)
    root_dir = Path.cwd()
    workspace.load_snapshot()
    workspace.evaluate_mt()
    destroy_process_group()

def wrapper(rank, world_size, cfg):
    global RANK, WORLD_SIZE
    RANK = rank

    WORLD_SIZE = world_size
    print(f'WORLD SIZE: {world_size}, RANK: {rank}')
    main(cfg)

def main_mp_launch_helper(cfg=None):
    world_size = torch.cuda.device_count()
    if world_size==1:  
        wrapper(0, 1, cfg)  
    else:
        mp.spawn(wrapper, args=(world_size, cfg), nprocs=world_size)

if __name__ == '__main__':
    main_mp_launch_helper()


