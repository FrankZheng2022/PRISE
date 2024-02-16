import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from collections import defaultdict, deque
import copy
import distutils.dir_util
import hydra
import numpy as np
import time
import torch
import torch.nn as nn
import libero_wrapper
from libero.libero import benchmark
from logger import Logger
from replay_buffer import make_replay_loader_dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group, gather
from tokenizer_api import Tokenizer
from pathlib import Path
import pickle
import utils

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

### Instantiate the agent with given config
def make_agent(obs_shape, action_dim, rank, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_dim = action_dim
    device_ids = list(range(torch.cuda.device_count()))
    cfg.device = device_ids[rank]
    return hydra.utils.instantiate(cfg)

### Construct the path to the demonstration dataset of a given task
def construct_task_data_path(root_dir, task_name, task_data_dir_suffix='framestack1'):
    return Path(root_dir) / (task_name.lower()+('' if not task_data_dir_suffix or task_data_dir_suffix == 'None' else task_data_dir_suffix))


class Workspace:
    def __init__(self, cfg, rank, world_size):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.rank = rank
        self.world_size = world_size
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        device_ids = list(range(torch.cuda.device_count()))
        self.device = device_ids[rank]

        a_dim = self.cfg.action_dim
        obs_shape = [3]+list(self.cfg.img_res)  
        self.agent = make_agent(obs_shape,
                                a_dim,
                                rank,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.results_dir = Path(self.cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.pretraining_data_dirs = []
        ### In stage 1, 2, self.pretraining_data_dirs contain directories to demonstration dataset of all tasks from libero-90
        if self.cfg.stage < 3 or self.cfg.multitask:
            for task_id in range(90): 
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict['libero_90']()
                task = task_suite.get_task(task_id)
                task_name = task.name
                offline_data_dir = construct_task_data_path(self.cfg.data_storage_dir, task_name, self.cfg.task_data_dir_suffix)
                self.pretraining_data_dirs.append(offline_data_dir)
            self.eval_env = None
        
        ### In stage 3, set up the eval environment
        else:
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[self.cfg.downstream_task_suite]()
            task = task_suite.get_task(int(self.cfg.downstream_task_name))
            task_name = task.name
            self.eval_env = libero_wrapper.make(self.cfg.downstream_task_name, 
                                                self.cfg.downstream_task_suite, seed=self.cfg.seed, 
                                                libero_path=self.cfg.libero_path)
            self.eval_env.task_name = task_name
            self.eval_env.task_embedding = libero_wrapper.get_task_embedding(task.language)

        assert self.cfg.stage in [1, 2, 3], "Stage must be 1, 2, or 3."
        
        
        ### Set up the directory to store experimental results
        ### In stage 3, the results will be stored at results_dir / eval / stage_3/XXX
        if self.cfg.stage < 3:
            self.eval_dir = self.results_dir
        else:
            self.eval_dir = self.results_dir / 'eval' / f'stage_{self.cfg.stage}' / f'{self.cfg.downstream_task_suite}_task{self.cfg.downstream_task_name}'
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        #### Don't need to load the data in stage 2 (calculating BPE)
        if self.cfg.stage == 2:
            return
        self.setup_replay_buffer()

    def setup_replay_buffer(self):
        # create logger
        log_dir = self.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_dir, use_tb=False, offline=True)
        # create envs
        print('Rank:{} World Size:{}'.format(self.rank, self.world_size))

        if self.cfg.stage == 1 or self.cfg.multitask:
            self.replay_loader = make_replay_loader_dist(
                self.pretraining_data_dirs, self.cfg.max_traj_per_task, self.cfg.replay_buffer_size,
                self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                True, self.cfg.nstep, self.cfg.nstep_history, 
                self.rank, self.world_size, n_code=self.cfg.n_code, vocab_size=self.cfg.vocab_size,
                    min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length)
        elif self.cfg.stage == 3:
            downstream_data_path = construct_task_data_path(self.cfg.data_storage_dir, self.eval_env.task_name, self.cfg.task_data_dir_suffix)
            print(f"Loading target task data from {downstream_data_path}")
            self.replay_loader = make_replay_loader_dist(
                [downstream_data_path], self.cfg.max_traj_per_task, self.cfg.replay_buffer_size,
                self.cfg.batch_size//self.world_size, self.cfg.replay_buffer_num_workers,
                True, self.cfg.nstep, self.cfg.nstep_history, 
                self.rank, self.world_size,
                n_code=self.cfg.n_code, vocab_size=self.cfg.vocab_size,
                min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length)
        else:
            assert self.cfg.stage != 2, "You shouldn't set up the replay buffer for stage 2. Most likely you ended up here due to a logic bug."

        print('Rank:{} Finish Reading Data'.format(self.rank))
        self._replay_iter = None
        self.performance = []


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter= iter(self.replay_loader)
        return self._replay_iter
    
    ### Query the PRISE agent's action given current observatio
    def act(self, env, obs, code_buffer, z_history_buffer):
        obs_agent = obs.agentview
        obs_wrist = obs.wristview
        state     = obs.state 
        task_embedding = env.task_embedding       
        ### convert to torch array
        task_embedding = torch.torch.as_tensor(task_embedding, device=self.device)
        obs_agent = torch.torch.as_tensor(obs_agent.copy(), device=self.device).unsqueeze(0)
        obs_wrist = torch.torch.as_tensor(obs_wrist.copy(), device=self.device).unsqueeze(0)
        state     = torch.torch.as_tensor(state, device=self.device).unsqueeze(0)
        ### get observation embedding
        z = self.agent.encode_obs((obs_agent, obs_wrist, state, task_embedding), aug=False)
        
        ### At timestep 0, pre-fill z_history_buffer 
        if len(z_history_buffer) == 0:
            for i in range(self.cfg.nstep_history):
                z_history_buffer.append(z)
        else:
            z_history_buffer.append(z) 
        
        ### Concatenate the historical observations and calculate observation embedding
        z_history = torch.concatenate(list(z_history_buffer), dim=1)
        z_history = self.agent.compute_transformer_embedding(z_history)
        
        ### If the code_bfufer is empty, re-query the skill token policy
        if len(code_buffer) == 0:
            meta_action = self.agent.PRISE.module.token_policy(z_history).max(-1)[1]
            tok = self.idx_to_tok[int(meta_action.item())]
            code_buffer = self.tokenizer.decode([tok], verbose=False)
        
        code_selected = code_buffer.pop(0)
        learned_code  = self.agent.PRISE.module.a_quantizer.embedding.weight
        u = learned_code[code_selected, :]
        action = self.agent.PRISE.module.decode(z_history, u, decoder_type=self.cfg.decoder_type)
        return code_buffer, action.detach().cpu().numpy()[0]
    
    
    ### Evaluate the trained PRISE agent's success rate
    def evaluate(self):
        self.agent.train(False)
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        eval_env, task_name = self.eval_env, self.cfg.downstream_task_name
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
        print('Success Rate:{}%'.format(success/self.cfg.num_eval_episodes*100))
        self.performance.append(success/self.cfg.num_eval_episodes*100)
        ### Store the evaluated success rate into the pickle file
        if self.rank == 0:
            with open(self.eval_dir / '{}.pkl'.format(self.cfg.exp_bc_name), 'wb') as f:
                pickle.dump(self.performance, f)
        self.agent.train(True)
    
    
    ###### Stage 1: Pretrain action quantization
    def pretrain_models(self):
        metrics = None
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nPretraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s.")
                if metrics is not None:
                    # log stats
                    print('DYNAMICS_LOSS:{}, QUANTIZE_LOSS:{}, DECODER_LOSS:{}'.format(metrics['dynamics_loss'], metrics['quantize_loss'], metrics['decoder_loss']))
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('total_time', total_time)
                        log('step', self.global_step)

                # save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage)

            self._global_step += 1
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')

        dest_log_dir = self.results_dir / 'logs'
        distutils.dir_util.copy_tree(str(self.logger._log_dir), str(dest_log_dir))

    
    ###### Stage 2: Use BPE to compute skill tokens
    def train_bpe(self):
        self.agent.n_code = self.cfg.n_code
        self.agent.train(False)
        lst_traj = []
        for task_dir in self.pretraining_data_dirs:
            lst_traj.extend(utils.choose(list(sorted(task_dir.glob('*.npz'))), self.cfg.max_traj_per_task))
        print('Loaded {} trajectories'.format(len(lst_traj)))
        with torch.no_grad():
            corpus, counter = [], 0
            for f in lst_traj:
                counter += 1
                episode = np.load(f)
                action    = episode['action'][1:]
                action  = torch.torch.as_tensor(action, device=self.device)
                obs_history = utils.compute_traj_latent_embedding(episode, device=self.device, nstep_history=self.cfg.nstep_history)
                z = self.agent.encode_history(obs_history, aug=False)
                z = self.agent.compute_transformer_embedding(z)
                u = self.agent.PRISE.module.action_encoder(z, action.float())
                _, _, _, _, codes = self.agent.PRISE.module.a_quantizer(u)
                codes = list(codes.reshape(-1).detach().cpu().numpy())
                codes = [int(idx) for idx in codes]
                corpus.append(codes)

                if counter % 100 == 0:
                    print(f"Processed {counter} trajectories")

            print('=========Offline Data Tokenized!==========')

            ### Train tokenizer on the tokenized pretraining trajectories
            tokenizer = Tokenizer(algo='bpe', vocab_size=self.cfg.vocab_size)
            tokenizer.train(corpus, min_frequency=self.cfg.min_frequency, max_token_length=self.cfg.max_token_length, verbose=True)
        
        ### Save pretrained tokenizer
        vocab_dir = self.results_dir / 'vocab'
        vocab_dir.mkdir(parents=True, exist_ok=True)
        with open(vocab_dir / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'wb') as f:
            pickle.dump([tokenizer, corpus], f)
    
    
    ##### Stage 3: Adapt to the downstream tasks
    def downstream_adapt(self):
        self.agent.train(False)
        ### set the hyperparameters for downstream adaptation
        self.agent.alpha  = self.cfg.alpha
        self.agent.PRISE.module.decoder.decoder_loss_coef = self.cfg.decoder_loss_coef
        
        ################## Load the BPE-Learned vocabulary #################
        vocab_dir = self.results_dir / 'vocab'
        with open(vocab_dir / 'vocab_mt45_code{}_vocab{}_minfreq{}_maxtoken{}.pkl'.format(self.cfg.n_code, self.cfg.vocab_size, self.cfg.min_frequency, self.cfg.max_token_length), 'rb') as f:
            loaded_data = pickle.load(f)
            self.tokenizer, corpus = loaded_data
            self.agent.tokenizer = self.tokenizer
        
        ################## Tokenize the downstream data #################
        print("========= Tokenizing the downstream data... ==========")
        self.tok_to_idx = {}
        self.idx_to_tok = []
        replay_buffer = self.replay_loader.dataset
        for episode in replay_buffer._episodes.values():
            with torch.no_grad():
                task_embedding    = episode['task_embedding']
                if self.eval_env is not None:
                    self.eval_env.task_embedding = task_embedding[None,:]
                action    = episode['action'][1:]
                action  = torch.torch.as_tensor(action, device=self.device)
                obs_history = utils.compute_traj_latent_embedding(episode, device=self.device, nstep_history=self.cfg.nstep_history)
                z = self.agent.encode_history(obs_history, aug=False)
                u = self.agent.PRISE.module.action_encoder(z, action.float())
                _, _, _, _, codes = self.agent.PRISE.module.a_quantizer(u)
                codes = list(codes.reshape(-1).detach().cpu().numpy())
                codes = [int(idx) for idx in codes]
                traj_tok = [self.tokenizer.encode(codes[t:], verbose=False)[0] for t in range(len(codes))]
                episode['token'] = traj_tok
                for tok in traj_tok:
                    if not tok in self.tok_to_idx:
                        self.tok_to_idx[tok] = len(self.tok_to_idx)
                        self.idx_to_tok.append(tok)
        self.agent.idx_to_tok = self.idx_to_tok
        self.agent.tok_to_idx = self.tok_to_idx
        print("========= Downstream data tokenized !!! ==========")
        
        
        ################## Initialize the model (skill token policy) ###############
        print(f"========= Initiaizing the model... ==========")
        # Initialize the skill token policy
        token_policy = nn.Sequential(
            nn.Linear(self.cfg.feature_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, len(self.tok_to_idx))
        ).to(self.device)
        token_policy.train(True)
        token_policy.apply(utils.weight_init)
        self.agent.PRISE.module.token_policy = token_policy
        self.agent.prise_opt = torch.optim.Adam(self.agent.PRISE.parameters(), lr=self.cfg.lr)
        tok_to_code = lambda tok: self.tokenizer.decode([int(tok.item())], verbose=False) # Token =>  First Code
        tok_to_idx  = lambda tok: self.tok_to_idx[int(tok.item())] # Token => Index

        
        ################## Finetune the model #####################
        print(f"========= Finetuning for {self.cfg.num_train_steps} steps... ==========")
        metrics = None
        start_train_block_time = time.time()
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%self.cfg.eval_freq == 0 and self.rank == 0:
                print(f"\nTraining for {self.global_step} steps of {self.cfg.batch_size}-sized batches has takes {time.time() - start_train_block_time}s (including eval time).")
                if metrics is not None:
                    # log stats
                    print('DECODER_LOSS:{}, SKILL_TOKEN_POLICY_LOSS:{}'.format(metrics['decoder_loss'], metrics['token_policy_loss']))
                    elapsed_time, total_time = self.timer.reset()

                # save snapshot
                if self.cfg.save_snapshot and self.rank == 0:
                    self.save_snapshot(self.cfg.stage, ckpt=self.global_step//self.cfg.eval_freq)

            metrics = self.agent.downstream_adapt(self.replay_iter, tok_to_code, tok_to_idx, self.idx_to_tok, finetune_decoder=self.cfg.finetune_decoder)

            if self.global_step>5000 and self.global_step%self.cfg.eval_freq == 0:
                if self.cfg.eval:
                    start_eval_block_time = time.time()
                    self.evaluate()
                    print(f"Evaluation on {self.cfg.num_eval_episodes} episodes took {time.time() - start_eval_block_time}s.")
            
            self._global_step += 1

    
    ### saving the model checkpoint
    def save_snapshot(self, stage, ckpt=None):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if stage == 1:
            snapshot = 'snapshot'
        elif stage == 2:
            snapshot = 'snapshot_vocab{}'.format(self.cfg.vocab_size)
        else:
            snapshot = 'snapshot_vocab{}_{}'.format(self.cfg.vocab_size, self.cfg.seed)
        if ckpt is not None:
            snapshot += '_ckpt{}'.format(ckpt)
        snapshot += '.pt'
        
        if stage < 3:
            snapshot = self.results_dir / snapshot
        else:
            snapshot = self.eval_dir / snapshot

        keys_to_save = ['agent', '_global_step']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.results_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            print(self.device)
            payload = torch.load(f, map_location=f'cuda:{self.device}')
        self.__dict__['agent'] = payload['agent']
        if self.cfg.stage == 1:
            self.__dict__['_global_step'] = payload['_global_step']
        self.agent.device = self.device
        self.agent.PRISE.device = self.device
        self.agent.PRISE.to(self.device)
        print('Resuming Snapshopt')

RANK = None
WORLD_SIZE = None

@hydra.main(config_path='cfgs', config_name='prise_config')
def main(cfg):
    global RANK, WORLD_SIZE
    ddp_setup(RANK, WORLD_SIZE, cfg.port)
    from train_prise import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, RANK, WORLD_SIZE)
    root_dir = Path.cwd()
    snapshot = root_dir / 'snapshot.pt'
    if cfg.load_snapshot:
        if snapshot.exists() and cfg.stage > 1:
            print(f'resuming: {snapshot}')
            workspace.load_snapshot()

    if cfg.stage == 1:
        workspace.pretrain_models()
    elif cfg.stage == 2:
        workspace.train_bpe()
    elif cfg.stage == 3:
        workspace.downstream_adapt()
    else:
        raise ValueError(f"Invalid stage: {cfg.stage}")
    destroy_process_group()

def wrapper(rank, world_size, cfg):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = world_size
    print(f'WORLD SIZE: {world_size}, RANK: {rank}')
    main(cfg)

def main_mp_launch_helper(cfg=None):
    world_size = torch.cuda.device_count()
    if world_size==1:  # If it is single GPU, don't use multiprocessing
        wrapper(0, 1, cfg)  
    else:
        mp.spawn(wrapper, args=(world_size, cfg), nprocs=world_size)

if __name__ == '__main__':
    main_mp_launch_helper()
