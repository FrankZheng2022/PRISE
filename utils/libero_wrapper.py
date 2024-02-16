from collections import deque
import numpy as np
import random
import gym
from gym.wrappers import TimeLimit
import dm_env
from dm_env import specs
from typing import Any, NamedTuple
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from transformers import BertTokenizer, BertModel
import torch
import os

class ExtendedTimeStep(NamedTuple):
    done: Any
    reward: Any
    discount: Any
    agentview: Any
    wristview: Any
    state: Any
    action: Any

    def last(self):
        return self.done

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        obs = self._env.reset()
        return self._augment_time_step(obs['agentview'], obs['wristview'], obs['state'])

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        discount = 1.0
        return self._augment_time_step(obs['agentview'],
                                       obs['wristview'],
                                       obs['state'],
                                       action,
                                       reward,
                                       discount,
                                       done)
    
    
    def _augment_time_step(self, agentview, wristview, state, action=None, reward=None, discount=1.0, done=False):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            reward = 0.0
            discount = 1.0
            done = False
        return ExtendedTimeStep(agentview=agentview,
                                wristview=wristview,
                                state=state,
                                action=action,
                                reward=reward,
                                discount=discount,
                                done = done)
    
    def state_spec(self):
        return specs.BoundedArray((9,), np.float32, name='state', minimum=0, maximum=255)
    
    def observation_spec(self):
        return specs.BoundedArray(self._env.observation_space.shape, np.float32, name='observation', minimum=0, maximum=255)
    
    def agentview_spec(self):
        return specs.BoundedArray(self._env.observation_space.shape, np.uint8, name='agentview', minimum=0, maximum=255)

    def wristview_spec(self):
        return specs.BoundedArray(self._env.observation_space.shape, np.uint8, name='wristview', minimum=0, maximum=255)
    
    def action_spec(self):
        return specs.BoundedArray(self._env.action_space.shape, np.float32, name='action', minimum=-1, maximum=1.0)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
class LiberoWrapper(gym.Wrapper):
    def __init__(self, env, img_size, seed):
        super().__init__(env)
        self.env = env
        env.seed(seed)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, img_size, img_size),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=255,
            shape=(7,),
            dtype=np.float32,
        )
        self._res = (img_size, img_size)
        self.img_size = img_size
        self.seed = seed
        self._state = None
    
    def _get_pixel_obs(self, pixel_obs):
        return pixel_obs[:, :, ::-1].transpose(
            2, 0, 1
        )
    
    def reset(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.env.task_suite_name]()
        
        init_states = task_suite.get_task_init_states(self.env.task_id) 
        init_state = random.choice(init_states)
        self.env.set_init_state(init_state)
        state = self.env.reset()
        self._state = state
        obs = {}
        
        #### Concatenate prop state info
        prop_info_lst  = ['robot0_gripper_qpos', 'robot0_joint_pos']
        prop_state_lst = []
        for key in prop_info_lst:
            prop_state_lst.append(state[key])
        prop_state = np.concatenate(prop_state_lst)
        obs['state'] = prop_state
        obs['agentview'] = self._get_pixel_obs(state['agentview_image'])
        obs['wristview'] = self._get_pixel_obs(state['robot0_eye_in_hand_image'])
        
        for _ in range(5):  # simulate the physics without any actions
            self.env.step(np.zeros(7))
        
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._state = state
        
        obs = {}
        #### Concatenate prop state info
        prop_info_lst  = ['robot0_gripper_qpos', 'robot0_joint_pos']
        prop_state_lst = []
        for key in prop_info_lst:
            prop_state_lst.append(state[key])
        prop_state = np.concatenate(prop_state_lst)
        obs['state'] = prop_state
        
        obs['agentview'] = self._get_pixel_obs(state['agentview_image'])
        obs['wristview'] = self._get_pixel_obs(state['robot0_eye_in_hand_image'])
        return obs, reward, done, info

    def render(self):
        return self._state['agentview_image']

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(task_id, task_suite_name, seed=1, train=False, img_size=128, device_id=-1, libero_path="/fs/cml-projects/taco_rl/LIBERO"):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(f"{libero_path}/libero/libero/bddl_files", task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
          f"language instruction is {task_description}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_size,
        "camera_widths": img_size,
        "render_gpu_device_id":device_id
    }
    env = OffScreenRenderEnv(**env_args)
    env.task_id = task_id
    env.task_name = task_name
    env.task_suite_name = task_suite_name
    env = LiberoWrapper(env, img_size, seed)
    #env = TimeLimit(env, max_episode_steps=episode_length)
    env = ExtendedTimeStepWrapper(env)
    return env


### Compute the bert embedding of the task description
def get_task_embedding(task_description):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = tokenizer.encode(task_description, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    last_hidden_states = outputs[0]
    sentence_embedding = last_hidden_states[:, 0, :]
    return sentence_embedding