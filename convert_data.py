### Convert the original h5py file into the format of PRISE's dataloader.
import h5py
import numpy as np
from pathlib import Path
import datetime
import re
import io
from transformers import BertTokenizer, BertModel
import torch

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def convert_frame_stack(obs, frame_stack=1):
    if frame_stack > 1 and frame_stack <= obs.shape[0]:
        stacked_obs = np.empty((obs.shape[0], obs.shape[1]*frame_stack, obs.shape[2], obs.shape[3]))

        # Loop over the observations and stack the frames with repetition for early timesteps
        for i in range(obs.shape[0]):
            for j in range(frame_stack):
                # Determine the index for the frame to be stacked
                frame_index = max(i-j, 0)
                stacked_obs[i, j*obs.shape[1]:(j+1)*obs.shape[1], :, :] = obs[frame_index, :, :, :]
    else:
        stacked_obs = obs
    return stacked_obs

### Compute the bert embedding of the task description
def get_task_embedding(task_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = tokenizer.encode(task_name, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        outputs = model(input_tensor)
    last_hidden_states = outputs[0]
    sentence_embedding = last_hidden_states[:, 0, :]
    return sentence_embedding

def extract_task_information(file_name, libero_path):
    """
    Extracts task information from the given file name.
    """
    # Regular expression pattern to extract the task name
    pattern = r'{}/((.+)_SCENE[0-9]+_(.+))_demo\.hdf5'.format(libero_path)

    # Extracting the task name
    match = re.search(pattern, file_name)
    
    task_embedding = get_task_embedding(match.group(3).lower().replace("_", " "))
    print(match.group(3).lower().replace("_", " "))
    return match.group(1).lower() if match else None, task_embedding

def convert_hdf5_file_to_npz(task_name, task_embedding, demo_data,save_path, frame_stack=1):
    num_demos = len(demo_data)
    
    for i in range(num_demos):
        ### Get agent's view camera
        obs = np.array(demo_data['demo_{}'.format(i)]['obs']['agentview_rgb'])
        obs = convert_frame_stack(obs.transpose(0,3,1,2), frame_stack)
        
        ### Get wrist's view camera
        obs_wrist = np.array(demo_data['demo_{}'.format(i)]['obs']['eye_in_hand_rgb'])
        obs_wrist = convert_frame_stack(obs_wrist.transpose(0,3,1,2), frame_stack)
        
        ### Get task embedding
        task_embedding_vector = np.array(task_embedding).reshape(-1)
        
        ### Get actions
        action = np.array(demo_data['demo_{}'.format(i)]['actions'])
        action = np.vstack([np.zeros_like(action[:1]), action[:-1]])
        
        ### Get the 9 dimensional state info
        prop_info_lst  = ['gripper_states', 'joint_states']
        state = []
        for prop_info in prop_info_lst:
            state.append(np.array(demo_data['demo_{}'.format(i)]['obs'][prop_info]))
        state = np.hstack(state)
        
        episode = {'observation':obs,
                   'observation_wrist':obs_wrist,
                   'state': state,
                   'task_embedding': task_embedding_vector,
                   'action':action, 
                  }
        fn = save_path / f'{task_name.replace("_", "")}_{i}_{obs.shape[0]}.npz'
        save_episode(episode, fn)
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./', help='path to store converted LIBERO data')
    parser.add_argument('--libero_path', type=str, default='./', help='path where LIBERO-90/10 dataset is stored')
    parser.add_argument('--frame_stack', type=int, default=1)
    args = parser.parse_args()
    
    for path in list(Path(args.libero_path).iterdir()):
        path_name = str(path)
        task_name, task_embedding = extract_task_information(path_name, args.libero_path)
        print('Processing Task Name:{}'.format(task_name))
        
        save_path = Path(f'{args.save_path}/{task_name}_framestack{args.frame_stack}')
        save_path.mkdir(parents=True, exist_ok=True)
        
        demo_data = h5py.File(path_name, 'r')['data']
        convert_hdf5_file_to_npz(task_name, task_embedding, demo_data,save_path, frame_stack=args.frame_stack)