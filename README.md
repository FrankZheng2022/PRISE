# PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/pdf/2402.10450.pdf">[Paper]</a>&emsp;<a href="">[Project Website]</a>
</p>

In this work, we propose a novel view that treats inducing temporal action abstractions as a sequence compression problem. To do so, we bring a subtle but critical component of LLM training pipelines -- input tokenization via byte pair encoding (BPE) -- to the seemingly distant task of learning skills of variable time span in continuous control domains. 

We introduce an approach called Primitive Sequence Encoding (**PRISE**) that combines continuous action quantization with BPE to learn powerful action abstractions. We empirically show that high-level skills discovered by **PRISE** from a multitask set of robotic manipulation demonstrations significantly boost the performance of both multitask imitation learning as well as few-shot imitation learning on unseen tasks. 

This thread contains implementation of **PRISE** on **LIBERO**. Check out the **MetaWorld** thread to see PRISE implementation on **MetaWorld** with slight modification compared to **LIBERO**.

<p align="center">
  <video loop autoplay muted src="images/prise_demo.mp4" style="max-width:100%;"></video>
</p>

# 🛠️ Installation Instructions

**Step 1: Setup Environment**: 
```
conda create -n prise python=3.8.13
conda activate prise
pip install -r requirements.txt
pip install -e .
```

**Step 2: Install LIBERO**: 
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

**Step 3: Download and Preprocess LIBERO Dataset**: 
First, download the LIBERO dataset.
```
cd LIBERO
python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```
Next, we need to convert the format of the LIBERO dataset. Copy the ``convert_data.py`` file into the LIBERO repo, and then run
```
python convert_data.py --save_path ${DATASET_PATH} --libero_path ${LIBERO_PATH}/LIBERO/libero/datasets/libero_90
python convert_data.py --save_path ${DATASET_PATH} --libero_path ${LIBERO_PATH}/LIBERO/libero/datasets/libero_10
```
Here ${DATASET_PATH} is the path of the libero dataset that you are going to store.
After you converting the data, rewrite ``libero_path`` in ``cfgs/prise_config.yaml`` to ${LIBERO_PATH}, and ``data_storage_dir`` to ${DATASET_PATH}.



## 💻 Code Usage

**Stage I: Pretrain PRISE action vector quantization**: 
```
python train_prise.py exp_name=${EXP_NAME} stage=1 n_code=${N_CODE} save_snapshot=true &
```
The model checkpoint and loss information is saved under the directory ``exp_local/${EXP_NAME}``. By default, we set number of quantized codes to be 10. In terms of the computational resources to train the first stage of PRISE, we use 4 NVIDIA A100 GPU w. 40G memory and 400 GB CPU memory. 


**Stage II: Run BPE tokenization algorithm to get skill tokens**: 
```
python train_prise.py exp_name=${EXP_NAME} stage=2 vocab_size=${VOCAB_SIZE} &
```
By default, we set voabulary size to be 200. The learned BPE tokenizer will be saved under the directory ``exp_local/${EXP_NAME}/vocab_libero90_code${N_CODE}_vocab${VOCAB_SIZE}_minfreq10_maxtoken20.pkl``. The second stage should take ~10 minutes to finish on a single A100 GPU.


**Stage III: Downstream Adaptation**:

***Case I: Multitask Learning***:
To train a multitask generalist policy on LIBERO-90:
```
python train_prise.py exp_name=${EXP_NAME} replay_buffer_num_workers=4 stage=3 downstream_exp_name=${DOWNSTREAM_EXP_NAME} multitask=true downstream_task_suite=libero_90 num_train_steps=30010 eval=false save_snapshot=true vocab_size=${VOCAB_SIZE} &
```
To evaluate the trained multitask policy:
```
python eval_libero90.py exp_name=${EXP_NAME} &
```


***Case II: Few-shot Adaptation to unseen tasks (5-shots)***:
To adapt to an unseen task with five expert demonstration trajectories:
```
python train_prise.py exp_name=${EXP_NAME} replay_buffer_num_workers=4 batch_size=64 stage=3 exp_bc_name=${DOWNSTREAM_EXP_NAME} downstream_task_name=${TASK_ID} downstream_task_suite=libero_10 num_train_steps=30010 eval_freq=2000 max_traj_per_task=5 vocab_size=${VOCAB_SIZE} &
```


## 📝 Citation

If you find our method or code relevant to your research, please consider citing the paper as follows:

```
@inproceedings{
zheng2024prise,
title={PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control},
author={Ruijie Zheng and Ching-An Cheng and Hal Daum{\'e} III and Furong Huang and Andrey Kolobov},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=p225Od0aYt}
}
```



