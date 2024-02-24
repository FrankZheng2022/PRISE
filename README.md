# PRISE: Learning Temporal Action Abstractions as a Sequence Compression Problem
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/pdf/2402.10450.pdf">[Paper]</a>&emsp;<a href="">[Project Website]</a>
</p>

In this work, we propose a novel view that treats inducing temporal action abstractions as a sequence compression problem. To do so, we bring a subtle but critical component of LLM training pipelines -- input tokenization via byte pair encoding (BPE) -- to the seemingly distant task of learning skills of variable time span in continuous control domains. 

We introduce an approach called Primitive Sequence Encoding (**PRISE**) that combines continuous action quantization with BPE to learn powerful action abstractions. We empirically show that high-level skills discovered by **PRISE** from a multitask set of robotic manipulation demonstrations significantly boost the performance of both multitask imitation learning as well as few-shot imitation learning on unseen tasks. 

This thread contains implementation of **PRISE** on **MetaWorld**. Check out the main thread to see PRISE implementation on **LIBERO**.

# 🛠️ Installation Instructions

**Step 1: Setup Environment**: 
```
conda create -n prise python=3.8.13
conda activate prise
pip install -r requirements.txt
pip install -e .
```



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
By default, we set voabulary size to be 200. The learned BPE tokenizer will be saved under the directory ``exp_local/${EXP_NAME}/vocab_mt45_code${N_CODE}_vocab${VOCAB_SIZE}_minfreq10_maxtoken20.pkl``


**Stage III: Downstream Adaptation**:


***Few-shot Adaptation to unseen tasks (5-shots)***:
To adapt to an unseen task with five expert demonstration trajectories from the five hold out tasks ``['hand-insert', 'box-close', 'stick-pull', 'disassemble', 'pick-place-wall']``:
```
python train_prise.py exp_name=${EXP_NAME} replay_buffer_num_workers=4 batch_size=256 stage=3 exp_bc_name=${DOWNSTREAM_EXP_NAME} downstream_task_name=${TASK_ID} num_train_steps=30010 eval_freq=2000 max_traj_per_task=5 vocab_size=${VOCAB_SIZE} &
```


## 📝 Citation

If you find our method or code relevant to your research, please consider citing the paper as follows:

```
@misc{zheng2024prise,
      title={PRISE: Learning Temporal Action Abstractions as a Sequence Compression Problem}, 
      author={Ruijie Zheng and Ching-An Cheng and Hal Daumé III au2 and Furong Huang and Andrey Kolobov},
      year={2024},
      eprint={2402.10450},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



