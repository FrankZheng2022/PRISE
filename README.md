# PRISE: Learning Temporal Action Abstractions as a Sequence Compression Problem
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/pdf/2402.10450.pdf">[Paper]</a>&emsp;<a href="">[Project Website]</a>
</p>

In this work, we propose a novel view that treats inducing temporal action abstractions as a sequence compression problem. To do so, we bring a subtle but critical component of LLM training pipelines -- input tokenization via byte pair encoding (BPE) -- to the seemingly distant task of learning skills of variable time span in continuous control domains. 

We introduce an approach called Primitive Sequence Encoding (**PRISE**) that combines continuous action quantization with BPE to learn powerful action abstractions. We empirically show that high-level skills discovered by **PRISE** from a multitask set of robotic manipulation demonstrations significantly boost the performance of both multitask imitation learning as well as few-shot imitation learning on unseen tasks. 

### MetaWorld Implementation Coming Soon!
This thread contains implementation of **PRISE** on **MetaWorld**. Check out the main thread to see PRISE implementation on **LIBERO**.


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



