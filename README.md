<a href="https://arxiv.org/abs/2503.01115"><img src="https://img.shields.io/badge/arXiv-2503.01115-b31b1b.svg" height=22.5></a>

# WeGen: A Unified Model for Interactive Multimodal Generation as We Chat

This repo is the official implementation of "*[WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](https://arxiv.org/abs/2503.01115)*", by Zhipeng Huang, Shaobin Zhuang, Canmiao Fu, Binxin Yang, Ying Zhang, Chong Sun, Zhizheng Zhang, Yali Wang, Chen Li, Zheng-Jun Zha


**WeGen** is a unified framework that integrates multimodal understanding and generation, enabling users to achieve various visual generation goals through natural conversation. It excels at generating diverse results with high creativity for less detailed instructions and can progressively refine prior generation results while maintaining consistency with user references.

### Key Features

- Unified Framework: Seamlessly integrates diverse capabilities including text-to-image generation, subject-driven generation, condition-driven generation, image restoration, and style transfer
- Dynamic Instance Identity Consistency (DIIC): Maintains instance identity consistency while allowing natural variations in generated contents


### Demo
coming soon.


### Installation
 
1. Clone the repository:
```bash
git clone https://github.com/hzphzp/WeGen.git
cd WeGen/
```

2. Prepare the base enviroment, we use ubuntu20, python3.8, with H20 or 910B GPUs

3. Install required packages:
```bash
bash env.sh
```

4. Download the pre-trained models from [here](https://drive.google.com/drive/folders/1HOpbP9T7ovBTDpC1mci-cvSzfBwwMvEL?usp=drive_link) and construct the pretrained model folder like:

```bash
WeGen
└── wegen_mllm_ckpt
    ├── pretrained
    │   ├── CLIPScore_eval
    │   ├── EVA-CLIP
    │   ├── SEED-X
    │   ├── meta-llama 
    │   │   └── Llama-2-7b-chat-hf
    │   └── stable-diffusion-xl-base-1.0
    ├── pytorch_model.bin
    ├── stage1_final
    │   └── unet
    └── stage2_final
        └── checkpoint-30000
```


### Data preparation

DIIC dataset coming soon.


### Training
run the following command to train the model on 128 H20/910B GPUs Node:
```bash
# stage1
bash scripts/wegen_mllm_stage1.sh
# stage2
bash scripts/wegen_mllm_stage2.sh
# stage3
bash scripts/wegen_mllm_stage3.sh
```

### Evaluation
run the following command to evaluate the model on 8 H20/910B GPUs Node:
```bash
bash scripts/inference.sh
```



## Citing
If you find this code and work useful, please consider **citing** the following paper and **star** this repo. Thank you very much!
```
@article{huang2025wegen,
  title={WeGen: A Unified Model for Interactive Multimodal Generation as We Chat},
  author={Huang, Zhipeng and Zhuang, Shaobin and Fu, Canmiao and Yang, Binxin and Zhang, Ying and Sun, Chong and Zhang, Zhizheng and Wang, Yali and Li, Chen and Zha, Zheng-Jun},
  journal={arXiv preprint arXiv:2503.01115},
  year={2025}
}
```
 
