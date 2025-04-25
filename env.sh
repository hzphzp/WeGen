#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
export HCCL_RDMA_TIMEOUT=20

pip install huggingface_hub==0.24.0
pip install transformers_stream_generator
pip install hydra-core
pip install pyrootutils
pip install torchdata
pip install wandb
pip install kornia
pip install transformers==4.30.2
pip install diffusers==0.25.0
pip install deepspeed==0.15.0
pip install torchdata==0.7.1
pip install -U peft
pip uninstall -y bitsandbytes
pip install accelerate==0.25.0
pip install protobuf==3.20.0
pip install json5 
pip install mmcv==2.1.0
pip install -U mmagic
pip install torchmetrics==1.3.1
pip install pycocotools
pip install dreamsim
