source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export HCCL_RDMA_TIMEOUT=20
export ASCEND_PROCESS_LOG_PATH=/root/tmp_plog/

# avoid hccl timeout
export ASCEND_WORK_PATH=/tmp
export ASCEND_WORK_PATH=/tmp
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_ENTRY_LOG_ENABLE=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export HCCL_RDMA_TIMEOUT=20
export HCCL_CONNECT_TIMEOUT=3600
echo '--------------------------'
ps -elf | grep python3.8 | grep -v grep | awk '{print $4}' | xargs kill -9


echo "start training"
PROJ_PATH='.'

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-23456}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


exp_name='wegen_mllm_stage2'
OUTPUT_PATH=${PROJ_PATH}/training_logs/wegen_mllm/${exp_name}

mkdir -p $OUTPUT_PATH
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    ${PROJ_PATH}/src/train/train_wegen_mllm.py \
    --image_transform ${PROJ_PATH}/configs/processer/clip_l_448_transform_resize.yaml \
    --image_transform2 ${PROJ_PATH}/configs/processer/sd_transform_resize.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml \
    --diffusion_model_path stabilityai/stable-diffusion-xl-base-1.0 \
    --diffusion_specific_unet_path ./wegen_mllm_ckpt/stage1_final/unet \
    --adapter_cfg_path configs/sdxl_adapter/sdxl_evaclip_vit_avgpool_q64_fullft_minisnr5_freeze_unet.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/eva_vision.yaml \
    --llm_model ${PROJ_PATH}/configs/clm_models/llm_wegen_mllm.yaml \
    --agent_model ${PROJ_PATH}/configs/clm_models/agent_wegen_mllm.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/sft_comp_gen_bs32_scaleup_cfg.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name  ${exp_name} \
    --learning_rate 5e-4 \
    --batch_size 4096 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 2 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 30000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 1000 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 0 \
    --training_sample_steps 1000 \
    --eval_steps 1 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2.yaml \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --auto_resume True \


echo '--------------------------'
echo main training task done
echo '--------------------------'
