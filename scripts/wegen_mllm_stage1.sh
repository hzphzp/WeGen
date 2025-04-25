source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export HCCL_RDMA_TIMEOUT=20
export ASCEND_PROCESS_LOG_PATH=/root/tmp_plog/
echo '--------------------------'

echo "start training"
PROJ_PATH='.'

exp_name='wegen_mllm_stage1'
OUTPUT_PATH=${PROJ_PATH}/training_logs/wegen_mllm/${exp_name}


# For IB
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_PORT=${MASTER_PORT:-23456}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export ASCEND_PROCESS_LOG_PATH=${OUTPUT_PATH}/ascend/log/$INDEX
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_ENTRY_LOG_ENABLE=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export HCCL_RDMA_TIMEOUT=20
# --deepspeed_multinode_launcher standard

mkdir -p $OUTPUT_PATH
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    ${PROJ_PATH}/src/train/train_detokenizer.py \
    --image_transform ${PROJ_PATH}/configs/processer/clip_l_448_transform_crop.yaml \
    --image_transform2 ${PROJ_PATH}/configs/processer/sd_transform_crop.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml \
    --diffusion_model_path stabilityai/stable-diffusion-xl-base-1.0 \
    --diffusion_specific_unet_path ./wegen_mllm_ckpt/stage1_final/unet \
    --adapter_cfg_path configs/sdxl_adapter/sdxl_evaclip_vit_avgpool_q64_fullft_minisnr5_train_unet.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/eva_vision.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/sdxl_adapter_finetune.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name  ${exp_name} \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 160000 \
    --save_steps 5000 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 0 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2.yaml \
    --eval_steps 5000 \
    --training_sample_steps 5000 \
    --warmup_resampler_steps 12000 \


echo '--------------------------'
echo main training task done
echo '--------------------------'
