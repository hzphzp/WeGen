echo "start inference"
export RESULT_PATH="./logs/test_results/"
export CKPT_PATH="./wegen_mllm_ckpt/pytorch_model.bin"

# FID
torchrun --nproc_per_node=8 \
    --nnodes 1 \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/eval/inference.py \
    --total_eval_samples 30000 \
    --validation_result_path=${RESULT_PATH}/fid/ \
    --ckpt_path=${CKPT_PATH} \
    --guidance_scale 2 \
    --var_cfg_guidance_scale 1 


# clip-t
torchrun --nproc_per_node=8 \
    --nnodes 1 \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/eval/inference.py \
    --total_eval_samples 4096 \
    --validation_result_path=${RESULT_PATH}/clip-t/ \
    --ckpt_path=${CKPT_PATH} \
    --guidance_scale 7 \
    --var_cfg_guidance_scale 1 


python ./src/eval/test_fid.py --result_path=${RESULT_PATH}/fid/images

python ./src/eval/test_clipt.py --result_path=${RESULT_PATH}/clip-t/images