#!/bin/bash
# trap "kill 0" EXIT
# source ~/user68/conda_env/llava/bin/activate
CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9097))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  llava/serve/flask_server_qwen.py \
        --model-path  /data/user/wsong890/user68/project/LLaVA-NeXT/checkpoints/onevision/llava-onevision-_data_user_wsong890_user68_project_rossvla_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9-qwen2-7b-no_anyres-0703/checkpoint-16744 \
        --action_stat /share/user/iperror/data/task_ABC_D/training/statistics.yaml \
        --port $port  \
        > ./log/debug.log 2>&1 &
done

# wait
#/data/user/wsong890/user68/project/LLaVA-NeXT/checkpoints/onevision/llava-onevision-_data_user_wsong890_user68_project_rossvla_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_stage_am9-qwen2-7b-no_anyres/checkpoint-61160