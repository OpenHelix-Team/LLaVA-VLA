#!/bin/bash
# trap "kill 0" EXIT
# source ~/user68/conda_env/llava/bin/activate
CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/user/wsong890/user68/project/vlas-qwen:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9097))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  llava/serve/flask_server_action_head.py \
        --model-path  /data/user/wsong890/user68/project/vlas-qwen/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-action_head-0721/checkpoint-8374\
        --action_stat /share/user/iperror/data/task_ABC_D/training/statistics.yaml \
        --port $port &
done

# wait