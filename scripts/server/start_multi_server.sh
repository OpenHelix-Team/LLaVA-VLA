#!/bin/bash
# trap "kill 0" EXIT
source ~/user68/conda_env/llava/bin/activate
CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9097))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  ./llava/serve/flask_server.py \
        --model-path  ./checkpoints/llava_checkpoint-21572\
        --action_stat /share/user/iperror/data/calvin/task_ABCD_D/training/statistics.yaml \
        --port $port 
done

# wait