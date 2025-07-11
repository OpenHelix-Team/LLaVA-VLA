#!/bin/bash
# trap "kill 0" EXIT
# source ~/user68/conda_env/llava/bin/activate
CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9097))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  llava/serve/flask_server_qwen.py \
        --model-path  /home/lg5/project/LLaVA-VLA/llava-onevision-siglip-so400m-patch14-384-qwen2_0.5b-calvin-rel-obs-reduce5-abc2d_4epoch \
        --action_stat /data1/songwx/calvin/dataset/calvin_debug_dataset/training/statistics.yaml \
        --port $port  \
        > ./log/debug.log 2>&1 &
done

# wait
