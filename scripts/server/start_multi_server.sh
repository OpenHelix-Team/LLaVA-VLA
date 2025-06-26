conda acitavate llava
CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9097))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  ./llava/serve/flask_server.py \
        --model-path  ./yourpath/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2epoch \
        --action_stat ./calvin/task_ABCD_D/training/statistics.yaml \
        --port $port & 
done

# Please replace 'yourpath' with your actual path!