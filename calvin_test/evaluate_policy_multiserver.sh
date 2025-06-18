#!/bin/bash
## activate the calvin environment
conda activate calvin_venv
## choose the gpu
export EGL_VISIBLE_DEVICES=0
trap "kill 0" EXIT

## set the port list , the evaluation sequence is split into chunks, each chunk is evaluated by a server
PORTSLIST=(9097 9098)
SERVER_IP="127.0.0.1"
EVAL_LOG_DIR="./calvin_test/evaluation/log"
CHUNKS=${#PORTSLIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    python ./calvin_test/evaluate_policy_multiserver.py \
        --dataset_path ./calvin/dataset/task_ABC_D \
        --question_file  ./calvin_test/question.json \
        --eval_log_dir $EVAL_LOG_DIR \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --port ${PORTSLIST[$IDX]} \
        --server_ip $SERVER_IP \
        --save_dir ./debug_video \
        --action_chunk 5 \
        --conf_dir ./calvin/calvin_models/conf \
        --custom_model &
done
        # --debug \
#save_name: 测试文件结果的名称的前缀，后缀自动为运行时间
#save_dir: 在debug模式下，测试视频结果的保存路径
#如果模型端和服务器端不在一台机器上，要设置一下网口位置


wait