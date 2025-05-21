# Install



1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Clone and isntall Calvin
```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
conda create -n calvin_venv python=3.8  # or use virtualenv
conda activate calvin_venv
sh install.sh
```

3. Download Calvin  dataset
```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh D | ABC | ABCD | debug
```
4. Preprocess Calvin dataset
```bash
cd vlas/scripts
python /data/user/wsong890/user68/project/vlas/scripts/helper/calvin2json.py
```

# Train LLaVA
```bash

cd vlas
bash ./scripts/train/calvin_finetune_obs.sh
```
calvin_finetune_obs.sh
```bash
#!/bin/bash
source ~/user68/conda_env/llava/bin/activate 
# export CUDA_VISIBLE_DEVICES=0,1,2
which python
echo $PATH
export WANDB_MODE=offline
export WANDB_DIR=./wandb

export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
export MODEL_NAME_OR_PATH=/data/user/wsong890/user68/project/vlas/llava-v1.5-7b
export OUTPUT_DIR=./checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abcd2d_2024_03_14
export CALVIN_PROCESSED_JSON_PATH=/data/user/wsong890/user68/data/calvin/calvin_processed_json
export CALVIN_PROCESSED_DIRECTORY=/data/user/wsong890/user68/data/calvin_process/task_ABCD_D/vla_processed_r5
export ACTION_STAT=/data/user/wsong890/user68/data/statistics.yaml
export VISION_TOWER=/data/user/wsong890/user68/project/clip-vit-large-patch14-336
export DEEPSPEED_CONFIG=/data/user/wsong890/user68/project/vlas/scripts/zero3.json

deepspeed --include=localhost:0,1 /data/user/wsong890/user68/project/vlas/llava/train/calvin_train_obs.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --data_path $CALVIN_PROCESSED_JSON_PATH \
    --image_folder $CALVIN_PROCESSED_DIRECTORY \
    --action_stat $ACTION_STAT \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --report_to_wandb_project your_project_name \
    --report_to_wandb_run_name your_run_name

```

# Evaluate LLaVA in Calvin

start model server on you own port(here is 9097)

```bash
bash  /data/user/wsong890/user68/project/vlas/scripts/server/start_multi_server.sh
```

start_multi_server.sh
```bash
#!/bin/bash
# trap "kill 0" EXIT
CUDA_VISIBLE_DEVICES=0
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
wait

```
start calvin client
```bash
bash /data/user/wsong890/user68/project/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy_multiserver.sh

```
evaluate_policy_multiserver.sh

```bash
#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
source ~/user68/conda_env/calvin_venv/bin/activate
export EGL_VISIBLE_DEVICES=0
trap "kill 0" EXIT
PORTSLIST=(9097)
export PYTHONPATH=/data/user/wsong890/user68/project/calvin:$PYTHONPATH
export PYTHONPATH=/data/user/wsong890/user68/project/calvin/calvin_env:$PYTHONPATH
export PYTHONPATH=/data/user/wsong890/user68/project/calvin/calvin/calvin_env/tacto:$PYTHONPATH
EVAL_LOG_DIR="/data/user/wsong890/user68/project/calvin/calvin_models/calvin_agent/evaluation/log"
CHUNKS=${#PORTSLIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    python /data/user/wsong890/user68/project/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy_multiserver.py \
        --dataset_path /share/user/iperror/data/calvin/task_ABCD_D \
        --question_file  /data/user/wsong890/user68/project/calvin/calvin_models/calvin_agent/evaluation/evaluation_sequence/questions/question.json\
        --eval_log_dir $EVAL_LOG_DIR \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --port ${PORTSLIST[$IDX]} \
        --custom_model 
done

wait

```

