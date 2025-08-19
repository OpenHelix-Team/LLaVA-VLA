#!/bin/bash
source ~/user68/conda_env/llava/bin/activate 
# export CUDA_VISIBLE_DEVICES=0,1,2
which python
echo $PATH
export WANDB_MODE=offline
export WANDB_DIR=./wandb
export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
export MODEL_NAME_OR_PATH=/yourpath/llava-v1.5-7b
export OUTPUT_DIR=./checkpoints
export CALVIN_PROCESSED_JSON_PATH=yourpath/calvin/calvin_processed_json
export CALVIN_PROCESSED_DIRECTORY=/yourpath/calvin_process/task_ABCD_D/vla_processed_r5
export ACTION_STAT=/yourpath/calvin/dataset/task_ABCD_D/training/statistics.yaml
export VISION_TOWER=/yourpath/clip-vit-large-patch14-336
export DEEPSPEED_CONFIG=scripts/zero3.json
# Please replace 'yourpath' with your actual path!
deepspeed --include=localhost:0,1 llava/train/calvin_train_obs.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --data_path $CALVIN_PROCESSED_JSON_PATH \
    --image_folder $CALVIN_PROCESSED_DIRECTORY \
    --action_stat $ACTION_STAT \
    --vision_tower $VISION_TOWER \
    --bin_size 256 \
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
