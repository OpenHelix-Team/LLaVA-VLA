#!/bin/bash
source ~/user68/conda_env/llava-pipper/bin/activate 
# export CUDA_VISIBLE_DEVICES=2,3
which python
echo $PATH
export WANDB_MODE=offline
export WANDB_DIR=./wandb
# export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
# export WANDB_API_KEY=local-73e66a04fa97e2a5d5c573a97e65bf1194533e1f
# export WANDB_BASE_URL=http://10.28.0.22:30437/
# export PYTHONPATH=/data/user/user68/conda_env/llava/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=/data/user/wsong890/user68/project/lerobot-piper:$PYTHONPATH
export PYTHONPATH=/data/user/wsong890/user68/project/vlas:$PYTHONPATH
deepspeed --include=localhost:0,1  /data/user/wsong890/user68/project/vlas/llava/train/calvin_train_obs_real_pos_single_val_clean.py \
    --deepspeed /data/user/wsong890/user68/project/vlas/scripts/zero3.json \
    --model_name_or_path /data/user/wsong890/user68/project/vlas/llava-v1.5-7b \
    --version v1 \
    --data_path /data/user/wsong890/user68/project/lerobot-piper/data/new_task03 \
    --image_folder /data/user/wsong890/user68/data/calvin_process/task_ABCD_D/vla_processed_r5 \
    --action_stat /data/user/wsong890/user68/project/vlas/stats/merged_stat_position.json \
    --vision_tower /data/user/wsong890/user68/project/clip-vit-large-patch14-336\
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /data/user/wsong890/user68/project/vlas/checkpoints/llava-v1.5-7b-lerobot-0515_singlenewtask03_pos_val_clean_save_steps50 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
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
    --report_to_wandb_project llava-lerobot \
    --report_to_wandb_run_name llava-lerobot-task_all_0515_singlenewtask03_pos_val_clean_save_steps50
