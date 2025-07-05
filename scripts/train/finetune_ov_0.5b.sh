#!/bin/bash
# wandb login 3cc75ad549e94669d4b230f5cdea68473279dc08
# conda activate llava-next
# export PYTHONPATH=/data/user/wsong890/user68/project/rossvla:$PYTHONPATH
set -x
# export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-11.7
# export PATH=/hpc2ssd/softwares/cuda/cuda-11.7/bin:$PATH
# export LD_LIBRARY_PATH=/hpc2ssd/softwares/cuda/cuda-11.7/lib64:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=0,1
# Set WandB offline
export WANDB_MODE=offline
export WANDB_DIR=./wandb




LLM_VERSION="Qwen/Qwen2-0.5B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION=<your_path_to_siglip-so400m-patch14-384>
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
# Stage 2
PROMPT_VERSION="qwen_2"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9-qwen2-0.5b" 
PREV_STAGE_CHECKPOINT="<your_path_to_lmms-labllava-onevision-qwen2-0.5b-ov>" # 
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"
# export CUDA_VISIBLE_DEVICES=0,1
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port="20228" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /data/user/wsong890/user68/project/vlas/playground/task_ABC_D_training_r5.json \
    --image_folder /data/user/wsong890/user68/data/calvin_process/task_ABC_D/vla_processed_r5 \
    --action_stat /data/user/wsong890/user68/data/statistics.yaml \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/onevision/$RUN_NAME \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    > ./log/train_0.5b.log 2>&1 &
    # --image_aspect_ratio anyres_max_9 \
    # --image_grid_pinpoints  "(1x1),...,(6x6)" \
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
