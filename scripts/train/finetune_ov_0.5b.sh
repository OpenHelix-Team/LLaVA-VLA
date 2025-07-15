#!/bin/bash

set -x

# ========== Environment Settings ==========
export WANDB_MODE=offline
export WANDB_DIR=./wandb

# ========== Model Versions ==========
LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/your/path/to/siglip-so400m-patch14-384"  # 🔧 Set your actual vision model path here(https://huggingface.co/google/siglip-so400m-patch14-384)
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION="qwen_2"

# ========== Data and Checkpoint Paths ==========
DATA_PATH="/data/user/wsong890/user68/project/vlas/playground/task_ABC_D_training_r5.json" # 🔧 Set your data json path afther running calvin2json.py
IMAGE_FOLDER="/data/user/wsong890/user68/data/calvin_process/task_ABC_D/vla_processed_r5" # 🔧 Set your  image path afther running calvin2json.py
ACTION_STAT="/data1/songwx/calvin/dataset/calvin_abcd/training/statistics.yaml " # 🔧 Set your statistics.yaml of calvin(ABD-D,ABCD-D,Debug dataset is equal)
PREV_STAGE_CHECKPOINT="/home/lg5/project/LLaVA-VLA/llava-onevision-qwen2-0.5b-ov" # 🔧 Set your LLaVA base model path here(https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov)

# ========== Output and Logging ==========
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9-qwen2-0.5b"
# LOG_FILE="./log/train_0.5b.log"
OUTPUT_DIR="./checkpoints/onevision/$RUN_NAME"

# ========== Launch Training ==========
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"
export CUDA_VISIBLE_DEVICES=0,1 #need to set the same number of nproc_per_node
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port="20228" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --action_stat $ACTION_STAT \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
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
    > ./log/train_0.5b.log 2>&1 & \
    # --image_aspect_ratio anyres_max_9 \
    # --image_grid_pinpoints  "(1x1),...,(6x6)" \

exit 0
