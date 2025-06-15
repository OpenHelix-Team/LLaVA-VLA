# LLaVA-VLA: A Simple Yet Powerful Vision-Language-Action Model

We introduce **LLaVA-VLA**, an open-source Vision-Language-Action model built upon the popular open-source VLM [LLaVA](https://github.com/haotian-liu/LLaVA). This implementation combines accessibility with strong performance for robotic manipulation tasks. The key features lie in:

1. 🏗️ **Minimalist design** - It is a vanilla VLA architecture without performance-hacking components. It is designed for easy modification and educational use. And it is also an ideal baseline for new researchers in embodied AI.
2. 🏆 **Competitive performance** - It achieves 3.68% success rate on [CALVIN](https://github.com/mees/calvin) benchmark, which outperforms the most popular OpenVLA.
3. 💸 **Efficient training** - It does not need pre-training on large-scale robot dataset. It only requires 7h fine-tuning from LLaVA-v1.5 checkpoint.
4. 🔌 **Seamless Extensibility** - It is built on the widely-used LLaVA ecosystem. It fosters easy technology transfer to derivative projects.
5. 🔄 **Active maintenance** - We will continuously improve it with new functions and environments.


## 🔥News

- **2025.06.15** 🌟 We are proud to introduce LLaVA, a model designed to balabala.

## 📌Contents
- [Model Overview](#model-overview)
- [Experimental Results](#Experimental-Results)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experimental Results](#experimental-results)
- [TODO](#todo)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)
- [Citation](#citation)

## 🛠️Model Overview
<a id="model-overview"></a>
![Model Architecture](./images/pipline.png)
The network architecture of our LLaVA-VLA. Given images, proprioception and language instructions, our method first tokenizes the input and then feeds the results into the LLM. The LLM outputs an action chunking, which are finally detokenized into valid action values and deployed on the mechanical arm.

## 📊Experimental Results
<a id="experimental-results"></a>
LLaVA-VLA's performance is shown in the following table:
![Result Visualization](./images/exp.png)


## 💾Installation
<a id="installation"></a>

### Dependencies

#### Python versions:
- Python 3.8, 3.10

#### Operating systems:
- Linux: Ubuntu 18.04+, CentOS 7+

#### Hardware:
- Rendering: NVIDIA or AMD GPU
  - Recommended CUDA Version: 12.1
- Ray tracing: NVIDIA RTX GPU or AMD equivalent
- Ray-tracing Denoising: NVIDIA GPU
- GPU Simulation: NVIDIA GPU

#### Software:
- Ray tracing: NVIDIA Driver >= 470
- Denoising (OIDN): NVIDIA Driver >= 520
- CUDA Version: 12.1

### Basic Env
1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  
pip install -e .
```
### Dataset
2. Clone and isntall Calvin
```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
conda create -n calvin_venv python=3.8  
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
python yourpath/calvin2json.py
```

## 📈Training
<a id="training"></a>
```bash
cd vlas
bash yourpath/scripts/train/calvin_finetune_obs.sh
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
export MODEL_NAME_OR_PATH=yourpath/vlas/llava-v1.5-7b
export OUTPUT_DIR=yourpath
export CALVIN_PROCESSED_JSON_PATH=yourpath/data/calvin/calvin_processed_json
export CALVIN_PROCESSED_DIRECTORY=yourpath/data/calvin_process/task_ABCD_D/vla_processed_r5
export ACTION_STAT=yourpath/data/statistics.yaml
export VISION_TOWER=yourpath/project/clip-vit-large-patch14-336
export DEEPSPEED_CONFIG=yourpath/project/vlas/scripts/zero3.json

deepspeed --include=localhost:0,1 yourpath/llava/train/calvin_train_obs.py \
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

## 🔬Evaluation
<a id="evaluation"></a>
start model server on you own port(here is 9097)，
CUDA_VISIBLE_DEVICES specifies the number of GPUs (e.g., if you have two GPUs, it would be 0,1).
```bash
bash  yourpath/vlas/scripts/server/start_multi_server.sh
```

start_multi_server.sh

start calvin client
```bash
bash yourpath/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy_multiserver.sh

```
evaluate_policy_multiserver.sh

## 🎓TODO
<a id="todo"></a>
abc

## 🙏Acknowledgement
<a id="acknowledgement"></a>
abc

## ✉️Contact
<a id="contact"></a>
abc

## 📑Citation
<a id="citation"></a>
abc

