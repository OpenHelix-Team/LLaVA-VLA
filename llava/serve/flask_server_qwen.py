from flask import Flask, jsonify, request, Response
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.action_tokenizer import ActionTokenizer, encode_robot_obs
from llava import conversation as conversation_lib
# from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
import argparse
import os
import socket
import io
import json
import numpy as np
import torch
from PIL import Image
from functools import partial
import time
import copy
from safetensors.torch import load_file
TARGET_IMG_SIZE = 334  # NOTE need to be consistent with that in calvin2json.py



class LLMRobotServer:
    def __init__(self, args, from_train=False):
        # if from_train:
        #     self.tokenizer=args.tokenizer
        #     self.llm_robot=args.llm_robot
        #     self.image_processor=args.image_processor
        #     # self.context_len=args.context_len
        # else:
        print("args.model_path:",args.model_path)
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        device_map = None
        device="cuda"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "flash_attention_2",
        }
        # model_name = "llava_qwen"
        # print("model_name:",model_name)
        model_base = args.model_base
        self.tokenizer, self.llm_robot, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name,**llava_model_args)
        # state_dict = load_file("/data/user/wsong890/user68/project/lmms-labllava-onevision-qwen2-0.5b-ov/model.safetensors",device="cuda")
        # print("state_dict.keys:",state_dict.keys())

        # self.llm_robot.to(device)
        # print("self.llm_robot.vision_tower:",self.llm_robot.get_vision_tower())
        # 手动加载 .safetensors 文件
    #     vison_model_path = "/data/user/wsong890/user68/project/rossvla/siglip-so400m-patch14-384/model.safetensors"
    #     safetensor_state_dict = load_file(vison_model_path)

    #     # 从 llm_robot 中获取 vision_tower 的参数
    #    # 2. 获取模型中的 vision_model（模块）
    #     vision_model = self.llm_robot.get_vision_tower().vision_tower

    # #     # 3. 遍历 vision_model 中的参数，对比值
    #     print("🔍 比较 vision_model 参数与 safetensors 中是否一致：")
    #     for name, param in vision_model.named_parameters():
    #         if name not in safetensor_state_dict:
    #             print(f"❌ Not found in safetensors: {name}")
    #             continue

    #         safetensor_param = safetensor_state_dict[name].detach().cpu()
    #         model_param = param.detach().cpu()

    #         if model_param.shape != safetensor_param.shape:
    #             print(f"❌ {name} shape mismatch: {model_param.shape} vs {safetensor_param.shape}")
    #         elif not torch.allclose(model_param.float(), safetensor_param.float(), atol=1e-6):
    #             print(f"❌ {name} value mismatch")
    #             print("   model_param[:5]:", model_param.view(-1)[:5])
    #             print("   safetensor_param[:5]:", safetensor_param.view(-1)[:5])
    #         else:
    #             print(f"✅ {name} values match")
        

        # print("lm_head.weight" in state_dict)  # True表示有保存
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.action_stat = args.action_stat

    def compose_robot_input(
        self, img_static, img_gripper, instruction, robot_obs, debug=True
    ):
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        image_sizes = [img_concat.size]
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        if debug:
            img_concat.save("./debug_img.png", "PNG")
        # print("self.image_processor:",self.image_processor)
        # The image height is equal to the width, thus no pad or square
        image_tensor = process_images([img_concat], self.image_processor, self.llm_robot.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
        print("image_tensor[0]:",image_tensor[0].shape)
        # print("img_concat.shape:",img_concat.shape)
        # image_tensor = self.image_processor.preprocess(img_concat, return_tensors="pt")[
        #     "pixel_values"
        # ][0]
        # image_tensor = image_tensor[None, :]
        # print("image_tensor.shape:",image_tensor.shape)

        system_message = "A chat between a curious human and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        tokenizer = copy.deepcopy(self.tokenizer)
  
        tokenizer.add_tokens(["<image>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        input_id = tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        robot_obs = [str(elem) for elem in robot_obs]
        robot_obs = " ".join(robot_obs)
        robot_obs_token, robot_obs  = encode_robot_obs(robot_obs, self.action_tokenizer, self.action_stat)
        print("instruction:",instruction)
        instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + robot_obs
        conv = [{"role" : "user", "content" : instruction}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        conv = [{"role" : "assistant", "content" : ""}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        for idx, encode_id in enumerate(input_id):
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        
        input_id = input_id[:-2]
        index_198 = np.where(np.array(input_id) == 198)[0]
        input_id[index_198[-2]-16:index_198[-2]-1] = robot_obs_token
        input_ids=[input_id]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        

        print("input_id为", input_id)
        return input_ids, image_tensor,image_sizes#  image_tensor[torch(3,336,336)]

    def robot_action_generate(self, input_ids, images,image_sizes):
            """生成机器人动作并记录推理时间和速度。"""
            time0 = time.time()
            with torch.inference_mode():
                output_ids = self.llm_robot.generate(
                    input_ids.cuda(),
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )
            time1 = time.time()

            generate_time = time1 - time0
            num_tokens = output_ids.shape[1]
            ar_speed_time = num_tokens / generate_time

            # 记录推理时间和速度
            # log_file = "/home/lg5/project/vlas/llava/serve.jsonl"
            # log_data = {
            #     # "timestamp": datetime.now().isoformat(),
            #     "generate_time": round(generate_time, 4),
            #     "tokens_per_second": round(ar_speed_time, 2),
            #     # "num_tokens": num_tokens
            # }
            # with open(log_file, "a") as f:
            #     f.write(json.dumps(log_data) + "\n")

            # 处理输出
            print("output_ids:",output_ids)
            output_ids = output_ids[0].cpu().numpy().tolist()[:35]
            # print("output_ids:",output_ids)
            actions = [self.action_tokenizer.decode_token_ids_to_actions(elem) for elem in output_ids]
            # print("actions:",actions)
            return np.array(actions)
def run_llm_robot_server_from_train(model,tokenizer,image_processor):
    flask_app = Flask(__name__)
    args = argparse.Namespace()
    args.tokenizer = tokenizer
    args.llm_robot = model
    args.image_processor = image_processor
    args.temperature = 0.0
    args.top_p = None
    args.num_beams = 1
    args.max_new_tokens = 256
    args.action_stat = "/data/user/wsong890/user68/data/statistics.yaml"
    args.port = 9097
    llm_robot = LLMRobotServer(args,from_train=True)

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            img_static = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8)
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            img_static = Image.fromarray(img_static)
            img_gripper = Image.fromarray(img_gripper)

            input_ids, images,image_sizes = llm_robot.compose_robot_input(
                img_static, img_gripper, instruction, robot_obs
            )
            action = llm_robot.robot_action_generate(input_ids, images,image_sizes)
            print(action)
            return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    inference_times = []
    torch.cuda.reset_peak_memory_stats()
    memory_samples = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--action_stat",
        type=str,
        default="/home/wenxuansong/chenjy/data/dataset/task_ABC_D/training/statistics.yaml",
    )
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--log_file", type=str, default="/data/user/user68/project/vlas/log")
    args = parser.parse_args()

    flask_app = Flask(__name__)
    llm_robot = LLMRobotServer(args)

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            img_static = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8)
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            img_static = Image.fromarray(img_static)
            img_gripper = Image.fromarray(img_gripper)

            input_ids, images,image_sizes = llm_robot.compose_robot_input(
                img_static, img_gripper, instruction, robot_obs
            )
            action = llm_robot.robot_action_generate(input_ids, images,image_sizes)
            print(action)
            return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)

    #数据记录
    # import statistics
    # median_time = statistics.median(inference_times)
    # print(f"Median time for generating: {median_time}")
    # print(f"推理次数: {len(inference_times)}")
    # # 计算平均显存使用
    # avg_memory = sum(memory_samples) / len(memory_samples)
    # # 获取峰值显存使用
    # peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    # print(f"平均显存使用: {avg_memory:.2f} MB")
    # print(f"显存使用峰值: {peak_memory:.2f} MB")

