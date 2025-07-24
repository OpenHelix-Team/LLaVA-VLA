#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.action_model.action_model import ActionModel
from llava.model.action_model.models import DiT

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    use_diffusion_head: bool = False
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_llama"
        # config.rope_scaling = None

        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.use_diffusion_head = config.use_diffusion_head
        if config.use_diffusion_head:
            self.action_model = ActionModel(
                model_type = 'DiT-B',                                     # 设定网络层数等，固定用基础模型即可
                token_size = 4096,                                        # 假设最后一个hidden_state的形状为(B, 1, D), 这里的4096指的就是最后的D的大小 
                in_channels = 7,                                          # action的维度，目前默认是7，即(x, y, z, euler_x, euler_y, euler_z, gripper)
                future_action_window_size = 4,                           # diffusion预测的未来时间窗口大小（不包含当前步），比如咋们一次预测16个step，那么future_action_window_size=15
                past_action_window_size = 0                               # 固定为0即可
                )
        else:
            self.action_model = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        action_lables: Optional[torch.FloatTensor] = None,
        cache_position=None,
        last_token_index=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                labels, 
                images, 
                modalities, 
                image_sizes)
        if  self.use_diffusion_head:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            assert last_token_index is not None
            # print("outputs.keys():",outputs.keys())
            last_hidden = outputs["last_hidden_state"]
            # print("last_hidden.shape:",last_hidden.shape)
            # print("last_token_index:",last_token_index)
            last_token_index = last_token_index.expand(-1, 1, last_hidden.size(2))
            # print("last_token_index",last_token_index)
            # print("last_token_index.shape:",last_token_index.shape)
            last_token_hidden = torch.gather(last_hidden, dim=1, index=last_token_index)
            # print("last_token_hidden.shape:",last_token_hidden.shape)
            # print("last_hidden.shape:",last_hidden.shape)
                # def forward(last_hidden, label_action):
            # last_hidden的shape需要为: (B, 1, D_h=4096), label_action的shape需要为: (B, T=16, D_a=7)
            last_token_hidden_repeated = last_token_hidden.repeat(8, 1, 1)        # [repeated_diffusion_steps*B, 1, D], 这里的8是diffusion的步数，固定用这个值即可
            label_action_repeated = action_lables.repeat(8, 1, 1)      # [repeated_diffusion_steps*B, T, D]
            # print("last_hidden_repeated.dtype:",last_hidden_repeated.dtype)
            # print("label_action_repeated.dtype:",label_action_repeated.dtype)
            loss = self.action_model.loss(label_action_repeated, last_token_hidden_repeated)
            return {"loss": loss}

        
        if dpo_forward:
            print("use dpo_forward")
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            ans=super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("lm_head weight mean:", self.lm_head.weight.data.abs().mean())
            # print("ans.logits.shape:",ans.logits.shape)
            # logits=ans.logits[:,-37:]
            # print("logits.shape:",logits.shape)
            # print("logits:",torch.argmax(logits,dim=-1))
            # # print("")
            # print("ans.keys():",ans.keys())
            # print("ans.logits.shape:",ans.logits.shape)
            # print("ans.loss:",ans.loss)
            return ans

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
            print("inputs_embeds.shape:",inputs_embeds.shape)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    def predict_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        action_lables: Optional[torch.FloatTensor] = None,
        cache_position=None,
            ):
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                labels, 
                images, 
                modalities, 
                image_sizes)
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("outputs.keys():",outputs.keys())
        last_hidden = outputs["last_hidden_state"][:, -1:, :]
        # last_hidden的shape需要为: (B, 1, D_h=4096), 推理的B要求是1
        # 1. 降噪前准备
        B=1
        cognition_features = last_hidden
        noise = torch.randn(B, 5, 7, device=cognition_features.device).to(cognition_features.dtype)   # [B, T, D_a], 这里的dtype指定为合适的精度数据类型即可, [B, T, D_a]的大小就是action的大小

        ### 2. 一种优化的降噪方式，无需修改，固定使用即可
        noise = torch.cat([noise, noise], 0)
        uncondition = self.action_model.net.z_embedder.uncondition
        uncondition = uncondition.unsqueeze(0)                               #[1, D]
        uncondition = uncondition.expand(B, 1, -1)                           #[B, 1, D]
        z = torch.cat([cognition_features, uncondition], 0)
        cfg_scale = 1.5
        model_kwargs = dict(z=z, cfg_scale=cfg_scale)
        sample_fn = self.action_model.net.forward_with_cfg

        use_ddim = True
        num_ddim_steps = 10
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0
                )
        samples, _ = samples.chunk(2, dim=0)                               # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()
        

        ### 3. 此时就得到了normalized_actions，需要反归一化到真实的action空间中
        # real_actions = denormalize(normalized_actions)                     # 这个denormalize函数需要使用统计量自己实现
        return normalized_actions
    


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
