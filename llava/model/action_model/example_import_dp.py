from llava.model.action_model.action_model import ActionModel
from llava.model.action_model.models import DiT



# 定义一个DP动作模型
action_model = ActionModel(
    model_type = 'DiT-B',                                     # 设定网络层数等，固定用基础模型即可
    token_size = 4096,                                        # 假设最后一个hidden_state的形状为(B, 1, D), 这里的4096指的就是最后的D的大小 
    in_channels = 7,                                          # action的维度，目前默认是7，即(x, y, z, euler_x, euler_y, euler_z, gripper)
    future_action_window_size = 15,                           # diffusion预测的未来时间窗口大小（不包含当前步），比如咋们一次预测16个step，那么future_action_window_size=15
    past_action_window_size = 0                               # 固定为0即可
    )
print(action_model)

# 在前向loss计算过程中的使用
def forward(last_hidden, label_action):
    # last_hidden的shape需要为: (B, 1, D_h=4096), label_action的shape需要为: (B, T=16, D_a=7)
    last_hidden_repeated = last_hidden.repeat(8, 1, 1)        # [repeated_diffusion_steps*B, 1, D], 这里的8是diffusion的步数，固定用这个值即可
    label_action_repeated = label_action.repeat(8, 1, 1)      # [repeated_diffusion_steps*B, T, D]

    loss = self.action_model.loss(label_action_repeated, last_hidden_repeated)


# 在预测推理预测action的过程中的使用，可以考虑使用一个开关符号统一到forward过程中
def predict_action(last_hidden):
    # last_hidden的shape需要为: (B, 1, D_h=4096), 推理的B要求是1
    # 1. 降噪前准备
    cognition_features = last_hidden
    noise = torch.randn(B, 16, 7, device=cognition_features.device).to(dtype)   # [B, T, D_a], 这里的dtype指定为合适的精度数据类型即可, [B, T, D_a]的大小就是action的大小

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
    real_actions = denormalize(normalized_actions)                     # 这个denormalize函数需要使用统计量自己实现