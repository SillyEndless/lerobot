#!/usr/bin/env python3
"""
基于源码的神经网络结构图、流程图和函数调用关系图
"""

import graphviz

def create_neural_network_structure():
    """基于源码的神经网络结构图"""
    
    dot = graphviz.Digraph('Neural_Network_Structure',
                          comment='基于源码的神经网络结构图',
                          format='png')
    
    dot.attr(rankdir='TB', size='16,20', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # DiffusionPolicy类
    with dot.subgraph(name='cluster_diffusion_policy') as c:
        c.attr(label='DiffusionPolicy类', style='filled', color='lightblue')
        
        c.node('diffusion_policy', 'DiffusionPolicy\n(主策略类)', fillcolor='lightcyan')
        c.node('normalize_inputs', 'normalize_inputs\n(输入归一化)', fillcolor='lightcyan')
        c.node('normalize_targets', 'normalize_targets\n(目标归一化)', fillcolor='lightcyan')
        c.node('unnormalize_outputs', 'unnormalize_outputs\n(输出反归一化)', fillcolor='lightcyan')
        c.node('_queues', '_queues\n(观察和动作队列)', fillcolor='lightcyan')
    
    # DiffusionModel类
    with dot.subgraph(name='cluster_diffusion_model') as c:
        c.attr(label='DiffusionModel类', style='filled', color='lightgreen')
        
        c.node('diffusion_model', 'DiffusionModel\n(核心扩散模型)', fillcolor='lightgreen')
        c.node('unet', 'DiffusionConditionalUnet1d\n(条件U-Net)', fillcolor='lightgreen')
        c.node('noise_scheduler', 'noise_scheduler\n(DDPM/DDIM调度器)', fillcolor='lightgreen')
        c.node('rgb_encoder', 'DiffusionRgbEncoder\n(视觉编码器)', fillcolor='lightgreen')
    
    # DiffusionRgbEncoder类
    with dot.subgraph(name='cluster_rgb_encoder') as c:
        c.attr(label='DiffusionRgbEncoder类', style='filled', color='lightyellow')
        
        c.node('backbone', 'ResNet Backbone\n(torchvision.models)', fillcolor='lightyellow')
        c.node('spatial_softmax', 'SpatialSoftmax\n(空间软最大值)', fillcolor='lightyellow')
        c.node('final_linear', 'Linear + ReLU\n(最终线性层)', fillcolor='lightyellow')
    
    # DiffusionConditionalUnet1d类
    with dot.subgraph(name='cluster_unet') as c:
        c.attr(label='DiffusionConditionalUnet1d类', style='filled', color='lightcoral')
        
        c.node('timestep_encoder', 'DiffusionSinusoidalPosEmb\n(时间步编码器)', fillcolor='lightcoral')
        c.node('down_modules', 'down_modules\n(编码器模块)', fillcolor='lightcoral')
        c.node('mid_modules', 'mid_modules\n(中间模块)', fillcolor='lightcoral')
        c.node('up_modules', 'up_modules\n(解码器模块)', fillcolor='lightcoral')
        c.node('final_conv', 'final_conv\n(最终卷积)', fillcolor='lightcoral')
    
    # DiffusionConditionalResidualBlock1d类
    with dot.subgraph(name='cluster_residual_block') as c:
        c.attr(label='DiffusionConditionalResidualBlock1d类', style='filled', color='lightpink')
        
        c.node('conv1', 'Conv1d + GroupNorm + Mish\n(第一个卷积块)', fillcolor='lightpink')
        c.node('film_encoder', 'FiLM编码器\n(条件调制)', fillcolor='lightpink')
        c.node('conv2', 'Conv1d + GroupNorm + Mish\n(第二个卷积块)', fillcolor='lightpink')
        c.node('residual_conv', 'residual_conv\n(残差连接)', fillcolor='lightpink')
    
    # 连接关系
    # DiffusionPolicy内部
    dot.edge('diffusion_policy', 'normalize_inputs')
    dot.edge('diffusion_policy', 'normalize_targets')
    dot.edge('diffusion_policy', 'unnormalize_outputs')
    dot.edge('diffusion_policy', '_queues')
    dot.edge('diffusion_policy', 'diffusion_model')
    
    # DiffusionModel内部
    dot.edge('diffusion_model', 'unet')
    dot.edge('diffusion_model', 'noise_scheduler')
    dot.edge('diffusion_model', 'rgb_encoder')
    
    # RGB编码器内部
    dot.edge('rgb_encoder', 'backbone')
    dot.edge('backbone', 'spatial_softmax')
    dot.edge('spatial_softmax', 'final_linear')
    
    # U-Net内部
    dot.edge('unet', 'timestep_encoder')
    dot.edge('unet', 'down_modules')
    dot.edge('unet', 'mid_modules')
    dot.edge('unet', 'up_modules')
    dot.edge('unet', 'final_conv')
    
    # 残差块内部
    dot.edge('conv1', 'film_encoder')
    dot.edge('film_encoder', 'conv2')
    dot.edge('conv2', 'residual_conv')
    
    dot.render('./images/neural_network_structure', view=True, cleanup=True)
    print("神经网络结构图已生成: ./images/neural_network_structure.png")

def create_training_flow():
    """训练流程图"""
    
    dot = graphviz.Digraph('Training_Flow',
                          comment='训练流程图',
                          format='png')
    
    dot.attr(rankdir='TB', size='14,18', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 数据准备
    with dot.subgraph(name='cluster_data_prep') as c:
        c.attr(label='数据准备', style='filled', color='lightblue')
        
        c.node('batch', '批次数据\n(batch)', fillcolor='lightcyan')
        c.node('normalize_inputs', 'normalize_inputs\n(输入归一化)', fillcolor='lightcyan')
        c.node('normalize_targets', 'normalize_targets\n(目标归一化)', fillcolor='lightcyan')
        c.node('stack_images', '图像堆叠\n(torch.stack)', fillcolor='lightcyan')
    
    # 特征编码
    with dot.subgraph(name='cluster_feature_encoding') as c:
        c.attr(label='特征编码', style='filled', color='lightgreen')
        
        c.node('prepare_global_cond', '_prepare_global_conditioning\n(准备全局条件)', fillcolor='lightgreen')
        c.node('rgb_encoder', 'DiffusionRgbEncoder\n(视觉编码)', fillcolor='lightgreen')
        c.node('global_cond', '全局条件特征\n(global_cond)', fillcolor='lightgreen')
    
    # 扩散过程
    with dot.subgraph(name='cluster_diffusion') as c:
        c.attr(label='扩散过程', style='filled', color='lightyellow')
        
        c.node('trajectory', '动作轨迹\n(trajectory)', fillcolor='lightyellow')
        c.node('sample_noise', '采样噪声\n(torch.randn)', fillcolor='lightyellow')
        c.node('sample_timesteps', '采样时间步\n(torch.randint)', fillcolor='lightyellow')
        c.node('add_noise', 'add_noise\n(添加噪声)', fillcolor='lightyellow')
        c.node('noisy_trajectory', '噪声轨迹\n(noisy_trajectory)', fillcolor='lightyellow')
    
    # 模型预测
    with dot.subgraph(name='cluster_prediction') as c:
        c.attr(label='模型预测', style='filled', color='lightcoral')
        
        c.node('unet_forward', 'unet.forward\n(U-Net前向传播)', fillcolor='lightcoral')
        c.node('prediction', '预测输出\n(pred)', fillcolor='lightcoral')
    
    # 损失计算
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='损失计算', style='filled', color='lightpink')
        
        c.node('compute_loss', 'compute_loss\n(计算损失)', fillcolor='lightpink')
        c.node('mse_loss', 'F.mse_loss\n(MSE损失)', fillcolor='lightpink')
        c.node('mask_loss', '掩码损失\n(可选)', fillcolor='lightpink')
        c.node('final_loss', '最终损失\n(loss.mean())', fillcolor='lightpink')
    
    # 连接关系
    # 数据准备流程
    dot.edge('batch', 'normalize_inputs')
    dot.edge('normalize_inputs', 'stack_images')
    dot.edge('stack_images', 'normalize_targets')
    
    # 特征编码流程
    dot.edge('normalize_targets', 'prepare_global_cond')
    dot.edge('prepare_global_cond', 'rgb_encoder')
    dot.edge('rgb_encoder', 'global_cond')
    
    # 扩散流程
    dot.edge('normalize_targets', 'trajectory')
    dot.edge('trajectory', 'sample_noise')
    dot.edge('trajectory', 'sample_timesteps')
    dot.edge('sample_noise', 'add_noise')
    dot.edge('sample_timesteps', 'add_noise')
    dot.edge('add_noise', 'noisy_trajectory')
    
    # 预测流程
    dot.edge('noisy_trajectory', 'unet_forward')
    dot.edge('global_cond', 'unet_forward')
    dot.edge('unet_forward', 'prediction')
    
    # 损失计算流程
    dot.edge('prediction', 'compute_loss')
    dot.edge('compute_loss', 'mse_loss')
    dot.edge('mse_loss', 'mask_loss')
    dot.edge('mask_loss', 'final_loss')
    
    dot.render('./images/training_flow', view=True, cleanup=True)
    print("训练流程图已生成: ./images/training_flow.png")

def create_inference_flow():
    """推理流程图"""
    
    dot = graphviz.Digraph('Inference_Flow',
                          comment='推理流程图',
                          format='png')
    
    dot.attr(rankdir='TB', size='14,18', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 观察输入
    with dot.subgraph(name='cluster_observation') as c:
        c.attr(label='观察输入', style='filled', color='lightblue')
        
        c.node('obs_batch', '观察批次\n(observation batch)', fillcolor='lightcyan')
        c.node('normalize_inputs', 'normalize_inputs\n(输入归一化)', fillcolor='lightcyan')
        c.node('stack_images', '图像堆叠\n(torch.stack)', fillcolor='lightcyan')
        c.node('populate_queues', 'populate_queues\n(填充队列)', fillcolor='lightcyan')
    
    # 队列管理
    with dot.subgraph(name='cluster_queues') as c:
        c.attr(label='队列管理', style='filled', color='lightgreen')
        
        c.node('obs_queue', '观察队列\n(observation queue)', fillcolor='lightgreen')
        c.node('action_queue', '动作队列\n(action queue)', fillcolor='lightgreen')
        c.node('check_action_queue', '检查动作队列\n(len(action_queue) == 0)', fillcolor='lightgreen')
    
    # 动作生成
    with dot.subgraph(name='cluster_action_generation') as c:
        c.attr(label='动作生成', style='filled', color='lightyellow')
        
        c.node('predict_action_chunk', 'predict_action_chunk\n(预测动作块)', fillcolor='lightyellow')
        c.node('generate_actions', 'generate_actions\n(生成动作)', fillcolor='lightyellow')
        c.node('conditional_sample', 'conditional_sample\n(条件采样)', fillcolor='lightyellow')
        c.node('extend_action_queue', 'extend动作队列\n(extend)', fillcolor='lightyellow')
    
    # 动作选择
    with dot.subgraph(name='cluster_action_selection') as c:
        c.attr(label='动作选择', style='filled', color='lightcoral')
        
        c.node('popleft_action', 'popleft动作\n(popleft)', fillcolor='lightcoral')
        c.node('unnormalize_outputs', 'unnormalize_outputs\n(输出反归一化)', fillcolor='lightcoral')
        c.node('final_action', '最终动作\n(final action)', fillcolor='lightcoral')
    
    # 连接关系
    # 观察处理流程
    dot.edge('obs_batch', 'normalize_inputs')
    dot.edge('normalize_inputs', 'stack_images')
    dot.edge('stack_images', 'populate_queues')
    dot.edge('populate_queues', 'obs_queue')
    
    # 队列检查流程
    dot.edge('obs_queue', 'check_action_queue')
    dot.edge('check_action_queue', 'predict_action_chunk')
    
    # 动作生成流程
    dot.edge('predict_action_chunk', 'generate_actions')
    dot.edge('generate_actions', 'conditional_sample')
    dot.edge('conditional_sample', 'extend_action_queue')
    dot.edge('extend_action_queue', 'action_queue')
    
    # 动作选择流程
    dot.edge('action_queue', 'popleft_action')
    dot.edge('popleft_action', 'unnormalize_outputs')
    dot.edge('unnormalize_outputs', 'final_action')
    
    # 条件分支
    dot.edge('check_action_queue', 'popleft_action', style='dashed')
    
    dot.render('./images/inference_flow', view=True, cleanup=True)
    print("推理流程图已生成: ./images/inference_flow.png")

def create_training_function_calls():
    """训练函数调用关系图"""
    
    dot = graphviz.Digraph('Training_Function_Calls',
                          comment='训练函数调用关系图',
                          format='png')
    
    dot.attr(rankdir='TB', size='16,20', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 主函数
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='主函数', style='filled', color='lightblue')
        
        c.node('forward', 'DiffusionPolicy.forward\n(主训练函数)', fillcolor='lightcyan')
        c.node('compute_loss', 'DiffusionModel.compute_loss\n(计算损失)', fillcolor='lightcyan')
    
    # 数据预处理函数
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='数据预处理', style='filled', color='lightgreen')
        
        c.node('normalize_inputs', 'Normalize.__call__\n(输入归一化)', fillcolor='lightgreen')
        c.node('normalize_targets', 'Normalize.__call__\n(目标归一化)', fillcolor='lightgreen')
        c.node('stack_images', 'torch.stack\n(图像堆叠)', fillcolor='lightgreen')
    
    # 特征编码函数
    with dot.subgraph(name='cluster_encoding') as c:
        c.attr(label='特征编码', style='filled', color='lightyellow')
        
        c.node('prepare_global_cond', '_prepare_global_conditioning\n(准备全局条件)', fillcolor='lightyellow')
        c.node('rgb_encoder_forward', 'DiffusionRgbEncoder.forward\n(视觉编码)', fillcolor='lightyellow')
        c.node('backbone_forward', 'ResNet.forward\n(骨干网络)', fillcolor='lightyellow')
        c.node('spatial_softmax_forward', 'SpatialSoftmax.forward\n(空间软最大值)', fillcolor='lightyellow')
    
    # 扩散过程函数
    with dot.subgraph(name='cluster_diffusion') as c:
        c.attr(label='扩散过程', style='filled', color='lightcoral')
        
        c.node('add_noise', 'DDPMScheduler.add_noise\n(添加噪声)', fillcolor='lightcoral')
        c.node('unet_forward', 'DiffusionConditionalUnet1d.forward\n(U-Net前向)', fillcolor='lightcoral')
        c.node('residual_block_forward', 'DiffusionConditionalResidualBlock1d.forward\n(残差块)', fillcolor='lightcoral')
        c.node('film_modulation', 'FiLM调制\n(特征线性调制)', fillcolor='lightcoral')
    
    # 损失计算函数
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='损失计算', style='filled', color='lightpink')
        
        c.node('mse_loss', 'F.mse_loss\n(MSE损失)', fillcolor='lightpink')
        c.node('mask_loss', '掩码损失计算\n(可选)', fillcolor='lightpink')
    
    # 连接关系
    # 主函数调用
    dot.edge('forward', 'normalize_inputs')
    dot.edge('forward', 'stack_images')
    dot.edge('forward', 'normalize_targets')
    dot.edge('forward', 'compute_loss')
    
    # 数据预处理调用
    dot.edge('normalize_inputs', 'normalize_inputs')
    dot.edge('normalize_targets', 'normalize_targets')
    
    # 特征编码调用
    dot.edge('compute_loss', 'prepare_global_cond')
    dot.edge('prepare_global_cond', 'rgb_encoder_forward')
    dot.edge('rgb_encoder_forward', 'backbone_forward')
    dot.edge('backbone_forward', 'spatial_softmax_forward')
    
    # 扩散过程调用
    dot.edge('compute_loss', 'add_noise')
    dot.edge('compute_loss', 'unet_forward')
    dot.edge('unet_forward', 'residual_block_forward')
    dot.edge('residual_block_forward', 'film_modulation')
    
    # 损失计算调用
    dot.edge('compute_loss', 'mse_loss')
    dot.edge('mse_loss', 'mask_loss')
    
    dot.render('./images/training_function_calls', view=True, cleanup=True)
    print("训练函数调用关系图已生成: ./images/training_function_calls.png")

def create_inference_function_calls():
    """推理函数调用关系图"""
    
    dot = graphviz.Digraph('Inference_Function_Calls',
                          comment='推理函数调用关系图',
                          format='png')
    
    dot.attr(rankdir='TB', size='16,20', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 主函数
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='主函数', style='filled', color='lightblue')
        
        c.node('select_action', 'DiffusionPolicy.select_action\n(主推理函数)', fillcolor='lightcyan')
        c.node('predict_action_chunk', 'predict_action_chunk\n(预测动作块)', fillcolor='lightcyan')
        c.node('generate_actions', 'DiffusionModel.generate_actions\n(生成动作)', fillcolor='lightcyan')
    
    # 数据预处理函数
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='数据预处理', style='filled', color='lightgreen')
        
        c.node('normalize_inputs', 'Normalize.__call__\n(输入归一化)', fillcolor='lightgreen')
        c.node('stack_images', 'torch.stack\n(图像堆叠)', fillcolor='lightgreen')
        c.node('populate_queues', 'populate_queues\n(填充队列)', fillcolor='lightgreen')
    
    # 特征编码函数
    with dot.subgraph(name='cluster_encoding') as c:
        c.attr(label='特征编码', style='filled', color='lightyellow')
        
        c.node('prepare_global_cond', '_prepare_global_conditioning\n(准备全局条件)', fillcolor='lightyellow')
        c.node('rgb_encoder_forward', 'DiffusionRgbEncoder.forward\n(视觉编码)', fillcolor='lightyellow')
        c.node('backbone_forward', 'ResNet.forward\n(骨干网络)', fillcolor='lightyellow')
        c.node('spatial_softmax_forward', 'SpatialSoftmax.forward\n(空间软最大值)', fillcolor='lightyellow')
    
    # 扩散采样函数
    with dot.subgraph(name='cluster_sampling') as c:
        c.attr(label='扩散采样', style='filled', color='lightcoral')
        
        c.node('conditional_sample', 'conditional_sample\n(条件采样)', fillcolor='lightcoral')
        c.node('set_timesteps', 'set_timesteps\n(设置时间步)', fillcolor='lightcoral')
        c.node('unet_forward', 'DiffusionConditionalUnet1d.forward\n(U-Net前向)', fillcolor='lightcoral')
        c.node('scheduler_step', 'scheduler.step\n(调度器步进)', fillcolor='lightcoral')
    
    # 后处理函数
    with dot.subgraph(name='cluster_postprocessing') as c:
        c.attr(label='后处理', style='filled', color='lightpink')
        
        c.node('extend_action_queue', 'extend\n(扩展动作队列)', fillcolor='lightpink')
        c.node('popleft_action', 'popleft\n(弹出动作)', fillcolor='lightpink')
        c.node('unnormalize_outputs', 'Unnormalize.__call__\n(输出反归一化)', fillcolor='lightpink')
    
    # 连接关系
    # 主函数调用
    dot.edge('select_action', 'normalize_inputs')
    dot.edge('select_action', 'stack_images')
    dot.edge('select_action', 'populate_queues')
    dot.edge('select_action', 'predict_action_chunk')
    dot.edge('select_action', 'popleft_action')
    dot.edge('select_action', 'unnormalize_outputs')
    
    # 数据预处理调用
    dot.edge('normalize_inputs', 'normalize_inputs')
    dot.edge('populate_queues', 'populate_queues')
    
    # 特征编码调用
    dot.edge('predict_action_chunk', 'generate_actions')
    dot.edge('generate_actions', 'prepare_global_cond')
    dot.edge('prepare_global_cond', 'rgb_encoder_forward')
    dot.edge('rgb_encoder_forward', 'backbone_forward')
    dot.edge('backbone_forward', 'spatial_softmax_forward')
    
    # 扩散采样调用
    dot.edge('generate_actions', 'conditional_sample')
    dot.edge('conditional_sample', 'set_timesteps')
    dot.edge('conditional_sample', 'unet_forward')
    dot.edge('conditional_sample', 'scheduler_step')
    
    # 后处理调用
    dot.edge('predict_action_chunk', 'extend_action_queue')
    dot.edge('extend_action_queue', 'extend_action_queue')
    dot.edge('popleft_action', 'popleft_action')
    dot.edge('unnormalize_outputs', 'unnormalize_outputs')
    
    dot.render('./images/inference_function_calls', view=True, cleanup=True)
    print("推理函数调用关系图已生成: ./images/inference_function_calls.png")

if __name__ == "__main__":
    create_neural_network_structure()
    create_training_flow()
    create_inference_flow()
    create_training_function_calls()
    create_inference_function_calls() 