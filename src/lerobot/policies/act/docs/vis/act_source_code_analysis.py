#!/usr/bin/env python3
"""
ACT源码分析可视化脚本
基于源码生成神经网络结构图、流程图和函数调用关系图
"""

import graphviz

def create_neural_network_structure():
    """创建基于源码的神经网络结构图"""
    dot = graphviz.Digraph(
        'ACT_Neural_Network',
        comment='ACT神经网络结构图（基于源码）',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='14,16', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='10')
    
    # 定义颜色方案
    colors = {
        'input': '#E8F4FD',
        'backbone': '#FFF2CC', 
        'encoder': '#E1D5E7',
        'decoder': '#D5E8D4',
        'vae': '#F8CECC',
        'attention': '#DAE8FC',
        'output': '#FFE6CC'
    }
    
    # 输入层
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='输入层', style='filled', color='lightblue', fontname='SimHei', fontsize='12')
        
        c.node('obs_images', 'observation.images\n(B, C, H, W)', fillcolor=colors['input'], color='blue')
        c.node('obs_state', 'observation.state\n(B, state_dim)', fillcolor=colors['input'], color='blue')
        c.node('obs_env', 'observation.environment_state\n(B, env_dim)', fillcolor=colors['input'], color='blue')
        c.node('action_target', 'action\n(B, chunk_size, action_dim)', fillcolor=colors['input'], color='blue')
    
    # VAE编码器
    with dot.subgraph(name='cluster_vae') as c:
        c.attr(label='VAE编码器 (ACTEncoder)', style='filled', color='lightcoral', fontname='SimHei', fontsize='12')
        
        c.node('vae_cls_embed', 'vae_encoder_cls_embed\n(Embedding)', fillcolor=colors['vae'], color='red')
        c.node('vae_state_proj', 'vae_encoder_robot_state_input_proj\n(Linear)', fillcolor=colors['vae'], color='red')
        c.node('vae_action_proj', 'vae_encoder_action_input_proj\n(Linear)', fillcolor=colors['vae'], color='red')
        c.node('vae_encoder', 'ACTEncoder\n(n_vae_encoder_layers=4)', fillcolor=colors['vae'], color='red')
        c.node('vae_latent_proj', 'vae_encoder_latent_output_proj\n(Linear)', fillcolor=colors['vae'], color='red')
        c.node('latent_params', 'latent_pdf_params\n(μ, log(σ²))', fillcolor=colors['vae'], color='red')
        c.node('latent_sample', 'latent_sample\n(重参数化)', fillcolor=colors['vae'], color='red')
    
    # 视觉骨干网络
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='视觉骨干网络', style='filled', color='lightyellow', fontname='SimHei', fontsize='12')
        
        c.node('backbone', 'backbone\n(ResNet18)', fillcolor=colors['backbone'], color='orange')
        c.node('img_proj', 'encoder_img_feat_input_proj\n(Conv2d)', fillcolor=colors['backbone'], color='orange')
        c.node('pos_embed_2d', 'encoder_cam_feat_pos_embed\n(ACTSinusoidalPositionEmbedding2d)', fillcolor=colors['attention'], color='navy')
    
    # Transformer编码器
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Transformer编码器', style='filled', color='lightyellow', fontname='SimHei', fontsize='12')
        
        c.node('latent_proj', 'encoder_latent_input_proj\n(Linear)', fillcolor=colors['encoder'], color='purple')
        c.node('state_proj', 'encoder_robot_state_input_proj\n(Linear)', fillcolor=colors['encoder'], color='purple')
        c.node('env_proj', 'encoder_env_state_input_proj\n(Linear)', fillcolor=colors['encoder'], color='purple')
        c.node('pos_embed_1d', 'encoder_1d_feature_pos_embed\n(Embedding)', fillcolor=colors['attention'], color='navy')
        c.node('transformer_encoder', 'ACTEncoder\n(n_encoder_layers=4)', fillcolor=colors['encoder'], color='purple')
    
    # Transformer解码器
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Transformer解码器', style='filled', color='lightpink', fontname='SimHei', fontsize='12')
        
        c.node('decoder_pos_embed', 'decoder_pos_embed\n(Embedding)', fillcolor=colors['attention'], color='navy')
        c.node('transformer_decoder', 'ACTDecoder\n(n_decoder_layers=1)', fillcolor=colors['decoder'], color='green')
        c.node('action_head', 'action_head\n(Linear)', fillcolor=colors['output'], color='brown')
    
    # 输出层
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层', style='filled', color='lightgreen', fontname='SimHei', fontsize='12')
        
        c.node('predicted_actions', 'predicted_actions\n(B, chunk_size, action_dim)', fillcolor=colors['output'], color='brown')
    
    # 连接VAE
    dot.edge('obs_state', 'vae_state_proj')
    dot.edge('action_target', 'vae_action_proj')
    dot.edge('vae_cls_embed', 'vae_encoder')
    dot.edge('vae_state_proj', 'vae_encoder')
    dot.edge('vae_action_proj', 'vae_encoder')
    dot.edge('vae_encoder', 'vae_latent_proj')
    dot.edge('vae_latent_proj', 'latent_params')
    dot.edge('latent_params', 'latent_sample')
    
    # 连接视觉
    dot.edge('obs_images', 'backbone')
    dot.edge('backbone', 'img_proj')
    dot.edge('img_proj', 'pos_embed_2d')
    
    # 连接编码器
    dot.edge('latent_sample', 'latent_proj')
    dot.edge('obs_state', 'state_proj')
    dot.edge('obs_env', 'env_proj')
    dot.edge('latent_proj', 'transformer_encoder')
    dot.edge('state_proj', 'transformer_encoder')
    dot.edge('env_proj', 'transformer_encoder')
    dot.edge('pos_embed_1d', 'transformer_encoder')
    dot.edge('pos_embed_2d', 'transformer_encoder')
    
    # 连接解码器
    dot.edge('transformer_encoder', 'transformer_decoder')
    dot.edge('decoder_pos_embed', 'transformer_decoder')
    dot.edge('transformer_decoder', 'action_head')
    dot.edge('action_head', 'predicted_actions')
    
    return dot

def create_training_flow():
    """创建训练流程图"""
    dot = graphviz.Digraph(
        'ACT_Training_Flow',
        comment='ACT训练流程图（基于源码）',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,14', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='11')
    
    # 数据准备
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='数据准备', style='filled', color='lightblue', fontname='SimHei', fontsize='12')
        
        c.node('batch_data', '批次数据\nbatch', fillcolor='#E8F4FD', color='blue')
        c.node('normalize_inputs', 'normalize_inputs\n(输入归一化)', fillcolor='#E8F4FD', color='blue')
        c.node('normalize_targets', 'normalize_targets\n(目标归一化)', fillcolor='#E8F4FD', color='blue')
    
    # VAE编码
    with dot.subgraph(name='cluster_vae_encoding') as c:
        c.attr(label='VAE编码', style='filled', color='lightcoral', fontname='SimHei', fontsize='12')
        
        c.node('vae_encoder_input', '准备VAE编码器输入\n[cls, state, action]', fillcolor='#F8CECC', color='red')
        c.node('vae_encoder_forward', 'vae_encoder.forward()\n(ACTEncoder)', fillcolor='#F8CECC', color='red')
        c.node('latent_dist', '获取潜在分布\nμ, log(σ²)', fillcolor='#F8CECC', color='red')
        c.node('reparam_sample', '重参数化采样\nlatent_sample', fillcolor='#F8CECC', color='red')
    
    # 特征提取
    with dot.subgraph(name='cluster_features') as c:
        c.attr(label='特征提取', style='filled', color='lightyellow', fontname='SimHei', fontsize='12')
        
        c.node('backbone_forward', 'backbone.forward()\n(ResNet18)', fillcolor='#FFF2CC', color='orange')
        c.node('img_projection', '图像特征投影\n(Conv2d)', fillcolor='#FFF2CC', color='orange')
        c.node('state_projection', '状态特征投影\n(Linear)', fillcolor='#FFF2CC', color='orange')
        c.node('pos_encoding', '位置编码\n(1D/2D)', fillcolor='#FFF2CC', color='orange')
    
    # Transformer处理
    with dot.subgraph(name='cluster_transformer') as c:
        c.attr(label='Transformer处理', style='filled', color='lightpink', fontname='SimHei', fontsize='12')
        
        c.node('encoder_forward', 'encoder.forward()\n(ACTEncoder)', fillcolor='#E1D5E7', color='purple')
        c.node('decoder_forward', 'decoder.forward()\n(ACTDecoder)', fillcolor='#E1D5E7', color='purple')
        c.node('action_prediction', 'action_head\n(动作预测)', fillcolor='#E1D5E7', color='purple')
    
    # 损失计算
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='损失计算', style='filled', color='lightgreen', fontname='SimHei', fontsize='12')
        
        c.node('l1_loss', 'L1重构损失\nF.l1_loss()', fillcolor='#D5E8D4', color='green')
        c.node('kl_loss', 'KL散度损失\nKL(q||p)', fillcolor='#D5E8D4', color='green')
        c.node('total_loss', '总损失\nL = L_recon + β·L_KL', fillcolor='#D5E8D4', color='green')
    
    # 连接
    dot.edge('batch_data', 'normalize_inputs')
    dot.edge('batch_data', 'normalize_targets')
    
    dot.edge('normalize_inputs', 'vae_encoder_input')
    dot.edge('normalize_targets', 'vae_encoder_input')
    dot.edge('vae_encoder_input', 'vae_encoder_forward')
    dot.edge('vae_encoder_forward', 'latent_dist')
    dot.edge('latent_dist', 'reparam_sample')
    
    dot.edge('normalize_inputs', 'backbone_forward')
    dot.edge('backbone_forward', 'img_projection')
    dot.edge('img_projection', 'pos_encoding')
    dot.edge('normalize_inputs', 'state_projection')
    
    dot.edge('reparam_sample', 'encoder_forward')
    dot.edge('pos_encoding', 'encoder_forward')
    dot.edge('state_projection', 'encoder_forward')
    dot.edge('encoder_forward', 'decoder_forward')
    dot.edge('decoder_forward', 'action_prediction')
    
    dot.edge('action_prediction', 'l1_loss')
    dot.edge('normalize_targets', 'l1_loss')
    dot.edge('latent_dist', 'kl_loss')
    dot.edge('l1_loss', 'total_loss')
    dot.edge('kl_loss', 'total_loss')
    
    return dot

def create_inference_flow():
    """创建推理流程图"""
    dot = graphviz.Digraph(
        'ACT_Inference_Flow',
        comment='ACT推理流程图（基于源码）',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,14', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='11')
    
    # 输入处理
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入处理', style='filled', color='lightblue', fontname='SimHei', fontsize='12')
        
        c.node('obs_input', '环境观察\nbatch', fillcolor='#E8F4FD', color='blue')
        c.node('normalize_input', 'normalize_inputs\n(输入归一化)', fillcolor='#E8F4FD', color='blue')
        c.node('check_queue', '检查动作队列\n_action_queue', fillcolor='#E8F4FD', color='blue')
    
    # 动作预测
    with dot.subgraph(name='cluster_prediction') as c:
        c.attr(label='动作预测', style='filled', color='lightyellow', fontname='SimHei', fontsize='12')
        
        c.node('predict_chunk', 'predict_action_chunk()\n(预测动作块)', fillcolor='#FFF2CC', color='orange')
        c.node('model_forward', 'model.forward()\n(ACT模型前向传播)', fillcolor='#FFF2CC', color='orange')
        c.node('unnormalize', 'unnormalize_outputs\n(输出反归一化)', fillcolor='#FFF2CC', color='orange')
    
    # 时间集成
    with dot.subgraph(name='cluster_ensemble') as c:
        c.attr(label='时间集成（可选）', style='filled', color='lightpink', fontname='SimHei', fontsize='12')
        
        c.node('temporal_ensemble', 'temporal_ensembler.update()\n(时间集成)', fillcolor='#E1D5E7', color='purple')
        c.node('ensemble_weights', '指数权重计算\nwᵢ = exp(-α·i)', fillcolor='#E1D5E7', color='purple')
    
    # 动作执行
    with dot.subgraph(name='cluster_execution') as c:
        c.attr(label='动作执行', style='filled', color='lightgreen', fontname='SimHei', fontsize='12')
        
        c.node('action_queue', '动作队列管理\n_action_queue', fillcolor='#D5E8D4', color='green')
        c.node('select_action', 'select_action()\n(选择单个动作)', fillcolor='#D5E8D4', color='green')
        c.node('return_action', '返回动作\n(执行到环境)', fillcolor='#D5E8D4', color='green')
    
    # 连接
    dot.edge('obs_input', 'normalize_input')
    dot.edge('normalize_input', 'check_queue')
    
    # 队列为空时的路径
    dot.edge('check_queue', 'predict_chunk', label='队列为空')
    dot.edge('predict_chunk', 'model_forward')
    dot.edge('model_forward', 'unnormalize')
    
    # 时间集成路径
    dot.edge('unnormalize', 'temporal_ensemble', label='启用时间集成')
    dot.edge('ensemble_weights', 'temporal_ensemble')
    dot.edge('temporal_ensemble', 'return_action')
    
    # 队列管理路径
    dot.edge('unnormalize', 'action_queue', label='n_action_steps > 1')
    dot.edge('action_queue', 'select_action')
    dot.edge('select_action', 'return_action')
    
    # 直接返回路径
    dot.edge('unnormalize', 'return_action', label='n_action_steps = 1')
    
    return dot

def create_function_call_relations():
    """创建函数调用关系图"""
    dot = graphviz.Digraph(
        'ACT_Function_Calls',
        comment='ACT函数调用关系图（基于源码）',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='14,16', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='10')
    
    # 主策略类
    with dot.subgraph(name='cluster_policy') as c:
        c.attr(label='ACTPolicy类', style='filled', color='lightblue', fontname='SimHei', fontsize='12')
        
        c.node('__init__', '__init__()\n(初始化)', fillcolor='#E8F4FD', color='blue')
        c.node('select_action', 'select_action()\n(选择动作)', fillcolor='#E8F4FD', color='blue')
        c.node('predict_chunk', 'predict_action_chunk()\n(预测动作块)', fillcolor='#E8F4FD', color='blue')
        c.node('forward', 'forward()\n(训练前向传播)', fillcolor='#E8F4FD', color='blue')
        c.node('reset', 'reset()\n(重置状态)', fillcolor='#E8F4FD', color='blue')
    
    # ACT模型类
    with dot.subgraph(name='cluster_model') as c:
        c.attr(label='ACT模型类', style='filled', color='lightyellow', fontname='SimHei', fontsize='12')
        
        c.node('act_forward', 'ACT.forward()\n(模型前向传播)', fillcolor='#FFF2CC', color='orange')
        c.node('vae_encoder', 'VAE编码器\n(ACTEncoder)', fillcolor='#FFF2CC', color='orange')
        c.node('backbone', '视觉骨干网络\n(ResNet18)', fillcolor='#FFF2CC', color='orange')
        c.node('transformer_encoder', 'Transformer编码器\n(ACTEncoder)', fillcolor='#FFF2CC', color='orange')
        c.node('transformer_decoder', 'Transformer解码器\n(ACTDecoder)', fillcolor='#FFF2CC', color='orange')
    
    # 编码器层
    with dot.subgraph(name='cluster_encoder_layers') as c:
        c.attr(label='编码器层', style='filled', color='lightpink', fontname='SimHei', fontsize='12')
        
        c.node('encoder_layer', 'ACTEncoderLayer.forward()\n(编码器层)', fillcolor='#E1D5E7', color='purple')
        c.node('self_attention', '多头自注意力\n(MultiheadAttention)', fillcolor='#E1D5E7', color='purple')
        c.node('feedforward', '前馈网络\n(Linear + Activation)', fillcolor='#E1D5E7', color='purple')
    
    # 解码器层
    with dot.subgraph(name='cluster_decoder_layers') as c:
        c.attr(label='解码器层', style='filled', color='lightgreen', fontname='SimHei', fontsize='12')
        
        c.node('decoder_layer', 'ACTDecoderLayer.forward()\n(解码器层)', fillcolor='#D5E8D4', color='green')
        c.node('self_attn_decoder', '自注意力\n(MultiheadAttention)', fillcolor='#D5E8D4', color='green')
        c.node('cross_attention', '交叉注意力\n(MultiheadAttention)', fillcolor='#D5E8D4', color='green')
        c.node('feedforward_decoder', '前馈网络\n(Linear + Activation)', fillcolor='#D5E8D4', color='green')
    
    # 时间集成器
    with dot.subgraph(name='cluster_ensemble') as c:
        c.attr(label='时间集成器', style='filled', color='lightcoral', fontname='SimHei', fontsize='12')
        
        c.node('ensemble_update', 'ACTTemporalEnsembler.update()\n(时间集成更新)', fillcolor='#F8CECC', color='red')
        c.node('ensemble_reset', 'ACTTemporalEnsembler.reset()\n(重置集成器)', fillcolor='#F8CECC', color='red')
    
    # 工具函数
    with dot.subgraph(name='cluster_utils') as c:
        c.attr(label='工具函数', style='filled', color='lightgray', fontname='SimHei', fontsize='12')
        
        c.node('pos_embedding', 'create_sinusoidal_pos_embedding()\n(1D位置编码)', fillcolor='#F0F0F0', color='gray')
        c.node('pos_embedding_2d', 'ACTSinusoidalPositionEmbedding2d.forward()\n(2D位置编码)', fillcolor='#F0F0F0', color='gray')
        c.node('activation_fn', 'get_activation_fn()\n(激活函数)', fillcolor='#F0F0F0', color='gray')
    
    # 主要调用关系
    dot.edge('select_action', 'predict_chunk')
    dot.edge('predict_chunk', 'act_forward')
    dot.edge('forward', 'act_forward')
    
    # ACT模型内部调用
    dot.edge('act_forward', 'vae_encoder')
    dot.edge('act_forward', 'backbone')
    dot.edge('act_forward', 'transformer_encoder')
    dot.edge('act_forward', 'transformer_decoder')
    
    # 编码器调用
    dot.edge('transformer_encoder', 'encoder_layer')
    dot.edge('encoder_layer', 'self_attention')
    dot.edge('encoder_layer', 'feedforward')
    
    # 解码器调用
    dot.edge('transformer_decoder', 'decoder_layer')
    dot.edge('decoder_layer', 'self_attn_decoder')
    dot.edge('decoder_layer', 'cross_attention')
    dot.edge('decoder_layer', 'feedforward_decoder')
    
    # 时间集成调用
    dot.edge('select_action', 'ensemble_update')
    dot.edge('reset', 'ensemble_reset')
    
    # 工具函数调用
    dot.edge('act_forward', 'pos_embedding')
    dot.edge('act_forward', 'pos_embedding_2d')
    dot.edge('encoder_layer', 'activation_fn')
    dot.edge('decoder_layer', 'activation_fn')
    
    return dot

def main():
    """主函数"""
    print("正在生成ACT源码分析图...")
    
    # 生成各个分析图
    diagrams = [
        ('neural_network', create_neural_network_structure()),
        ('training_flow', create_training_flow()),
        ('inference_flow', create_inference_flow()),
        ('function_calls', create_function_call_relations())
    ]
    
    for name, dot in diagrams:
        output_path = f'./images/act_{name}'
        dot.render(output_path, cleanup=True)
        print(f"已生成: {output_path}.png")
    
    print("\n所有源码分析图已生成完成！")
    print("包含以下分析图：")
    print("1. 神经网络结构图（基于源码）")
    print("2. 训练流程图")
    print("3. 推理流程图")
    print("4. 函数调用关系图")

if __name__ == "__main__":
    main() 