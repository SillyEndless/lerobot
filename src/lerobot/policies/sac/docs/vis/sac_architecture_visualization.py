#!/usr/bin/env python3
"""
SAC模型架构可视化脚本
使用graphviz生成神经网络结构图
"""

import graphviz

def create_sac_architecture_diagram():
    """创建SAC模型整体架构图"""
    dot = graphviz.Digraph(comment='SAC模型架构图')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    # 设置节点样式
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='14')
    
    # 输入层
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='输入层', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('obs_state', 'observation.state\n(机器人状态)', fillcolor='lightcyan')
        c.node('obs_image', 'observation.image\n(图像观测)', fillcolor='lightcyan')
        c.node('obs_env', 'observation.environment_state\n(环境状态)', fillcolor='lightcyan')
    
    # 编码器层
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label='编码器层', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('state_encoder', '状态编码器\n(MLP + LayerNorm + Tanh)', fillcolor='lightyellow')
        c.node('image_encoder', '图像编码器\n(ResNet/CNN)', fillcolor='lightyellow')
        c.node('env_encoder', '环境编码器\n(MLP + LayerNorm + Tanh)', fillcolor='lightyellow')
        c.node('spatial_embed', '空间嵌入\n(SpatialLearnedEmbeddings)', fillcolor='lightyellow')
        c.node('post_encoder', '后处理编码器\n(Dropout + Linear + LayerNorm + Tanh)', fillcolor='lightyellow')
    
    # 特征融合
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='特征融合', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('concat', '特征拼接\n(torch.cat)', fillcolor='lightpink')
        c.node('latent_features', '潜在特征\n(latent_dim=256)', fillcolor='lightpink')
    
    # Actor网络
    with dot.subgraph(name='cluster_actor') as c:
        c.attr(label='Actor网络 (策略网络)', style='filled', color='lightsteelblue', fontname='SimHei', fontsize='16')
        c.node('actor_mlp', 'MLP网络\n(hidden_dims=[256, 256])', fillcolor='lightblue')
        c.node('mean_layer', '均值层\n(Linear)', fillcolor='lightblue')
        c.node('std_layer', '标准差层\n(Linear)', fillcolor='lightblue')
        c.node('tanh_dist', 'Tanh变换分布\n(TanhMultivariateNormalDiag)', fillcolor='lightblue')
        c.node('actions', '动作输出\n(连续动作)', fillcolor='lightblue')
    
    # Critic网络
    with dot.subgraph(name='cluster_critic') as c:
        c.attr(label='Critic网络 (价值网络)', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('critic_ensemble', 'Critic集成\n(num_critics=2)', fillcolor='lightyellow')
        c.node('critic_heads', 'Critic头\n(CriticHead)', fillcolor='lightyellow')
        c.node('q_values', 'Q值输出\n(状态-动作价值)', fillcolor='lightyellow')
        c.node('target_critic', '目标Critic\n(EMA更新)', fillcolor='lightyellow')
    
    # 温度参数
    with dot.subgraph(name='cluster_temperature') as c:
        c.attr(label='温度控制', style='filled', color='lightseagreen', fontname='SimHei', fontsize='16')
        c.node('log_alpha', 'log_alpha参数\n(可学习)', fillcolor='lightgreen')
        c.node('temperature', '温度参数\n(exp(log_alpha))', fillcolor='lightgreen')
    
    # 离散动作处理
    with dot.subgraph(name='cluster_discrete') as c:
        c.attr(label='离散动作处理', style='filled', color='lightpink', fontname='SimHei', fontsize='16')
        c.node('discrete_critic', '离散Critic\n(DiscreteCritic)', fillcolor='lightcoral')
        c.node('discrete_actions', '离散动作\n(如夹爪控制)', fillcolor='lightcoral')
    
    # 连接关系
    # 输入到编码器
    dot.edge('obs_state', 'state_encoder')
    dot.edge('obs_image', 'image_encoder')
    dot.edge('obs_env', 'env_encoder')
    
    # 图像编码器处理
    dot.edge('image_encoder', 'spatial_embed')
    dot.edge('spatial_embed', 'post_encoder')
    
    # 特征融合
    dot.edge('state_encoder', 'concat')
    dot.edge('post_encoder', 'concat')
    dot.edge('env_encoder', 'concat')
    dot.edge('concat', 'latent_features')
    
    # Actor网络连接
    dot.edge('latent_features', 'actor_mlp')
    dot.edge('actor_mlp', 'mean_layer')
    dot.edge('actor_mlp', 'std_layer')
    dot.edge('mean_layer', 'tanh_dist')
    dot.edge('std_layer', 'tanh_dist')
    dot.edge('tanh_dist', 'actions')
    
    # Critic网络连接
    dot.edge('latent_features', 'critic_ensemble')
    dot.edge('actions', 'critic_ensemble')
    dot.edge('critic_ensemble', 'critic_heads')
    dot.edge('critic_heads', 'q_values')
    dot.edge('critic_ensemble', 'target_critic')
    
    # 温度参数连接
    dot.edge('log_alpha', 'temperature')
    dot.edge('temperature', 'tanh_dist')
    
    # 离散动作连接
    dot.edge('latent_features', 'discrete_critic')
    dot.edge('discrete_critic', 'discrete_actions')
    dot.edge('discrete_actions', 'actions')
    
    return dot

def create_sac_training_flow():
    """创建SAC训练流程图"""
    dot = graphviz.Digraph(comment='SAC训练流程图')
    dot.attr(rankdir='TB', size='14,18', dpi='300')
    
    # 设置节点样式
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 数据收集阶段
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='数据收集阶段', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('env_interaction', '环境交互\n(收集经验)', fillcolor='lightcyan')
        c.node('replay_buffer', '经验回放缓冲区\n(ReplayBuffer)', fillcolor='lightcyan')
        c.node('batch_sampling', '批次采样\n(随机采样)', fillcolor='lightcyan')
    
    # Critic训练阶段
    with dot.subgraph(name='cluster_critic_training') as c:
        c.attr(label='Critic训练阶段', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('next_action', '计算下一动作\n(actor.forward)', fillcolor='lightyellow')
        c.node('target_q', '计算目标Q值\n(critic_target.forward)', fillcolor='lightyellow')
        c.node('td_target', 'TD目标计算\n(r + γ * min(Q_target))', fillcolor='lightyellow')
        c.node('critic_loss', 'Critic损失\n(MSE Loss)', fillcolor='lightyellow')
        c.node('critic_update', 'Critic参数更新\n(Adam优化器)', fillcolor='lightyellow')
    
    # Actor训练阶段
    with dot.subgraph(name='cluster_actor_training') as c:
        c.attr(label='Actor训练阶段', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('actor_action', 'Actor动作采样\n(策略网络)', fillcolor='lightpink')
        c.node('actor_q', 'Actor Q值\n(critic.forward)', fillcolor='lightpink')
        c.node('actor_loss', 'Actor损失\n(α*log_prob - Q)', fillcolor='lightpink')
        c.node('actor_update', 'Actor参数更新\n(Adam优化器)', fillcolor='lightpink')
    
    # 温度训练阶段
    with dot.subgraph(name='cluster_temperature_training') as c:
        c.attr(label='温度训练阶段', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('temp_loss', '温度损失\n(-α*(log_prob + H_target))', fillcolor='lightyellow')
        c.node('temp_update', '温度参数更新\n(Adam优化器)', fillcolor='lightyellow')
    
    # 目标网络更新
    with dot.subgraph(name='cluster_target_update') as c:
        c.attr(label='目标网络更新', style='filled', color='lightsteelblue', fontname='SimHei', fontsize='16')
        c.node('target_update', '目标网络更新\n(EMA: τ=0.005)', fillcolor='lightblue')
    
    # 连接关系
    # 数据流
    dot.edge('env_interaction', 'replay_buffer')
    dot.edge('replay_buffer', 'batch_sampling')
    
    # Critic训练流
    dot.edge('batch_sampling', 'next_action')
    dot.edge('next_action', 'target_q')
    dot.edge('target_q', 'td_target')
    dot.edge('td_target', 'critic_loss')
    dot.edge('critic_loss', 'critic_update')
    
    # Actor训练流
    dot.edge('batch_sampling', 'actor_action')
    dot.edge('actor_action', 'actor_q')
    dot.edge('actor_q', 'actor_loss')
    dot.edge('actor_loss', 'actor_update')
    
    # 温度训练流
    dot.edge('actor_action', 'temp_loss')
    dot.edge('temp_loss', 'temp_update')
    
    # 目标网络更新
    dot.edge('critic_update', 'target_update')
    
    return dot

def create_sac_inference_flow():
    """创建SAC推理流程图"""
    dot = graphviz.Digraph(comment='SAC推理流程图')
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    
    # 设置节点样式
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入处理
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入处理', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('obs_input', '观测输入\n(状态/图像)', fillcolor='lightcyan')
        c.node('normalize', '输入标准化\n(NormalizeBuffer)', fillcolor='lightcyan')
    
    # 特征编码
    with dot.subgraph(name='cluster_encoding') as c:
        c.attr(label='特征编码', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('encoder', '观测编码器\n(SACObservationEncoder)', fillcolor='lightyellow')
        c.node('cached_features', '缓存特征\n(避免重复计算)', fillcolor='lightyellow')
    
    # 动作生成
    with dot.subgraph(name='cluster_action') as c:
        c.attr(label='动作生成', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('actor_forward', 'Actor前向传播\n(select_action)', fillcolor='lightpink')
        c.node('action_sampling', '动作采样\n(TanhMultivariateNormalDiag)', fillcolor='lightpink')
        c.node('continuous_action', '连续动作输出', fillcolor='lightpink')
    
    # 离散动作处理
    with dot.subgraph(name='cluster_discrete') as c:
        c.attr(label='离散动作处理', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('discrete_critic', '离散Critic\n(DiscreteCritic)', fillcolor='lightyellow')
        c.node('discrete_action', '离散动作\n(argmax)', fillcolor='lightyellow')
    
    # 动作组合
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出组合', style='filled', color='lightsteelblue', fontname='SimHei', fontsize='16')
        c.node('action_concat', '动作拼接\n(torch.cat)', fillcolor='lightblue')
        c.node('final_action', '最终动作输出', fillcolor='lightblue')
    
    # 连接关系
    dot.edge('obs_input', 'normalize')
    dot.edge('normalize', 'encoder')
    dot.edge('encoder', 'cached_features')
    dot.edge('cached_features', 'actor_forward')
    dot.edge('actor_forward', 'action_sampling')
    dot.edge('action_sampling', 'continuous_action')
    
    # 离散动作分支
    dot.edge('cached_features', 'discrete_critic')
    dot.edge('discrete_critic', 'discrete_action')
    
    # 动作组合
    dot.edge('continuous_action', 'action_concat')
    dot.edge('discrete_action', 'action_concat')
    dot.edge('action_concat', 'final_action')
    
    return dot

def create_function_call_relationships():
    """创建函数调用关系图"""
    dot = graphviz.Digraph(comment='SAC函数调用关系图')
    dot.attr(rankdir='TB', size='14,20', dpi='300')
    
    # 设置节点样式
    dot.attr('node', shape='box', style='filled', fontname='SimHei', fontsize='10')
    
    # 训练相关函数
    with dot.subgraph(name='cluster_training') as c:
        c.attr(label='训练函数调用关系', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        c.node('forward', 'forward(model="critic/actor/temperature")', fillcolor='lightcyan')
        c.node('compute_loss_critic', 'compute_loss_critic()', fillcolor='lightcyan')
        c.node('compute_loss_actor', 'compute_loss_actor()', fillcolor='lightcyan')
        c.node('compute_loss_temperature', 'compute_loss_temperature()', fillcolor='lightcyan')
        c.node('critic_forward', 'critic_forward()', fillcolor='lightcyan')
        c.node('update_target_networks', 'update_target_networks()', fillcolor='lightcyan')
        c.node('update_temperature', 'update_temperature()', fillcolor='lightcyan')
    
    # 推理相关函数
    with dot.subgraph(name='cluster_inference') as c:
        c.attr(label='推理函数调用关系', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        c.node('select_action', 'select_action()', fillcolor='lightyellow')
        c.node('predict_action_chunk', 'predict_action_chunk()', fillcolor='lightyellow')
        c.node('reset', 'reset()', fillcolor='lightyellow')
    
    # 网络组件函数
    with dot.subgraph(name='cluster_networks') as c:
        c.attr(label='网络组件函数', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        c.node('actor_forward', 'Policy.forward()', fillcolor='lightpink')
        c.node('critic_ensemble_forward', 'CriticEnsemble.forward()', fillcolor='lightpink')
        c.node('encoder_forward', 'SACObservationEncoder.forward()', fillcolor='lightpink')
        c.node('get_cached_features', 'get_cached_image_features()', fillcolor='lightpink')
    
    # 初始化函数
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='初始化函数', style='filled', color='lightgoldenrod', fontname='SimHei', fontsize='16')
        c.node('__init__', '__init__()', fillcolor='lightyellow')
        c.node('_init_normalization', '_init_normalization()', fillcolor='lightyellow')
        c.node('_init_encoders', '_init_encoders()', fillcolor='lightyellow')
        c.node('_init_critics', '_init_critics()', fillcolor='lightyellow')
        c.node('_init_actor', '_init_actor()', fillcolor='lightyellow')
        c.node('_init_temperature', '_init_temperature()', fillcolor='lightyellow')
    
    # 连接关系
    # 训练调用关系
    dot.edge('forward', 'compute_loss_critic')
    dot.edge('forward', 'compute_loss_actor')
    dot.edge('forward', 'compute_loss_temperature')
    dot.edge('compute_loss_critic', 'critic_forward')
    dot.edge('compute_loss_actor', 'critic_forward')
    dot.edge('compute_loss_actor', 'actor_forward')
    dot.edge('compute_loss_temperature', 'actor_forward')
    
    # 推理调用关系
    dot.edge('select_action', 'actor_forward')
    dot.edge('select_action', 'get_cached_features')
    dot.edge('select_action', 'critic_ensemble_forward')
    
    # 网络组件调用关系
    dot.edge('actor_forward', 'encoder_forward')
    dot.edge('critic_ensemble_forward', 'encoder_forward')
    dot.edge('encoder_forward', 'get_cached_features')
    
    # 初始化调用关系
    dot.edge('__init__', '_init_normalization')
    dot.edge('__init__', '_init_encoders')
    dot.edge('__init__', '_init_critics')
    dot.edge('__init__', '_init_actor')
    dot.edge('__init__', '_init_temperature')
    
    return dot

def main():
    """主函数：生成所有可视化图表"""
    print("正在生成SAC模型可视化图表...")
    
    # 生成架构图
    print("1. 生成SAC模型架构图...")
    arch_dot = create_sac_architecture_diagram()
    arch_dot.render('./images/sac_architecture', format='png', cleanup=True)
    
    # 生成训练流程图
    print("2. 生成SAC训练流程图...")
    train_dot = create_sac_training_flow()
    train_dot.render('./images/sac_training_flow', format='png', cleanup=True)
    
    # 生成推理流程图
    print("3. 生成SAC推理流程图...")
    infer_dot = create_sac_inference_flow()
    infer_dot.render('./images/sac_inference_flow', format='png', cleanup=True)
    
    # 生成函数调用关系图
    print("4. 生成SAC函数调用关系图...")
    func_dot = create_function_call_relationships()
    func_dot.render('./images/sac_function_calls', format='png', cleanup=True)
    
    print("所有图表生成完成！")
    print("生成的图片文件：")
    print("- ./images/sac_architecture.png")
    print("- ./images/sac_training_flow.png")
    print("- ./images/sac_inference_flow.png")
    print("- ./images/sac_function_calls.png")

if __name__ == "__main__":
    main() 