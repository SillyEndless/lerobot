#!/usr/bin/env python3
"""
Diffusion Policy 架构知识点可视化脚本
"""

import graphviz

def create_multimodal_fusion():
    """多模态观察融合架构图"""
    
    dot = graphviz.Digraph('Multimodal_Fusion',
                          comment='多模态观察融合架构',
                          format='png')
    
    dot.attr(rankdir='LR', size='14,10', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入模态
    with dot.subgraph(name='cluster_modalities') as c:
        c.attr(label='多模态输入', style='filled', color='lightblue')
        
        c.node('robot_state', '机器人状态\n(关节角度、位置等)\n(B, n_obs_steps, state_dim)',
               fillcolor='lightcyan')
        
        c.node('visual_obs', '视觉观察\n(相机图像)\n(B, n_obs_steps, num_cameras, C, H, W)',
               fillcolor='lightcyan')
        
        c.node('env_state', '环境状态\n(任务相关信息)\n(B, n_obs_steps, env_dim)',
               fillcolor='lightcyan')
    
    # 编码器
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label='特征编码', style='filled', color='lightgreen')
        
        c.node('state_enc', '状态编码器\n(直接使用)', fillcolor='lightgreen')
        c.node('visual_enc', '视觉编码器\n(ResNet + SpatialSoftmax)', fillcolor='lightgreen')
        c.node('env_enc', '环境编码器\n(直接使用)', fillcolor='lightgreen')
    
    # 融合
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='特征融合', style='filled', color='lightyellow')
        
        c.node('concat', '特征拼接\n(torch.cat)', fillcolor='lightyellow')
        c.node('flatten', '维度展平\n(flatten)', fillcolor='lightyellow')
        c.node('global_cond', '全局条件\n(B, global_cond_dim)', fillcolor='lightyellow')
    
    # 连接
    dot.edge('robot_state', 'state_enc')
    dot.edge('visual_obs', 'visual_enc')
    dot.edge('env_state', 'env_enc')
    
    dot.edge('state_enc', 'concat')
    dot.edge('visual_enc', 'concat')
    dot.edge('env_enc', 'concat')
    
    dot.edge('concat', 'flatten')
    dot.edge('flatten', 'global_cond')
    
    dot.render('./images/multimodal_fusion', view=True, cleanup=True)
    print("多模态融合架构图已生成: ./images/multimodal_fusion.png")

def create_temporal_action_generation():
    """时序动作生成架构图"""
    
    dot = graphviz.Digraph('Temporal_Action_Generation',
                          comment='时序动作生成架构',
                          format='png')
    
    dot.attr(rankdir='TB', size='12,14', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 时间轴
    with dot.subgraph(name='cluster_timeline') as c:
        c.attr(label='时间轴 (Timeline)', style='filled', color='lightgray')
        
        c.node('t_n_obs', 't-n_obs+1', fillcolor='lightblue')
        c.node('t_n_obs_1', 't-n_obs+2', fillcolor='lightblue')
        c.node('t_current', 't (当前)', fillcolor='lightgreen')
        c.node('t_future', 't+1', fillcolor='lightcoral')
        c.node('t_horizon', 't+horizon', fillcolor='lightcoral')
    
    # 观察和动作
    with dot.subgraph(name='cluster_obs_actions') as c:
        c.attr(label='观察与动作', style='filled', color='lightyellow')
        
        c.node('obs_history', '历史观察\n(observation history)\nn_obs_steps个时间步',
               fillcolor='lightyellow')
        
        c.node('action_sequence', '动作序列\n(action sequence)\nhorizon个时间步',
               fillcolor='lightyellow')
        
        c.node('action_chunk', '执行动作\n(action chunk)\nn_action_steps个时间步',
               fillcolor='lightyellow')
    
    # 队列管理
    with dot.subgraph(name='cluster_queues') as c:
        c.attr(label='队列管理', style='filled', color='lightpink')
        
        c.node('obs_queue', '观察队列\n(Observation Queue)\n维护历史观察',
               fillcolor='lightpink')
        
        c.node('action_queue', '动作队列\n(Action Queue)\n缓存生成动作',
               fillcolor='lightpink')
    
    # 连接关系
    dot.edge('obs_history', 'action_sequence')
    dot.edge('action_sequence', 'action_chunk')
    
    dot.edge('obs_queue', 'obs_history', style='dashed')
    dot.edge('action_sequence', 'action_queue', style='dashed')
    dot.edge('action_queue', 'action_chunk', style='dashed')
    
    # 时间轴连接
    dot.edge('t_n_obs', 'obs_history', style='dotted')
    dot.edge('t_current', 'action_chunk', style='dotted')
    dot.edge('t_horizon', 'action_sequence', style='dotted')
    
    dot.render('./images/temporal_action_generation', view=True, cleanup=True)
    print("时序动作生成架构图已生成: ./images/temporal_action_generation.png")

def create_conditional_diffusion():
    """条件扩散建模架构图"""
    
    dot = graphviz.Digraph('Conditional_Diffusion',
                          comment='条件扩散建模架构',
                          format='png')
    
    dot.attr(rankdir='TB', size='12,16', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 条件输入
    with dot.subgraph(name='cluster_conditioning') as c:
        c.attr(label='条件输入', style='filled', color='lightblue')
        
        c.node('global_cond', '全局条件\n(Global Conditioning)\n观察特征',
               fillcolor='lightcyan')
        
        c.node('timestep', '时间步\n(Timestep)\n扩散步数',
               fillcolor='lightcyan')
    
    # 扩散过程
    with dot.subgraph(name='cluster_diffusion') as c:
        c.attr(label='扩散过程', style='filled', color='lightgreen')
        
        c.node('noise_scheduler', '噪声调度器\n(Noise Scheduler)\nDDPM/DDIM',
               fillcolor='lightgreen')
        
        c.node('unet', '条件U-Net\n(Conditional U-Net)\n1D卷积 + FiLM',
               fillcolor='lightgreen')
    
    # FiLM调制
    with dot.subgraph(name='cluster_film') as c:
        c.attr(label='FiLM调制', style='filled', color='lightyellow')
        
        c.node('film_encoder', 'FiLM编码器\n(Feature Linear Modulation)\n生成scale和bias',
               fillcolor='lightyellow')
        
        c.node('film_modulation', '特征调制\n(Scale * Feature + Bias)',
               fillcolor='lightyellow')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出', style='filled', color='lightcoral')
        
        c.node('prediction', '预测输出\n(噪声或样本)',
               fillcolor='lightcoral')
    
    # 连接关系
    dot.edge('global_cond', 'film_encoder')
    dot.edge('timestep', 'film_encoder')
    
    dot.edge('film_encoder', 'film_modulation')
    dot.edge('noise_scheduler', 'unet')
    dot.edge('film_modulation', 'unet')
    
    dot.edge('unet', 'prediction')
    
    dot.render('./images/conditional_diffusion', view=True, cleanup=True)
    print("条件扩散建模架构图已生成: ./images/conditional_diffusion.png")

def create_queue_caching():
    """队列缓存机制架构图"""
    
    dot = graphviz.Digraph('Queue_Caching',
                          comment='队列缓存机制架构',
                          format='png')
    
    dot.attr(rankdir='LR', size='16,10', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 观察队列
    with dot.subgraph(name='cluster_obs_queue') as c:
        c.attr(label='观察队列 (Observation Queue)', style='filled', color='lightblue')
        
        c.node('obs_1', 'obs_t-1', fillcolor='lightcyan')
        c.node('obs_2', 'obs_t', fillcolor='lightcyan')
        c.node('obs_3', 'obs_t+1', fillcolor='lightcyan')
        c.node('obs_queue', '观察队列\n(maxlen=n_obs_steps)', fillcolor='lightblue')
    
    # 动作队列
    with dot.subgraph(name='cluster_action_queue') as c:
        c.attr(label='动作队列 (Action Queue)', style='filled', color='lightgreen')
        
        c.node('action_1', 'action_t', fillcolor='lightgreen')
        c.node('action_2', 'action_t+1', fillcolor='lightgreen')
        c.node('action_3', 'action_t+2', fillcolor='lightgreen')
        c.node('action_queue', '动作队列\n(maxlen=n_action_steps)', fillcolor='lightgreen')
    
    # 策略组件
    with dot.subgraph(name='cluster_policy') as c:
        c.attr(label='策略组件', style='filled', color='lightyellow')
        
        c.node('diffusion_model', '扩散模型\n(Diffusion Model)',
               fillcolor='lightyellow')
        
        c.node('select_action', '动作选择\n(select_action)',
               fillcolor='lightyellow')
    
    # 环境交互
    with dot.subgraph(name='cluster_environment') as c:
        c.attr(label='环境交互', style='filled', color='lightcoral')
        
        c.node('env', '环境\n(Environment)',
               fillcolor='lightcoral')
    
    # 连接关系
    # 观察队列操作
    dot.edge('obs_1', 'obs_queue')
    dot.edge('obs_2', 'obs_queue')
    dot.edge('obs_3', 'obs_queue')
    dot.edge('obs_queue', 'diffusion_model')
    
    # 动作生成和缓存
    dot.edge('diffusion_model', 'action_1')
    dot.edge('action_1', 'action_queue')
    dot.edge('action_2', 'action_queue')
    dot.edge('action_3', 'action_queue')
    
    # 动作选择
    dot.edge('action_queue', 'select_action')
    dot.edge('select_action', 'env')
    
    # 环境反馈
    dot.edge('env', 'obs_1', style='dashed')
    
    dot.render('./images/queue_caching', view=True, cleanup=True)
    print("队列缓存机制架构图已生成: ./images/queue_caching.png")

if __name__ == "__main__":
    create_multimodal_fusion()
    create_temporal_action_generation()
    create_conditional_diffusion()
    create_queue_caching() 