#!/usr/bin/env python3
"""
Diffusion Policy 模型架构可视化脚本
"""

import graphviz

def create_model_architecture():
    """创建Diffusion Policy的整体模型架构图"""
    
    # 创建有向图
    dot = graphviz.Digraph('DiffusionPolicy_Architecture', 
                          comment='Diffusion Policy 模型架构',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='12,16',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='12',
                         margin='0.3')
    
    # 输入层节点
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='输入层 (Input Layer)',
               style='filled',
               color='lightblue',
               fontname='SimHei',
               fontsize='14')
        
        c.node('obs_state', '机器人状态\n(observation.state)\n(B, n_obs_steps, state_dim)',
               fillcolor='lightcyan')
        
        c.node('obs_images', '图像观察\n(observation.images)\n(B, n_obs_steps, num_cameras, C, H, W)',
               fillcolor='lightcyan')
        
        c.node('obs_env', '环境状态\n(observation.environment_state)\n(B, n_obs_steps, env_dim)',
               fillcolor='lightcyan')
    
    # 编码器层节点
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(label='编码器层 (Encoder Layer)',
               style='filled',
               color='lightgreen',
               fontname='SimHei',
               fontsize='14')
        
        c.node('rgb_encoder', '视觉编码器\n(DiffusionRgbEncoder)\nResNet + SpatialSoftmax',
               fillcolor='lightgreen')
        
        c.node('state_encoder', '状态编码器\n(直接使用状态向量)',
               fillcolor='lightgreen')
        
        c.node('env_encoder', '环境状态编码器\n(直接使用环境向量)',
               fillcolor='lightgreen')
    
    # 融合层节点
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='特征融合层 (Feature Fusion)',
               style='filled',
               color='lightyellow',
               fontname='SimHei',
               fontsize='14')
        
        c.node('global_cond', '全局条件特征\n(Global Conditioning)\n(B, global_cond_dim)',
               fillcolor='lightyellow')
    
    # 扩散模型层节点
    with dot.subgraph(name='cluster_diffusion') as c:
        c.attr(label='扩散模型层 (Diffusion Model)',
               style='filled',
               color='lightcoral',
               fontname='SimHei',
               fontsize='14')
        
        c.node('unet', '条件U-Net\n(DiffusionConditionalUnet1d)\n1D卷积 + FiLM调制',
               fillcolor='lightcoral')
        
        c.node('noise_scheduler', '噪声调度器\n(DDPM/DDIM)\n噪声添加与去噪',
               fillcolor='lightcoral')
    
    # 输出层节点
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层 (Output Layer)',
               style='filled',
               color='lightpink',
               fontname='SimHei',
               fontsize='14')
        
        c.node('action_seq', '动作序列\n(action)\n(B, horizon, action_dim)',
               fillcolor='lightpink')
        
        c.node('action_chunk', '执行动作块\n(action chunk)\n(B, n_action_steps, action_dim)',
               fillcolor='lightpink')
    
    # 队列管理节点
    with dot.subgraph(name='cluster_queues') as c:
        c.attr(label='队列管理 (Queue Management)',
               style='filled',
               color='lightgray',
               fontname='SimHei',
               fontsize='14')
        
        c.node('obs_queue', '观察队列\n(Observation Queue)\n历史观察缓存',
               fillcolor='lightgray')
        
        c.node('action_queue', '动作队列\n(Action Queue)\n生成动作缓存',
               fillcolor='lightgray')
    
    # 连接关系
    # 输入到编码器
    dot.edge('obs_state', 'state_encoder')
    dot.edge('obs_images', 'rgb_encoder')
    dot.edge('obs_env', 'env_encoder')
    
    # 编码器到融合
    dot.edge('state_encoder', 'global_cond')
    dot.edge('rgb_encoder', 'global_cond')
    dot.edge('env_encoder', 'global_cond')
    
    # 融合到扩散模型
    dot.edge('global_cond', 'unet')
    dot.edge('noise_scheduler', 'unet')
    
    # 扩散模型到输出
    dot.edge('unet', 'action_seq')
    dot.edge('action_seq', 'action_chunk')
    
    # 队列连接
    dot.edge('obs_queue', 'global_cond', style='dashed')
    dot.edge('action_seq', 'action_queue', style='dashed')
    dot.edge('action_queue', 'action_chunk', style='dashed')
    
    # 保存图片
    dot.render('./images/diffusion_policy_architecture', view=True, cleanup=True)
    print("模型架构图已生成: ./images/diffusion_policy_architecture.png")

if __name__ == "__main__":
    create_model_architecture() 