#!/usr/bin/env python3
"""
Diffusion Policy 主要技术点可视化脚本
"""

import graphviz

def create_diffusion_process():
    """扩散过程技术图"""
    
    dot = graphviz.Digraph('Diffusion_Process',
                          comment='扩散过程技术详解',
                          format='png')
    
    dot.attr(rankdir='TB', size='14,18', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 前向扩散过程
    with dot.subgraph(name='cluster_forward') as c:
        c.attr(label='前向扩散过程 (Forward Diffusion)', style='filled', color='lightblue')
        
        c.node('x0', 'x₀ (干净轨迹)\n(Clean Trajectory)', fillcolor='lightcyan')
        c.node('x1', 'x₁', fillcolor='lightcyan')
        c.node('x2', 'x₂', fillcolor='lightcyan')
        c.node('xt', 'xₜ', fillcolor='lightcyan')
        c.node('xT', 'xₜ (纯噪声)\n(Pure Noise)', fillcolor='lightcyan')
        
        c.node('eps', 'ε (噪声)\n(Noise)', fillcolor='lightyellow')
        c.node('beta', 'βₜ (噪声调度)\n(Noise Schedule)', fillcolor='lightyellow')
    
    # 反向扩散过程
    with dot.subgraph(name='cluster_reverse') as c:
        c.attr(label='反向扩散过程 (Reverse Diffusion)', style='filled', color='lightgreen')
        
        c.node('xT_rev', 'xₜ (纯噪声)', fillcolor='lightgreen')
        c.node('xt_rev', 'xₜ', fillcolor='lightgreen')
        c.node('x2_rev', 'x₂', fillcolor='lightgreen')
        c.node('x1_rev', 'x₁', fillcolor='lightgreen')
        c.node('x0_rev', 'x₀ (生成轨迹)', fillcolor='lightgreen')
        
        c.node('unet', 'U-Net\n(去噪网络)', fillcolor='lightcoral')
        c.node('pred', '预测\n(Prediction)', fillcolor='lightcoral')
    
    # 数学公式
    with dot.subgraph(name='cluster_math') as c:
        c.attr(label='数学公式', style='filled', color='lightgray')
        
        c.node('forward_eq', '前向: xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε', fillcolor='lightgray')
        c.node('reverse_eq', '反向: xₜ₋₁ = f(xₜ, ε_θ, t)', fillcolor='lightgray')
    
    # 连接关系
    # 前向过程
    dot.edge('x0', 'x1')
    dot.edge('x1', 'x2')
    dot.edge('x2', 'xt')
    dot.edge('xt', 'xT')
    dot.edge('eps', 'x1')
    dot.edge('beta', 'x1')
    
    # 反向过程
    dot.edge('xT_rev', 'xt_rev')
    dot.edge('xt_rev', 'x2_rev')
    dot.edge('x2_rev', 'x1_rev')
    dot.edge('x1_rev', 'x0_rev')
    
    dot.edge('xt_rev', 'unet')
    dot.edge('unet', 'pred')
    dot.edge('pred', 'x2_rev')
    
    # 数学公式连接
    dot.edge('forward_eq', 'x1', style='dashed')
    dot.edge('reverse_eq', 'x1_rev', style='dashed')
    
    dot.render('./images/diffusion_process', view=True, cleanup=True)
    print("扩散过程技术图已生成: ./images/diffusion_process.png")

def create_film_modulation():
    """FiLM调制技术图"""
    
    dot = graphviz.Digraph('FiLM_Modulation',
                          comment='FiLM调制技术详解',
                          format='png')
    
    dot.attr(rankdir='LR', size='16,12', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='输入', style='filled', color='lightblue')
        
        c.node('feature', '特征图\n(Feature Map)\n(B, C, T)', fillcolor='lightcyan')
        c.node('condition', '条件信息\n(Condition)\n全局条件 + 时间步', fillcolor='lightcyan')
    
    # FiLM编码器
    with dot.subgraph(name='cluster_film_encoder') as c:
        c.attr(label='FiLM编码器', style='filled', color='lightgreen')
        
        c.node('film_net', 'FiLM网络\n(Mish + Linear)\n生成γ和β', fillcolor='lightgreen')
        c.node('gamma', 'γ (Scale)\n(Scale Parameter)', fillcolor='lightgreen')
        c.node('beta', 'β (Bias)\n(Bias Parameter)', fillcolor='lightgreen')
    
    # 调制过程
    with dot.subgraph(name='cluster_modulation') as c:
        c.attr(label='调制过程', style='filled', color='lightyellow')
        
        c.node('scale_op', '缩放操作\n(Scale Operation)\nγ * Feature', fillcolor='lightyellow')
        c.node('bias_op', '偏置操作\n(Bias Operation)\n+ β', fillcolor='lightyellow')
        c.node('modulated', '调制后特征\n(Modulated Feature)', fillcolor='lightyellow')
    
    # 连接关系
    dot.edge('condition', 'film_net')
    dot.edge('film_net', 'gamma')
    dot.edge('film_net', 'beta')
    
    dot.edge('feature', 'scale_op')
    dot.edge('gamma', 'scale_op')
    dot.edge('scale_op', 'bias_op')
    dot.edge('beta', 'bias_op')
    dot.edge('bias_op', 'modulated')
    
    # 数学公式
    with dot.subgraph(name='cluster_math') as c:
        c.attr(label='数学公式', style='filled', color='lightgray')
        
        c.node('film_eq', 'FiLM(x) = γ * x + β', fillcolor='lightgray')
    
    dot.edge('film_eq', 'modulated', style='dashed')
    
    dot.render('./images/film_modulation', view=True, cleanup=True)
    print("FiLM调制技术图已生成: ./images/film_modulation.png")

def create_spatial_softmax():
    """空间软最大值技术图"""
    
    dot = graphviz.Digraph('Spatial_Softmax',
                          comment='空间软最大值技术详解',
                          format='png')
    
    dot.attr(rankdir='TB', size='14,16', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入特征图
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入特征图', style='filled', color='lightblue')
        
        c.node('feature_map', '特征图\n(Feature Map)\n(B, C, H, W)', fillcolor='lightcyan')
        c.node('grid', '坐标网格\n(Coordinate Grid)\n(H×W, 2)', fillcolor='lightcyan')
    
    # 处理过程
    with dot.subgraph(name='cluster_process') as c:
        c.attr(label='处理过程', style='filled', color='lightgreen')
        
        c.node('reshape', '重塑\n(Reshape)\n(B×C, H×W)', fillcolor='lightgreen')
        c.node('softmax', '软最大值\n(Softmax)\n注意力权重', fillcolor='lightgreen')
        c.node('dot_product', '点积\n(Dot Product)\n权重 × 坐标', fillcolor='lightgreen')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出', style='filled', color='lightyellow')
        
        c.node('keypoints', '关键点\n(Keypoints)\n(B, C, 2)', fillcolor='lightyellow')
    
    # 连接关系
    dot.edge('feature_map', 'reshape')
    dot.edge('reshape', 'softmax')
    dot.edge('softmax', 'dot_product')
    dot.edge('grid', 'dot_product')
    dot.edge('dot_product', 'keypoints')
    
    # 数学公式
    with dot.subgraph(name='cluster_math') as c:
        c.attr(label='数学公式', style='filled', color='lightgray')
        
        c.node('attention_eq', 'Attention = Softmax(Feature)', fillcolor='lightgray')
        c.node('keypoint_eq', 'Keypoint = Attention @ Grid', fillcolor='lightgray')
    
    dot.edge('attention_eq', 'softmax', style='dashed')
    dot.edge('keypoint_eq', 'keypoints', style='dashed')
    
    dot.render('./images/spatial_softmax', view=True, cleanup=True)
    print("空间软最大值技术图已生成: ./images/spatial_softmax.png")

def create_1d_conv_unet():
    """1D卷积U-Net技术图"""
    
    dot = graphviz.Digraph('1D_Conv_UNet',
                          comment='1D卷积U-Net技术详解',
                          format='png')
    
    dot.attr(rankdir='TB', size='16,20', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入', style='filled', color='lightblue')
        
        c.node('input', '输入序列\n(Input Sequence)\n(B, T, D)', fillcolor='lightcyan')
    
    # 编码器
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='编码器 (Encoder)', style='filled', color='lightgreen')
        
        c.node('conv1', 'Conv1d Block 1\n(D, D1)', fillcolor='lightgreen')
        c.node('conv2', 'Conv1d Block 2\n(D1, D2)', fillcolor='lightgreen')
        c.node('conv3', 'Conv1d Block 3\n(D2, D3)', fillcolor='lightgreen')
        
        c.node('down1', 'Downsample 1\n(Conv1d 3,2,1)', fillcolor='lightgreen')
        c.node('down2', 'Downsample 2\n(Conv1d 3,2,1)', fillcolor='lightgreen')
    
    # 中间层
    with dot.subgraph(name='cluster_middle') as c:
        c.attr(label='中间层 (Middle)', style='filled', color='lightyellow')
        
        c.node('mid1', 'Middle Block 1\n(D3, D3)', fillcolor='lightyellow')
        c.node('mid2', 'Middle Block 2\n(D3, D3)', fillcolor='lightyellow')
    
    # 解码器
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='解码器 (Decoder)', style='filled', color='lightcoral')
        
        c.node('up1', 'Upsample 1\n(ConvTranspose1d)', fillcolor='lightcoral')
        c.node('conv4', 'Conv1d Block 4\n(D3*2, D2)', fillcolor='lightcoral')
        c.node('conv5', 'Conv1d Block 5\n(D2, D2)', fillcolor='lightcoral')
        
        c.node('up2', 'Upsample 2\n(ConvTranspose1d)', fillcolor='lightcoral')
        c.node('conv6', 'Conv1d Block 6\n(D2*2, D1)', fillcolor='lightcoral')
        c.node('conv7', 'Conv1d Block 7\n(D1, D1)', fillcolor='lightcoral')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出', style='filled', color='lightpink')
        
        c.node('final_conv', '最终卷积\n(Final Conv)\n(D1, D)', fillcolor='lightpink')
        c.node('output', '输出序列\n(Output Sequence)\n(B, T, D)', fillcolor='lightpink')
    
    # 跳跃连接
    with dot.subgraph(name='cluster_skip') as c:
        c.attr(label='跳跃连接 (Skip Connections)', style='filled', color='lightgray')
        
        c.node('skip1', 'Skip 1', fillcolor='lightgray')
        c.node('skip2', 'Skip 2', fillcolor='lightgray')
    
    # 连接关系
    # 编码器路径
    dot.edge('input', 'conv1')
    dot.edge('conv1', 'conv2')
    dot.edge('conv2', 'down1')
    dot.edge('down1', 'conv3')
    dot.edge('conv3', 'down2')
    
    # 中间层
    dot.edge('down2', 'mid1')
    dot.edge('mid1', 'mid2')
    
    # 解码器路径
    dot.edge('mid2', 'up1')
    dot.edge('up1', 'conv4')
    dot.edge('conv4', 'conv5')
    dot.edge('conv5', 'up2')
    dot.edge('up2', 'conv6')
    dot.edge('conv6', 'conv7')
    dot.edge('conv7', 'final_conv')
    dot.edge('final_conv', 'output')
    
    # 跳跃连接
    dot.edge('conv2', 'skip1', style='dashed')
    dot.edge('skip1', 'conv4', style='dashed')
    dot.edge('conv3', 'skip2', style='dashed')
    dot.edge('skip2', 'conv6', style='dashed')
    
    dot.render('./images/1d_conv_unet', view=True, cleanup=True)
    print("1D卷积U-Net技术图已生成: ./images/1d_conv_unet.png")

def create_action_sequence_prediction():
    """动作序列预测技术图"""
    
    dot = graphviz.Digraph('Action_Sequence_Prediction',
                          comment='动作序列预测技术详解',
                          format='png')
    
    dot.attr(rankdir='LR', size='18,12', dpi='300', fontname='SimHei', fontsize='14')
    dot.node_attr.update(shape='box', style='filled', fontname='SimHei', fontsize='12')
    
    # 观察历史
    with dot.subgraph(name='cluster_observation') as c:
        c.attr(label='观察历史', style='filled', color='lightblue')
        
        c.node('obs_t_1', 'obs_{t-1}', fillcolor='lightcyan')
        c.node('obs_t', 'obs_t', fillcolor='lightcyan')
        c.node('obs_history', '观察历史\n(Observation History)\nn_obs_steps', fillcolor='lightblue')
    
    # 扩散模型
    with dot.subgraph(name='cluster_diffusion') as c:
        c.attr(label='扩散模型', style='filled', color='lightgreen')
        
        c.node('noise', '噪声采样\n(Noise Sampling)', fillcolor='lightgreen')
        c.node('unet', 'U-Net\n(去噪网络)', fillcolor='lightgreen')
        c.node('scheduler', '噪声调度器\n(Noise Scheduler)', fillcolor='lightgreen')
    
    # 动作序列
    with dot.subgraph(name='cluster_actions') as c:
        c.attr(label='动作序列', style='filled', color='lightyellow')
        
        c.node('action_seq', '动作序列\n(Action Sequence)\nhorizon个时间步', fillcolor='lightyellow')
        c.node('action_t', 'action_t', fillcolor='lightyellow')
        c.node('action_t_1', 'action_{t+1}', fillcolor='lightyellow')
        c.node('action_t_2', 'action_{t+2}', fillcolor='lightyellow')
        c.node('action_chunk', '动作块\n(Action Chunk)\nn_action_steps', fillcolor='lightyellow')
    
    # 时间轴
    with dot.subgraph(name='cluster_timeline') as c:
        c.attr(label='时间轴', style='filled', color='lightgray')
        
        c.node('timeline', '时间轴\n(Timeline)\nt → t+horizon', fillcolor='lightgray')
    
    # 连接关系
    # 观察到扩散模型
    dot.edge('obs_t_1', 'obs_history')
    dot.edge('obs_t', 'obs_history')
    dot.edge('obs_history', 'unet')
    
    # 扩散过程
    dot.edge('noise', 'unet')
    dot.edge('scheduler', 'unet')
    dot.edge('unet', 'action_seq')
    
    # 动作序列分解
    dot.edge('action_seq', 'action_t')
    dot.edge('action_seq', 'action_t_1')
    dot.edge('action_seq', 'action_t_2')
    dot.edge('action_t', 'action_chunk')
    dot.edge('action_t_1', 'action_chunk')
    dot.edge('action_t_2', 'action_chunk')
    
    # 时间轴连接
    dot.edge('timeline', 'action_seq', style='dashed')
    
    dot.render('./images/action_sequence_prediction', view=True, cleanup=True)
    print("动作序列预测技术图已生成: ./images/action_sequence_prediction.png")

if __name__ == "__main__":
    create_diffusion_process()
    create_film_modulation()
    create_spatial_softmax()
    create_1d_conv_unet()
    create_action_sequence_prediction() 