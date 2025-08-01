#!/usr/bin/env python3
"""
ACT模型架构可视化脚本
生成ACT（Action Chunking Transformer）的整体架构图
"""

import graphviz

def create_act_architecture_diagram():
    """创建ACT模型架构图"""
    
    # 创建有向图
    dot = graphviz.Digraph(
        'ACT_Architecture',
        comment='ACT模型整体架构图',
        format='png',
        engine='dot'
    )
    
    # 设置图形属性
    dot.attr(rankdir='TB', size='12,16', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='14')
    
    # 定义颜色方案
    colors = {
        'input': '#E8F4FD',      # 浅蓝色 - 输入
        'encoder': '#FFF2CC',    # 浅黄色 - 编码器
        'decoder': '#E1D5E7',    # 浅紫色 - 解码器
        'output': '#D5E8D4',     # 浅绿色 - 输出
        'vae': '#F8CECC',        # 浅红色 - VAE
        'attention': '#DAE8FC'   # 浅蓝色 - 注意力
    }
    
    # 创建子图：输入层
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='输入层', style='filled', color='lightblue', fontname='SimHei', fontsize='16')
        
        # 输入节点
        c.node('obs_images', '图像观察\n(observation.images)', 
               fillcolor=colors['input'], color='blue')
        c.node('obs_state', '机器人状态\n(observation.state)', 
               fillcolor=colors['input'], color='blue')
        c.node('obs_env', '环境状态\n(observation.environment_state)', 
               fillcolor=colors['input'], color='blue')
        c.node('action_target', '目标动作序列\n(action)', 
               fillcolor=colors['input'], color='blue')
    
    # 创建子图：VAE编码器（可选）
    with dot.subgraph(name='cluster_vae') as c:
        c.attr(label='VAE编码器（训练时）', style='filled', color='lightcoral', fontname='SimHei', fontsize='16')
        
        c.node('vae_encoder', 'VAE编码器\n(ACTEncoder)', 
               fillcolor=colors['vae'], color='red')
        c.node('latent_dist', '潜在分布\n(μ, log(σ²))', 
               fillcolor=colors['vae'], color='red')
        c.node('latent_sample', '潜在采样\n(重参数化)', 
               fillcolor=colors['vae'], color='red')
    
    # 创建子图：视觉特征提取
    with dot.subgraph(name='cluster_vision') as c:
        c.attr(label='视觉特征提取', style='filled', color='lightyellow', fontname='SimHei', fontsize='16')
        
        c.node('backbone', 'ResNet骨干网络\n(ResNet18)', 
               fillcolor=colors['encoder'], color='orange')
        c.node('img_proj', '图像特征投影\n(Conv2d)', 
               fillcolor=colors['encoder'], color='orange')
        c.node('pos_embed_2d', '2D位置编码\n(ACTSinusoidalPositionEmbedding2d)', 
               fillcolor=colors['attention'], color='navy')
    
    # 创建子图：Transformer编码器
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Transformer编码器', style='filled', color='lightyellow', fontname='SimHei', fontsize='16')
        
        c.node('state_proj', '状态投影层\n(Linear)', 
               fillcolor=colors['encoder'], color='orange')
        c.node('latent_proj', '潜在表示投影\n(Linear)', 
               fillcolor=colors['encoder'], color='orange')
        c.node('pos_embed_1d', '1D位置编码\n(Embedding)', 
               fillcolor=colors['attention'], color='navy')
        c.node('transformer_encoder', 'Transformer编码器\n(ACTEncoder)', 
               fillcolor=colors['encoder'], color='orange')
    
    # 创建子图：Transformer解码器
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Transformer解码器', style='filled', color='lightpink', fontname='SimHei', fontsize='16')
        
        c.node('decoder_pos_embed', '解码器位置编码\n(Embedding)', 
               fillcolor=colors['attention'], color='navy')
        c.node('transformer_decoder', 'Transformer解码器\n(ACTDecoder)', 
               fillcolor=colors['decoder'], color='purple')
        c.node('action_head', '动作回归头\n(Linear)', 
               fillcolor=colors['output'], color='green')
    
    # 创建子图：时间集成
    with dot.subgraph(name='cluster_ensemble') as c:
        c.attr(label='时间集成（推理时）', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        
        c.node('temporal_ensemble', '时间集成器\n(ACTTemporalEnsembler)', 
               fillcolor=colors['output'], color='green')
    
    # 创建子图：输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层', style='filled', color='lightgreen', fontname='SimHei', fontsize='16')
        
        c.node('predicted_actions', '预测动作序列\n(B, chunk_size, action_dim)', 
               fillcolor=colors['output'], color='green')
    
    # 添加边：输入到VAE
    dot.edge('obs_state', 'vae_encoder')
    dot.edge('action_target', 'vae_encoder')
    dot.edge('vae_encoder', 'latent_dist')
    dot.edge('latent_dist', 'latent_sample')
    
    # 添加边：输入到视觉处理
    dot.edge('obs_images', 'backbone')
    dot.edge('backbone', 'img_proj')
    dot.edge('img_proj', 'pos_embed_2d')
    
    # 添加边：输入到编码器
    dot.edge('obs_state', 'state_proj')
    dot.edge('obs_env', 'state_proj')
    dot.edge('latent_sample', 'latent_proj')
    dot.edge('pos_embed_2d', 'transformer_encoder')
    dot.edge('state_proj', 'transformer_encoder')
    dot.edge('latent_proj', 'transformer_encoder')
    dot.edge('pos_embed_1d', 'transformer_encoder')
    
    # 添加边：编码器到解码器
    dot.edge('transformer_encoder', 'transformer_decoder')
    dot.edge('decoder_pos_embed', 'transformer_decoder')
    dot.edge('transformer_decoder', 'action_head')
    
    # 添加边：到时间集成和输出
    dot.edge('action_head', 'temporal_ensemble')
    dot.edge('temporal_ensemble', 'predicted_actions')
    
    # 添加边：直接输出（训练时）
    dot.edge('action_head', 'predicted_actions')
    
    return dot

def main():
    """主函数"""
    print("正在生成ACT模型架构图...")
    
    # 创建架构图
    dot = create_act_architecture_diagram()
    
    # 保存图片
    output_path = './images/act_architecture'
    dot.render(output_path, cleanup=True)
    
    print(f"架构图已保存到: {output_path}.png")
    print("图片特点：")
    print("- 使用不同颜色区分不同模块")
    print("- 清晰显示数据流向")
    print("- 标注了关键组件和维度信息")
    print("- 支持中文显示")

if __name__ == "__main__":
    main() 