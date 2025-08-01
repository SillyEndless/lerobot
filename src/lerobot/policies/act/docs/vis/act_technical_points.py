#!/usr/bin/env python3
"""
ACT架构知识点和主要技术点可视化脚本
生成ACT模型的关键技术点图解
"""

import graphviz

def create_action_chunking_diagram():
    """创建动作分块机制图"""
    dot = graphviz.Digraph(
        'Action_Chunking',
        comment='动作分块机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='LR', size='12,8', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 连续动作序列
    with dot.subgraph(name='cluster_continuous') as c:
        c.attr(label='连续动作序列', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        for i in range(10):
            c.node(f'action_{i}', f'a_{i}', fillcolor='#E8F4FD', color='blue')
            if i > 0:
                c.edge(f'action_{i-1}', f'action_{i}')
    
    # 分块处理
    with dot.subgraph(name='cluster_chunks') as c:
        c.attr(label='分块处理 (chunk_size=4)', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        # 第一个块
        c.node('chunk1', 'Chunk 1\n[a₀, a₁, a₂, a₃]', fillcolor='#FFF2CC', color='orange')
        c.node('pred1', '预测输出\n[â₀, â₁, â₂, â₃]', fillcolor='#D5E8D4', color='green')
        c.edge('chunk1', 'pred1')
        
        # 第二个块
        c.node('chunk2', 'Chunk 2\n[a₄, a₅, a₆, a₇]', fillcolor='#FFF2CC', color='orange')
        c.node('pred2', '预测输出\n[â₄, â₅, â₆, â₇]', fillcolor='#D5E8D4', color='green')
        c.edge('chunk2', 'pred2')
        
        # 第三个块
        c.node('chunk3', 'Chunk 3\n[a₈, a₉, ...]', fillcolor='#FFF2CC', color='orange')
        c.node('pred3', '预测输出\n[â₈, â₉, ...]', fillcolor='#D5E8D4', color='green')
        c.edge('chunk3', 'pred3')
    
    # 连接连续序列到分块
    dot.edge('action_3', 'chunk1')
    dot.edge('action_7', 'chunk2')
    dot.edge('action_9', 'chunk3')
    
    return dot

def create_multimodal_fusion_diagram():
    """创建多模态融合图"""
    dot = graphviz.Digraph(
        'Multimodal_Fusion',
        comment='多模态融合机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 输入模态
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='多模态输入', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('images', '图像观察\n(B, C, H, W)', fillcolor='#E8F4FD', color='blue')
        c.node('robot_state', '机器人状态\n(B, state_dim)', fillcolor='#E8F4FD', color='blue')
        c.node('env_state', '环境状态\n(B, env_dim)', fillcolor='#E8F4FD', color='blue')
        c.node('latent', '潜在表示\n(B, latent_dim)', fillcolor='#E8F4FD', color='blue')
    
    # 特征提取
    with dot.subgraph(name='cluster_features') as c:
        c.attr(label='特征提取', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('img_features', '图像特征\n(B, dim_model)', fillcolor='#FFF2CC', color='orange')
        c.node('state_features', '状态特征\n(B, dim_model)', fillcolor='#FFF2CC', color='orange')
        c.node('latent_features', '潜在特征\n(B, dim_model)', fillcolor='#FFF2CC', color='orange')
    
    # Transformer编码器
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Transformer编码器', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        c.node('encoder', '多头注意力\n+ 前馈网络', fillcolor='#E1D5E7', color='purple')
        c.node('fused_features', '融合特征\n(B, seq_len, dim_model)', fillcolor='#E1D5E7', color='purple')
    
    # 连接
    dot.edge('images', 'img_features')
    dot.edge('robot_state', 'state_features')
    dot.edge('env_state', 'state_features')
    dot.edge('latent', 'latent_features')
    
    dot.edge('img_features', 'encoder')
    dot.edge('state_features', 'encoder')
    dot.edge('latent_features', 'encoder')
    dot.edge('encoder', 'fused_features')
    
    return dot

def create_vae_mechanism_diagram():
    """创建VAE机制图"""
    dot = graphviz.Digraph(
        'VAE_Mechanism',
        comment='VAE变分自编码器机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='LR', size='14,8', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 编码器部分
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='VAE编码器', style='filled', color='lightcoral', fontname='SimHei', fontsize='14')
        
        c.node('input_actions', '动作序列\n(B, chunk_size, action_dim)', fillcolor='#F8CECC', color='red')
        c.node('vae_encoder', 'Transformer编码器\n(ACTEncoder)', fillcolor='#F8CECC', color='red')
        c.node('latent_params', '潜在参数\nμ, log(σ²)', fillcolor='#F8CECC', color='red')
        c.node('latent_sample', '潜在采样\n(重参数化)', fillcolor='#F8CECC', color='red')
    
    # 解码器部分
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='VAE解码器', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('transformer_decoder', 'Transformer解码器\n(ACTDecoder)', fillcolor='#D5E8D4', color='green')
        c.node('reconstructed', '重构动作\n(B, chunk_size, action_dim)', fillcolor='#D5E8D4', color='green')
    
    # 损失计算
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='损失函数', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('recon_loss', '重构损失\n(L1 Loss)', fillcolor='#FFF2CC', color='orange')
        c.node('kl_loss', 'KL散度损失\nKL(q||p)', fillcolor='#FFF2CC', color='orange')
        c.node('total_loss', '总损失\nL = L_recon + β·L_KL', fillcolor='#FFF2CC', color='orange')
    
    # 连接
    dot.edge('input_actions', 'vae_encoder')
    dot.edge('vae_encoder', 'latent_params')
    dot.edge('latent_params', 'latent_sample')
    dot.edge('latent_sample', 'transformer_decoder')
    dot.edge('transformer_decoder', 'reconstructed')
    
    dot.edge('input_actions', 'recon_loss')
    dot.edge('reconstructed', 'recon_loss')
    dot.edge('latent_params', 'kl_loss')
    dot.edge('recon_loss', 'total_loss')
    dot.edge('kl_loss', 'total_loss')
    
    return dot

def create_temporal_ensemble_diagram():
    """创建时间集成机制图"""
    dot = graphviz.Digraph(
        'Temporal_Ensemble',
        comment='时间集成机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 历史预测
    with dot.subgraph(name='cluster_history') as c:
        c.attr(label='历史预测序列', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        for i in range(5):
            c.node(f'pred_{i}', f'预测t-{4-i}\n权重: w_{4-i}', fillcolor='#E8F4FD', color='blue')
    
    # 当前预测
    with dot.subgraph(name='cluster_current') as c:
        c.attr(label='当前预测', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('current_pred', '当前预测t\n权重: w₀', fillcolor='#FFF2CC', color='orange')
    
    # 集成过程
    with dot.subgraph(name='cluster_ensemble') as c:
        c.attr(label='指数加权平均', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('weighted_sum', '加权求和\nΣ(wᵢ × aᵢ)', fillcolor='#D5E8D4', color='green')
        c.node('normalized', '归一化\nΣ(wᵢ × aᵢ) / Σwᵢ', fillcolor='#D5E8D4', color='green')
        c.node('final_action', '最终动作\na_final', fillcolor='#D5E8D4', color='green')
    
    # 权重计算
    with dot.subgraph(name='cluster_weights') as c:
        c.attr(label='权重计算', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        c.node('weight_formula', 'wᵢ = exp(-α·i)\nα = temporal_ensemble_coeff', fillcolor='#E1D5E7', color='purple')
    
    # 连接
    for i in range(5):
        dot.edge(f'pred_{i}', 'weighted_sum')
    dot.edge('current_pred', 'weighted_sum')
    dot.edge('weighted_sum', 'normalized')
    dot.edge('normalized', 'final_action')
    dot.edge('weight_formula', 'weighted_sum')
    
    return dot

def create_position_encoding_diagram():
    """创建位置编码图"""
    dot = graphviz.Digraph(
        'Position_Encoding',
        comment='位置编码机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='LR', size='14,8', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 1D位置编码
    with dot.subgraph(name='cluster_1d') as c:
        c.attr(label='1D正弦位置编码', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('pos_1d', '位置索引\npos', fillcolor='#E8F4FD', color='blue')
        c.node('sin_1d', '正弦函数\nsin(pos/10000^(2i/d))', fillcolor='#E8F4FD', color='blue')
        c.node('cos_1d', '余弦函数\ncos(pos/10000^(2i/d))', fillcolor='#E8F4FD', color='blue')
        c.node('embed_1d', '位置嵌入\nPE(pos)', fillcolor='#E8F4FD', color='blue')
    
    # 2D位置编码
    with dot.subgraph(name='cluster_2d') as c:
        c.attr(label='2D正弦位置编码', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('pos_2d', '2D坐标\n(x, y)', fillcolor='#FFF2CC', color='orange')
        c.node('normalize_2d', '归一化\n[0, 2π]', fillcolor='#FFF2CC', color='orange')
        c.node('freq_2d', '频率计算\n1/10000^(2i/d)', fillcolor='#FFF2CC', color='orange')
        c.node('embed_2d', '2D位置嵌入\nPE(x, y)', fillcolor='#FFF2CC', color='orange')
    
    # 应用
    with dot.subgraph(name='cluster_application') as c:
        c.attr(label='位置编码应用', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('token_embed', 'Token嵌入', fillcolor='#D5E8D4', color='green')
        c.node('pos_embed', '位置嵌入', fillcolor='#D5E8D4', color='green')
        c.node('final_embed', '最终嵌入\nToken + PE', fillcolor='#D5E8D4', color='green')
    
    # 连接1D
    dot.edge('pos_1d', 'sin_1d')
    dot.edge('pos_1d', 'cos_1d')
    dot.edge('sin_1d', 'embed_1d')
    dot.edge('cos_1d', 'embed_1d')
    
    # 连接2D
    dot.edge('pos_2d', 'normalize_2d')
    dot.edge('normalize_2d', 'freq_2d')
    dot.edge('freq_2d', 'embed_2d')
    
    # 连接应用
    dot.edge('embed_1d', 'pos_embed')
    dot.edge('embed_2d', 'pos_embed')
    dot.edge('token_embed', 'final_embed')
    dot.edge('pos_embed', 'final_embed')
    
    return dot

def main():
    """主函数"""
    print("正在生成ACT技术点图解...")
    
    # 生成各个技术点图
    diagrams = [
        ('action_chunking', create_action_chunking_diagram()),
        ('multimodal_fusion', create_multimodal_fusion_diagram()),
        ('vae_mechanism', create_vae_mechanism_diagram()),
        ('temporal_ensemble', create_temporal_ensemble_diagram()),
        ('position_encoding', create_position_encoding_diagram())
    ]
    
    for name, dot in diagrams:
        output_path = f'./images/act_{name}'
        dot.render(output_path, cleanup=True)
        print(f"已生成: {output_path}.png")
    
    print("\n所有技术点图解已生成完成！")
    print("包含以下技术点：")
    print("1. 动作分块机制")
    print("2. 多模态融合")
    print("3. VAE变分自编码器")
    print("4. 时间集成推理")
    print("5. 位置编码机制")

if __name__ == "__main__":
    main() 