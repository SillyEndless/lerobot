#!/usr/bin/env python3
"""
ACT注意力机制可视化脚本
生成自注意力、交叉注意力和多头注意力的图解
"""

import graphviz

def create_self_attention_diagram():
    """创建自注意力机制图"""
    dot = graphviz.Digraph(
        'Self_Attention',
        comment='自注意力机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 输入序列
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入序列', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('x1', 'x₁\n(位置1)', fillcolor='#E8F4FD', color='blue')
        c.node('x2', 'x₂\n(位置2)', fillcolor='#E8F4FD', color='blue')
        c.node('x3', 'x₃\n(位置3)', fillcolor='#E8F4FD', color='blue')
        c.node('x4', 'x₄\n(位置4)', fillcolor='#E8F4FD', color='blue')
    
    # 线性变换
    with dot.subgraph(name='cluster_linear') as c:
        c.attr(label='线性变换', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('q1', 'Q₁\n(Query)', fillcolor='#FFF2CC', color='orange')
        c.node('k1', 'K₁\n(Key)', fillcolor='#FFF2CC', color='orange')
        c.node('v1', 'V₁\n(Value)', fillcolor='#FFF2CC', color='orange')
        
        c.node('q2', 'Q₂', fillcolor='#FFF2CC', color='orange')
        c.node('k2', 'K₂', fillcolor='#FFF2CC', color='orange')
        c.node('v2', 'V₂', fillcolor='#FFF2CC', color='orange')
        
        c.node('q3', 'Q₃', fillcolor='#FFF2CC', color='orange')
        c.node('k3', 'K₃', fillcolor='#FFF2CC', color='orange')
        c.node('v3', 'V₃', fillcolor='#FFF2CC', color='orange')
        
        c.node('q4', 'Q₄', fillcolor='#FFF2CC', color='orange')
        c.node('k4', 'K₄', fillcolor='#FFF2CC', color='orange')
        c.node('v4', 'V₄', fillcolor='#FFF2CC', color='orange')
    
    # 注意力计算
    with dot.subgraph(name='cluster_attention') as c:
        c.attr(label='注意力计算', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        c.node('attn1', 'Attention(Q₁,K,V)\n= softmax(Q₁K^T/√d_k)V', fillcolor='#E1D5E7', color='purple')
        c.node('attn2', 'Attention(Q₂,K,V)', fillcolor='#E1D5E7', color='purple')
        c.node('attn3', 'Attention(Q₃,K,V)', fillcolor='#E1D5E7', color='purple')
        c.node('attn4', 'Attention(Q₄,K,V)', fillcolor='#E1D5E7', color='purple')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出序列', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('out1', '输出₁\n(位置1)', fillcolor='#D5E8D4', color='green')
        c.node('out2', '输出₂\n(位置2)', fillcolor='#D5E8D4', color='green')
        c.node('out3', '输出₃\n(位置3)', fillcolor='#D5E8D4', color='green')
        c.node('out4', '输出₄\n(位置4)', fillcolor='#D5E8D4', color='green')
    
    # 连接输入到线性变换
    dot.edge('x1', 'q1')
    dot.edge('x1', 'k1')
    dot.edge('x1', 'v1')
    dot.edge('x2', 'q2')
    dot.edge('x2', 'k2')
    dot.edge('x2', 'v2')
    dot.edge('x3', 'q3')
    dot.edge('x3', 'k3')
    dot.edge('x3', 'v3')
    dot.edge('x4', 'q4')
    dot.edge('x4', 'k4')
    dot.edge('x4', 'v4')
    
    # 连接注意力计算
    dot.edge('q1', 'attn1')
    dot.edge('k1', 'attn1')
    dot.edge('k2', 'attn1')
    dot.edge('k3', 'attn1')
    dot.edge('k4', 'attn1')
    dot.edge('v1', 'attn1')
    dot.edge('v2', 'attn1')
    dot.edge('v3', 'attn1')
    dot.edge('v4', 'attn1')
    
    dot.edge('q2', 'attn2')
    dot.edge('k1', 'attn2')
    dot.edge('k2', 'attn2')
    dot.edge('k3', 'attn2')
    dot.edge('k4', 'attn2')
    dot.edge('v1', 'attn2')
    dot.edge('v2', 'attn2')
    dot.edge('v3', 'attn2')
    dot.edge('v4', 'attn2')
    
    dot.edge('q3', 'attn3')
    dot.edge('k1', 'attn3')
    dot.edge('k2', 'attn3')
    dot.edge('k3', 'attn3')
    dot.edge('k4', 'attn3')
    dot.edge('v1', 'attn3')
    dot.edge('v2', 'attn3')
    dot.edge('v3', 'attn3')
    dot.edge('v4', 'attn3')
    
    dot.edge('q4', 'attn4')
    dot.edge('k1', 'attn4')
    dot.edge('k2', 'attn4')
    dot.edge('k3', 'attn4')
    dot.edge('k4', 'attn4')
    dot.edge('v1', 'attn4')
    dot.edge('v2', 'attn4')
    dot.edge('v3', 'attn4')
    dot.edge('v4', 'attn4')
    
    # 连接输出
    dot.edge('attn1', 'out1')
    dot.edge('attn2', 'out2')
    dot.edge('attn3', 'out3')
    dot.edge('attn4', 'out4')
    
    return dot

def create_cross_attention_diagram():
    """创建交叉注意力机制图"""
    dot = graphviz.Digraph(
        'Cross_Attention',
        comment='交叉注意力机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='LR', size='14,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 编码器输出
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='编码器输出', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('enc1', '编码器输出₁', fillcolor='#E8F4FD', color='blue')
        c.node('enc2', '编码器输出₂', fillcolor='#E8F4FD', color='blue')
        c.node('enc3', '编码器输出₃', fillcolor='#E8F4FD', color='blue')
        c.node('enc4', '编码器输出₄', fillcolor='#E8F4FD', color='blue')
    
    # 解码器输入
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='解码器输入', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('dec1', '解码器输入₁', fillcolor='#FFF2CC', color='orange')
        c.node('dec2', '解码器输入₂', fillcolor='#FFF2CC', color='orange')
        c.node('dec3', '解码器输入₃', fillcolor='#FFF2CC', color='orange')
    
    # 线性变换
    with dot.subgraph(name='cluster_linear') as c:
        c.attr(label='线性变换', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        # 编码器到Key和Value
        c.node('k1', 'K₁', fillcolor='#E1D5E7', color='purple')
        c.node('k2', 'K₂', fillcolor='#E1D5E7', color='purple')
        c.node('k3', 'K₃', fillcolor='#E1D5E7', color='purple')
        c.node('k4', 'K₄', fillcolor='#E1D5E7', color='purple')
        
        c.node('v1', 'V₁', fillcolor='#E1D5E7', color='purple')
        c.node('v2', 'V₂', fillcolor='#E1D5E7', color='purple')
        c.node('v3', 'V₃', fillcolor='#E1D5E7', color='purple')
        c.node('v4', 'V₄', fillcolor='#E1D5E7', color='purple')
        
        # 解码器到Query
        c.node('q1', 'Q₁', fillcolor='#E1D5E7', color='purple')
        c.node('q2', 'Q₂', fillcolor='#E1D5E7', color='purple')
        c.node('q3', 'Q₃', fillcolor='#E1D5E7', color='purple')
    
    # 交叉注意力计算
    with dot.subgraph(name='cluster_cross_attn') as c:
        c.attr(label='交叉注意力', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('cross_attn1', 'CrossAttention(Q₁,K,V)\n= softmax(Q₁K^T/√d_k)V', fillcolor='#D5E8D4', color='green')
        c.node('cross_attn2', 'CrossAttention(Q₂,K,V)', fillcolor='#D5E8D4', color='green')
        c.node('cross_attn3', 'CrossAttention(Q₃,K,V)', fillcolor='#D5E8D4', color='green')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='解码器输出', style='filled', color='lightcoral', fontname='SimHei', fontsize='14')
        
        c.node('out1', '解码器输出₁', fillcolor='#F8CECC', color='red')
        c.node('out2', '解码器输出₂', fillcolor='#F8CECC', color='red')
        c.node('out3', '解码器输出₃', fillcolor='#F8CECC', color='red')
    
    # 连接编码器到Key和Value
    dot.edge('enc1', 'k1')
    dot.edge('enc1', 'v1')
    dot.edge('enc2', 'k2')
    dot.edge('enc2', 'v2')
    dot.edge('enc3', 'k3')
    dot.edge('enc3', 'v3')
    dot.edge('enc4', 'k4')
    dot.edge('enc4', 'v4')
    
    # 连接解码器到Query
    dot.edge('dec1', 'q1')
    dot.edge('dec2', 'q2')
    dot.edge('dec3', 'q3')
    
    # 连接交叉注意力
    dot.edge('q1', 'cross_attn1')
    dot.edge('k1', 'cross_attn1')
    dot.edge('k2', 'cross_attn1')
    dot.edge('k3', 'cross_attn1')
    dot.edge('k4', 'cross_attn1')
    dot.edge('v1', 'cross_attn1')
    dot.edge('v2', 'cross_attn1')
    dot.edge('v3', 'cross_attn1')
    dot.edge('v4', 'cross_attn1')
    
    dot.edge('q2', 'cross_attn2')
    dot.edge('k1', 'cross_attn2')
    dot.edge('k2', 'cross_attn2')
    dot.edge('k3', 'cross_attn2')
    dot.edge('k4', 'cross_attn2')
    dot.edge('v1', 'cross_attn2')
    dot.edge('v2', 'cross_attn2')
    dot.edge('v3', 'cross_attn2')
    dot.edge('v4', 'cross_attn2')
    
    dot.edge('q3', 'cross_attn3')
    dot.edge('k1', 'cross_attn3')
    dot.edge('k2', 'cross_attn3')
    dot.edge('k3', 'cross_attn3')
    dot.edge('k4', 'cross_attn3')
    dot.edge('v1', 'cross_attn3')
    dot.edge('v2', 'cross_attn3')
    dot.edge('v3', 'cross_attn3')
    dot.edge('v4', 'cross_attn3')
    
    # 连接输出
    dot.edge('cross_attn1', 'out1')
    dot.edge('cross_attn2', 'out2')
    dot.edge('cross_attn3', 'out3')
    
    return dot

def create_multihead_attention_diagram():
    """创建多头注意力机制图"""
    dot = graphviz.Digraph(
        'Multihead_Attention',
        comment='多头注意力机制图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='14,12', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='11')
    
    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('input', '输入序列\nX', fillcolor='#E8F4FD', color='blue')
    
    # 线性变换
    with dot.subgraph(name='cluster_linear') as c:
        c.attr(label='线性变换', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('q_proj', 'Q投影\nXW^Q', fillcolor='#FFF2CC', color='orange')
        c.node('k_proj', 'K投影\nXW^K', fillcolor='#FFF2CC', color='orange')
        c.node('v_proj', 'V投影\nXW^V', fillcolor='#FFF2CC', color='orange')
    
    # 多头分割
    with dot.subgraph(name='cluster_heads') as c:
        c.attr(label='多头分割', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        c.node('head1', 'Head 1\n(Q₁,K₁,V₁)', fillcolor='#E1D5E7', color='purple')
        c.node('head2', 'Head 2\n(Q₂,K₂,V₂)', fillcolor='#E1D5E7', color='purple')
        c.node('head3', 'Head 3\n(Q₃,K₃,V₃)', fillcolor='#E1D5E7', color='purple')
        c.node('head4', 'Head 4\n(Q₄,K₄,V₄)', fillcolor='#E1D5E7', color='purple')
        c.node('head_n', '...\nHead h', fillcolor='#E1D5E7', color='purple')
    
    # 注意力计算
    with dot.subgraph(name='cluster_attention') as c:
        c.attr(label='注意力计算', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('attn1', 'Attention(Q₁,K₁,V₁)', fillcolor='#D5E8D4', color='green')
        c.node('attn2', 'Attention(Q₂,K₂,V₂)', fillcolor='#D5E8D4', color='green')
        c.node('attn3', 'Attention(Q₃,K₃,V₃)', fillcolor='#D5E8D4', color='green')
        c.node('attn4', 'Attention(Q₄,K₄,V₄)', fillcolor='#D5E8D4', color='green')
        c.node('attn_n', 'Attention(Qₕ,Kₕ,Vₕ)', fillcolor='#D5E8D4', color='green')
    
    # 拼接和输出投影
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='拼接和输出', style='filled', color='lightcoral', fontname='SimHei', fontsize='14')
        
        c.node('concat', '拼接\nConcat(head₁,...,headₕ)', fillcolor='#F8CECC', color='red')
        c.node('output_proj', '输出投影\nConcat(...)W^O', fillcolor='#F8CECC', color='red')
        c.node('output', '输出\nMultiHead(Q,K,V)', fillcolor='#F8CECC', color='red')
    
    # 连接
    dot.edge('input', 'q_proj')
    dot.edge('input', 'k_proj')
    dot.edge('input', 'v_proj')
    
    dot.edge('q_proj', 'head1')
    dot.edge('k_proj', 'head1')
    dot.edge('v_proj', 'head1')
    
    dot.edge('q_proj', 'head2')
    dot.edge('k_proj', 'head2')
    dot.edge('v_proj', 'head2')
    
    dot.edge('q_proj', 'head3')
    dot.edge('k_proj', 'head3')
    dot.edge('v_proj', 'head3')
    
    dot.edge('q_proj', 'head4')
    dot.edge('k_proj', 'head4')
    dot.edge('v_proj', 'head4')
    
    dot.edge('q_proj', 'head_n')
    dot.edge('k_proj', 'head_n')
    dot.edge('v_proj', 'head_n')
    
    dot.edge('head1', 'attn1')
    dot.edge('head2', 'attn2')
    dot.edge('head3', 'attn3')
    dot.edge('head4', 'attn4')
    dot.edge('head_n', 'attn_n')
    
    dot.edge('attn1', 'concat')
    dot.edge('attn2', 'concat')
    dot.edge('attn3', 'concat')
    dot.edge('attn4', 'concat')
    dot.edge('attn_n', 'concat')
    
    dot.edge('concat', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

def create_attention_formula_diagram():
    """创建注意力公式图解"""
    dot = graphviz.Digraph(
        'Attention_Formula',
        comment='注意力公式图解',
        format='png',
        engine='dot'
    )
    
    dot.attr(rankdir='TB', size='12,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
    
    # 输入
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入', style='filled', color='lightblue', fontname='SimHei', fontsize='14')
        
        c.node('q', 'Query (Q)\n查询向量', fillcolor='#E8F4FD', color='blue')
        c.node('k', 'Key (K)\n键向量', fillcolor='#E8F4FD', color='blue')
        c.node('v', 'Value (V)\n值向量', fillcolor='#E8F4FD', color='blue')
    
    # 计算步骤
    with dot.subgraph(name='cluster_steps') as c:
        c.attr(label='计算步骤', style='filled', color='lightyellow', fontname='SimHei', fontsize='14')
        
        c.node('step1', '步骤1: QK^T\n计算相似度', fillcolor='#FFF2CC', color='orange')
        c.node('step2', '步骤2: QK^T/√d_k\n缩放', fillcolor='#FFF2CC', color='orange')
        c.node('step3', '步骤3: softmax(QK^T/√d_k)\n归一化权重', fillcolor='#FFF2CC', color='orange')
        c.node('step4', '步骤4: softmax(...)V\n加权求和', fillcolor='#FFF2CC', color='orange')
    
    # 输出
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出', style='filled', color='lightgreen', fontname='SimHei', fontsize='14')
        
        c.node('output', '输出\nAttention(Q,K,V)', fillcolor='#D5E8D4', color='green')
    
    # 公式说明
    with dot.subgraph(name='cluster_formula') as c:
        c.attr(label='数学公式', style='filled', color='lightpink', fontname='SimHei', fontsize='14')
        
        c.node('formula', 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', fillcolor='#E1D5E7', color='purple')
        c.node('explanation', '其中:\n• d_k 是Key的维度\n• √d_k 用于缩放\n• softmax确保权重和为1', fillcolor='#E1D5E7', color='purple')
    
    # 连接
    dot.edge('q', 'step1')
    dot.edge('k', 'step1')
    dot.edge('step1', 'step2')
    dot.edge('step2', 'step3')
    dot.edge('step3', 'step4')
    dot.edge('v', 'step4')
    dot.edge('step4', 'output')
    
    dot.edge('formula', 'output')
    dot.edge('explanation', 'formula')
    
    return dot

def main():
    """主函数"""
    print("正在生成ACT注意力机制图解...")
    
    # 生成各个注意力机制图
    diagrams = [
        ('self_attention', create_self_attention_diagram()),
        ('cross_attention', create_cross_attention_diagram()),
        ('multihead_attention', create_multihead_attention_diagram()),
        ('attention_formula', create_attention_formula_diagram())
    ]
    
    for name, dot in diagrams:
        output_path = f'./images/act_{name}'
        dot.render(output_path, cleanup=True)
        print(f"已生成: {output_path}.png")
    
    print("\n所有注意力机制图解已生成完成！")
    print("包含以下注意力机制图：")
    print("1. 自注意力机制")
    print("2. 交叉注意力机制")
    print("3. 多头注意力机制")
    print("4. 注意力公式图解")

if __name__ == "__main__":
    main() 