#!/usr/bin/env python3
"""
注意力机制可视化脚本
展示SmolVLA中自注意力和交叉注意力的工作原理
"""

import graphviz

def create_attention_mechanism_diagram():
    """创建注意力机制图"""
    
    # 创建有向图
    dot = graphviz.Digraph('Attention_Mechanism', 
                          comment='注意力机制工作原理',
                          format='png')
    
    # 设置图形属性
    dot.attr(rankdir='TB', 
             size='14,12',
             dpi='300',
             fontname='SimHei',
             fontsize='16')
    
    # 设置节点默认属性
    dot.node_attr.update(shape='box',
                         style='filled',
                         fontname='SimHei',
                         fontsize='11',
                         height='0.5',
                         width='1.5')
    
    # 设置边默认属性
    dot.edge_attr.update(fontname='SimHei',
                         fontsize='9',
                         arrowsize='0.8')
    
    # 自注意力机制
    with dot.subgraph(name='cluster_self_attention') as self_attn:
        self_attn.attr(label='自注意力机制 (Self-Attention)', 
                      style='filled',
                      color='lightblue',
                      fontname='SimHei',
                      fontsize='14')
        
        # 输入序列
        self_attn.node('input1', '输入序列\n[token1, token2, token3]', 
                      fillcolor='lightcyan',
                      color='darkblue')
        
        # 线性变换
        self_attn.node('q_proj', 'Q投影\n(Query)', 
                      fillcolor='lightyellow',
                      color='darkorange')
        self_attn.node('k_proj', 'K投影\n(Key)', 
                      fillcolor='lightyellow',
                      color='darkorange')
        self_attn.node('v_proj', 'V投影\n(Value)', 
                      fillcolor='lightyellow',
                      color='darkorange')
        
        # 注意力计算
        self_attn.node('attention', '注意力计算\nAttention(Q,K,V)', 
                      fillcolor='lightgreen',
                      color='darkgreen')
        
        # 输出
        self_attn.node('output1', '输出序列\n[out1, out2, out3]', 
                      fillcolor='lightseagreen',
                      color='darkgreen')
        
        # 连接
        self_attn.edge('input1', 'q_proj')
        self_attn.edge('input1', 'k_proj')
        self_attn.edge('input1', 'v_proj')
        self_attn.edge('q_proj', 'attention')
        self_attn.edge('k_proj', 'attention')
        self_attn.edge('v_proj', 'attention')
        self_attn.edge('attention', 'output1')
    
    # 交叉注意力机制
    with dot.subgraph(name='cluster_cross_attention') as cross_attn:
        cross_attn.attr(label='交叉注意力机制 (Cross-Attention)', 
                       style='filled',
                       color='lightpink',
                       fontname='SimHei',
                       fontsize='14')
        
        # 两个输入序列
        cross_attn.node('query_seq', '查询序列\n(动作token)', 
                       fillcolor='lightpink',
                       color='darkred')
        cross_attn.node('key_value_seq', '键值序列\n(图像+文本特征)', 
                       fillcolor='lightcyan',
                       color='darkblue')
        
        # 投影
        cross_attn.node('q_proj2', 'Q投影\n(来自查询序列)', 
                       fillcolor='lightyellow',
                       color='darkorange')
        cross_attn.node('k_proj2', 'K投影\n(来自键值序列)', 
                       fillcolor='lightyellow',
                       color='darkorange')
        cross_attn.node('v_proj2', 'V投影\n(来自键值序列)', 
                       fillcolor='lightyellow',
                       color='darkorange')
        
        # 注意力计算
        cross_attn.node('cross_attention', '交叉注意力\nCrossAttention(Q,K,V)', 
                       fillcolor='lightgreen',
                       color='darkgreen')
        
        # 输出
        cross_attn.node('output2', '融合输出\n(动作+上下文)', 
                       fillcolor='lightseagreen',
                       color='darkgreen')
        
        # 连接
        cross_attn.edge('query_seq', 'q_proj2')
        cross_attn.edge('key_value_seq', 'k_proj2')
        cross_attn.edge('key_value_seq', 'v_proj2')
        cross_attn.edge('q_proj2', 'cross_attention')
        cross_attn.edge('k_proj2', 'cross_attention')
        cross_attn.edge('v_proj2', 'cross_attention')
        cross_attn.edge('cross_attention', 'output2')
    
    # 数学公式
    with dot.subgraph(name='cluster_math') as math:
        math.attr(label='注意力计算公式', 
                 style='filled',
                 color='lightgoldenrod',
                 fontname='SimHei',
                 fontsize='14')
        
        math.node('formula1', 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
        math.node('formula2', 'MultiHead = Concat(head_1,...,head_h)W^O', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
        math.node('formula3', 'head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)', 
                 fillcolor='lightgoldenrod',
                 color='darkgoldenrod')
    
    # 添加注释
    dot.node('note1', '自注意力：序列内部关联\n交叉注意力：序列间关联', 
             shape='note',
             fillcolor='lightyellow',
             color='darkorange')
    dot.node('note2', '在SmolVLA中：\n- 自注意力：处理图像/文本内部关系\n- 交叉注意力：动作token关注视觉-语言特征', 
             shape='note',
             fillcolor='lightyellow',
             color='darkorange')
    
    # 连接注释
    dot.edge('note1', 'attention', style='dashed', color='gray')
    dot.edge('note2', 'cross_attention', style='dashed', color='gray')
    
    return dot

def main():
    """主函数"""
    print("正在生成注意力机制图...")
    
    # 创建注意力机制图
    dot = create_attention_mechanism_diagram()
    
    # 保存图片
    output_path = './images/attention_mechanism'
    dot.render(output_path, cleanup=True)
    
    print(f"注意力机制图已保存到: {output_path}.png")
    print("注意力机制图生成完成！")

if __name__ == "__main__":
    main() 